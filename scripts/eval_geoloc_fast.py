"""Évaluation rapide (index de mots de crête) sur CH1 ou une liste smapshot, pour plusieurs réglages du
filtre de colonnes fiables à la fois (la segmentation n'est calculée qu'une fois par photo).

Usage : python scripts/eval_geoloc_fast.py <ch1|crosslocate|liste_smapshot> <sortie> <conf:netteté,...> [n_photos]

crosslocate : test du jeu « Uniform » de CrossLocate (3 513 photos en Suisse), voir scripts/crosslocate_meta.py.

En parallèle sur plusieurs GPU : SHARD=i/N (photo n° k traitée si k % N == i), rapport dans <sortie>/shard_i,
puis fusion : python scripts/eval_geoloc_fast.py merge <sortie>
Progression : <sortie>/shard_i/progress.json (photos faites, total).

Exemple : ... data/smapshot/lists/val.txt outputs/x 0:0,0.5:0.8,0.6:0.9 200
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dem_render import TO_LV95, lv95_rotation  # noqa: E402
from eval_ch1 import load_ch1  # noqa: E402
from eval_geoloc_smapshot import MAX_ABOVE_GROUND, poses  # noqa: E402
from eval_ridges import predict  # noqa: E402
from geolocate import Refiner, localize_indexed, reliable_columns, skyline_rows  # noqa: E402
from horizon_db import REGIO, load_many  # noqa: E402
from ridge_index import ShardedIndex  # noqa: E402
from train_ridges import load_trained  # noqa: E402

CKPT = "outputs/smap_S1_more/best.pt"
PREFIX = os.environ.get("GEOLOC_PREFIX", "")         # ex. "cop" : base et index construits sur Copernicus
DBS = [f"data/horizon/{PREFIX}ch125_band{i}" for i in range(3)]
INDEXES = [f"data/horizon/{PREFIX}index125_band{i}" for i in range(3)]


def ch1_samples(root="data/ch1/extracted"):
    metas = {m.with_suffix("").stem: m.read_text().split() for m in Path(root).glob("*/*.png.txt")}
    for name, image, _, focal in load_ch1(root):
        e, n = TO_LV95.transform(float(metas[name][2]), float(metas[name][1]))
        yield name, image, focal, e, n, None


def smapshot_samples(id_list, photo_dir="data/smapshot/out/photos"):
    ids = Path(id_list).read_text().split()
    cams = poses(ids)
    regio = rasterio.open(REGIO)
    for i in ids:
        p = cams[i]["pose"]
        e, n = TO_LV95.transform(p["longitude"], p["latitude"])
        if not p["altitude"] - float(next(regio.sample([(e, n)]))[0]) < MAX_ABOVE_GROUND:
            continue
        image = np.asarray(Image.open(Path(photo_dir) / f"{i}.jpg").convert("RGB"))
        heading = p["azimuth"] + np.degrees(lv95_rotation(p["latitude"], p["longitude"]))   # cap grille
        yield i, image, p["focal"] * image.shape[1] / cams[i]["width"], e, n, heading


def crosslocate_samples(meta="data/crosslocate/test_queries.json", photo_dir="data/crosslocate/query_photos"):
    """Le cap de CrossLocate (yaw, radians) suit sa propre convention : il est rangé tel quel dans le rapport
    (champ cl_yaw) et comparé après coup. Focale tirée du champ de vision horizontal fourni."""
    for q in json.load(open(meta)):
        path = Path(photo_dir) / q["fn"]
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((1536, 1536), Image.LANCZOS)      # comme crestline.model.prepare (originaux jusqu'à 7 360 px)
        image = np.asarray(im)
        e, n = TO_LV95.transform(q["lon"], q["lat"])
        focal = image.shape[1] / 2 / np.tan(q["hfov"] / 2)
        yield q["fn"], image, focal, e, n, {"cl_yaw": q["yaw"]}


def prefetch(gen, depth=4):
    """Décode les photos suivantes dans un fil à part pendant que le GPU travaille."""
    import queue
    import threading
    q, end = queue.Queue(depth), object()

    def work():
        for x in gen:
            q.put(x)
        q.put(end)
    threading.Thread(target=work, daemon=True).start()
    while (x := q.get()) is not end:
        yield x


def summarize(res):
    """res : {réglage : [résultat par photo]} -> taux de réussite à 1 km (top 1, 5, 20) et temps médian."""
    summary = {}
    for s, rs in res.items():
        first = [r["first_hit"] for r in rs]
        summary[s] = {"n": len(rs), **{f"recall@{k}": float(np.mean([f is not None and f <= k for f in first]))
                                      for k in (1, 5, 20)},
                      "median_seconds": float(np.median([r["seconds"] for r in rs]))}
    return summary


def merge(out_dir):
    """Fusionne les rapports des morceaux <out_dir>/shard_*/report.json."""
    out_dir = Path(out_dir)
    res = {}
    for f in sorted(out_dir.glob("shard_*/report.json")):
        for s, rs in json.load(open(f))["images"].items():
            res.setdefault(s, []).extend(rs)
    summary = summarize(res)
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "images": res}, open(out_dir / "report.json", "w"), indent=1)


def main(dataset, out_dir, settings, n_photos=None):
    out_dir = Path(out_dir)
    shard, n_shards = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    if n_shards > 1:
        out_dir = out_dir / f"shard_{shard}"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = [tuple(float(x) for x in s.split(":")) for s in settings.split(",")]
    points, elev = load_many(DBS)
    index = ShardedIndex(INDEXES)
    if os.environ.get("GEOLOC_GPU_INDEX", "1") == "1":   # index sur GPU (4,6 Go), mêmes résultats, ~3,5x plus rapide
        index.to_gpu("cuda")
    refiner = Refiner((*points[:, :2].min(0), *points[:, :2].max(0)))
    model = load_trained(CKPT, "cuda")
    samples = (ch1_samples() if dataset == "ch1" else crosslocate_samples() if dataset == "crosslocate"
               else smapshot_samples(dataset))
    res = {s: [] for s in settings}
    total = {"ch1": 203, "crosslocate": 3513}.get(dataset) or len(Path(dataset).read_text().split())
    total = (total - shard + n_shards - 1) // n_shards
    samples = (x for k, x in enumerate(samples) if k % n_shards == shard)
    done = 0
    for count, (name, image, focal, e, n, heading) in enumerate(prefetch(samples)):
        if n_photos and count >= int(n_photos):
            break
        done += 1
        (out_dir / "progress.json").write_text(json.dumps({"dataset": dataset, "done": done, "total": total}))
        prob = predict(model, image, "cuda")
        rows = skyline_rows(prob[0] > 0.5)
        line = []
        for s in settings:
            t = time.time()
            cands, diag = localize_indexed(index, elev, points, refiner, prob[0] > 0.5, focal, truth=(e, n),
                                           col_ok=reliable_columns(prob, rows, *s))
            d = [np.hypot(c["e"] - e, c["n"] - n) for c in cands] or [np.inf]
            hit = next((k + 1 for k, x in enumerate(d) if x < 1000), None)
            r = {"id": name, "top1_m": float(d[0]), "first_hit": hit, "seconds": time.time() - t, **diag}
            if isinstance(heading, dict):
                r.update(heading)
                if cands:
                    r["heading_grid"] = cands[0]["heading_grid"]
            elif heading is not None and cands:
                r["heading_err"] = float(abs((cands[0]["heading_grid"] - heading + 180) % 360 - 180))
            res[s].append(r)
            line.append(f"{s}:{hit}")
        print(name, " ".join(line), flush=True)
    res = {":".join(f"{x:g}" for x in s): rs for s, rs in res.items()}
    summary = summarize(res)
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "images": res}, open(out_dir / "report.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2])
    else:
        main(*sys.argv[1:])
