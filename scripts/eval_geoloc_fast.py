"""Évaluation rapide (index de mots de crête) sur CH1 ou une liste smapshot, pour plusieurs réglages du
filtre de colonnes fiables à la fois (la segmentation n'est calculée qu'une fois par photo).

Usage : python scripts/eval_geoloc_fast.py <ch1|liste_smapshot> <sortie> <conf:netteté,...> [n_photos]

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


def main(dataset, out_dir, settings, n_photos=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = [tuple(float(x) for x in s.split(":")) for s in settings.split(",")]
    points, elev = load_many(DBS)
    index = ShardedIndex(INDEXES)
    refiner = Refiner((*points[:, :2].min(0), *points[:, :2].max(0)))
    model = load_trained(CKPT, "cuda")
    samples = ch1_samples() if dataset == "ch1" else smapshot_samples(dataset)
    res = {s: [] for s in settings}
    for count, (name, image, focal, e, n, heading) in enumerate(samples):
        if n_photos and count >= int(n_photos):
            break
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
            if heading is not None and cands:
                r["heading_err"] = float(abs((cands[0]["heading_grid"] - heading + 180) % 360 - 180))
            res[s].append(r)
            line.append(f"{s}:{hit}")
        print(name, " ".join(line), flush=True)
    summary = {}
    for s, rs in res.items():
        first = [r["first_hit"] for r in rs]
        summary[":".join(f"{x:g}" for x in s)] = {"n": len(rs), **{f"recall@{k}": float(np.mean([f is not None and f <= k for f in first]))
                                                     for k in (1, 5, 20)},
                                     "median_seconds": float(np.median([r["seconds"] for r in rs]))}
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "images": {":".join(f"{x:g}" for x in s): rs for s, rs in res.items()}},
              open(out_dir / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
