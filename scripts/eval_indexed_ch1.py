"""Géolocalisation CH1 avec l'index de mots de crête (au lieu de la recherche exhaustive).

Usage : python scripts/eval_indexed_ch1.py <dossier_CH1_extrait> <sortie> <ckpt> <n_lieux> <bases,...> <index,...>

Mêmes mesures que eval_geoloc_ch1.py, plus : la vraie position est-elle dans la présélection, et le temps
de recherche par photo (hors réseau de segmentation).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dem_render import TO_LV95  # noqa: E402
from eval_ch1 import load_ch1  # noqa: E402
from eval_ridges import predict  # noqa: E402
from geolocate import Refiner, localize_indexed  # noqa: E402
from horizon_db import load_many  # noqa: E402
from ridge_index import ShardedIndex  # noqa: E402
from train_ridges import load_trained  # noqa: E402


def main(root, out_dir, ckpt, n_places, dbs, indexes):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points, elev = load_many(dbs.split(","))
    index = ShardedIndex(indexes.split(","))
    refiner = Refiner((*points[:, :2].min(0), *points[:, :2].max(0)))
    model = load_trained(ckpt, "cuda")
    metas = {m.with_suffix("").stem: m.read_text().split() for m in Path(root).glob("*/*.png.txt")}
    results = []
    for name, image, _, focal in load_ch1(root):
        e, n = TO_LV95.transform(float(metas[name][2]), float(metas[name][1]))
        sky = predict(model, image, "cuda")[0] > 0.5
        torch.cuda.synchronize()
        t = time.time()
        cands, diag = localize_indexed(index, elev, points, refiner, sky, focal, int(n_places), truth=(e, n))
        torch.cuda.synchronize()
        dt = time.time() - t
        d = [np.hypot(c["e"] - e, c["n"] - n) for c in cands] or [np.inf]
        hit = next((k + 1 for k, x in enumerate(d) if x < 1000), None)
        results.append({"id": name, "top1_m": float(d[0]), "first_hit": hit, "seconds": dt, **diag, "top": cands})
        print(f"{name}: top1 {d[0] / 1000:.1f} km, 1er bon {hit}, présélection {diag.get('in_shortlist')}, {dt:.2f} s",
              flush=True)
    n_img = len(results)
    first = [r["first_hit"] for r in results]
    summary = {"n_images": n_img,
               **{f"recall@{k}": sum(1 for f in first if f is not None and f <= k) / n_img for k in (1, 5, 20)},
               "shortlist_recall": float(np.mean([r.get("in_shortlist", False) for r in results])),
               "median_top1_km": float(np.median([r["top1_m"] for r in results]) / 1000),
               "median_seconds": float(np.median([r["seconds"] for r in results]))}
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "images": results}, open(out_dir / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
