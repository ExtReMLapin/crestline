"""Rappel de la présélection par l'index de mots de crête sur CH1.

Usage : python scripts/eval_index_ch1.py <dossier_CH1_extrait> <base> <index> <k1,k2,...>

Pour chaque photo dont la vraie position est dans la base : la vraie position (à moins de 1 km) est-elle
parmi les k premiers lieux distincts proposés par l'index ? Affiche aussi le temps de requête.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dem_render import TO_LV95  # noqa: E402
from eval_ch1 import load_ch1  # noqa: E402
from eval_ridges import predict  # noqa: E402
from geolocate import PITCHES, photo_profiles, skyline_rows  # noqa: E402
from horizon_db import load as load_db  # noqa: E402
from ridge_index import RidgeIndex  # noqa: E402
from train_ridges import load_trained  # noqa: E402


def main(root, db, index_dir, ks):
    ks = [int(k) for k in ks.split(",")]
    pts = load_db(db)[0]
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    index = RidgeIndex.load(index_dir)
    model = load_trained("outputs/smap_S1_more/best.pt", "cuda")
    metas = {m.with_suffix("").stem: m.read_text().split() for m in Path(root).glob("*/*.png.txt")}
    ranks, times = [], []
    for name, image, _, focal in load_ch1(root):
        e, n = TO_LV95.transform(float(metas[name][2]), float(metas[name][1]))
        if not (lo[0] < e < hi[0] and lo[1] < n < hi[1]):
            continue
        sky = predict(model, image, "cuda")[0] > 0.5
        P, M, k_center = photo_profiles(skyline_rows(sky), sky.shape[0], focal)
        j = int(np.argmin(np.abs(PITCHES)))                    # hypothèse de tangage nul pour l'instant
        torch.cuda.synchronize()
        t = time.time()
        cand, heads, votes = index.query(P[j], M[j], k_center, k=20000)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        _, first = np.unique(cand, return_index=True)          # lieux distincts, dans l'ordre des votes
        uniq = cand[np.sort(first)]
        d = np.hypot(pts[uniq, 0] - e, pts[uniq, 1] - n)
        hit = np.nonzero(d < 1000)[0]
        ranks.append(int(hit[0]) + 1 if len(hit) else None)
        print(f"{name}: rang {ranks[-1]}  ({times[-1] * 1000:.0f} ms)", flush=True)
    for k in ks:
        print(f"rappel@{k} : {np.mean([r is not None and r <= k for r in ranks]):.3f}")
    print(f"{len(ranks)} photos, requête médiane {np.median(times) * 1000:.0f} ms")


if __name__ == "__main__":
    main(*sys.argv[1:])
