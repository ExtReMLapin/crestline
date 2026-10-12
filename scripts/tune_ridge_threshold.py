"""Choisit le seuil de la carte de crêtes qui maximise le F1 (tolérance 3 px) sur la validation smapshot.

Usage : python scripts/tune_ridge_threshold.py <liste_val.txt> <dossier_labels> <ckpt>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_ridges import predict, ridge_prf, thin  # noqa: E402
from train_ridges import load_trained  # noqa: E402

THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def main(val_list, labels_dir, ckpt):
    model = load_trained(ckpt, "cuda")
    acc = {t: np.zeros(4) for t in THRESHOLDS}
    for sid in [l.strip() for l in open(val_list) if l.strip()]:
        z = np.load(Path(labels_dir) / f"{sid}.npz")
        prob = predict(model, z["image"], "cuda")
        for t in THRESHOLDS:
            r = ridge_prf(thin(prob[2], t), z["ridge"], z["ignore"])
            if r is not None:
                acc[t] += r
    out = {}
    for t, a in acc.items():
        p, r = a[0] / max(a[1], 1), a[2] / max(a[3], 1)
        out[t] = {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}
        print(f"seuil {t:.2f}: P {p:.3f} R {r:.3f} F1 {out[t]['f1']:.3f}")
    best = max(out, key=lambda t: out[t]["f1"])
    print("meilleur seuil :", best)
    json.dump({"best": best, "sweep": out}, open(Path(ckpt).parent / "ridge_threshold.json", "w"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:4])
