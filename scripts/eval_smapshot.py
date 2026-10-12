"""Évalue des modèles de crêtes sur le test smapshot (étiquettes MNT suisse 2 m, zones jamais vues).

Usage : python scripts/eval_smapshot.py <liste_test.txt> <dossier_labels> <sortie> <ckpt1> [<ckpt2> ...]

Pour chaque modèle (et la baseline Mask2Former, dont la segmentation est stockée avec les étiquettes) :
- skyline : |ligne prédite - ligne MNT| par colonne, sur les colonnes dont le bord n'est pas ignoré ;
- crêtes internes : précision / rappel / F1 à 3 px, hors zones ignorées.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import smapshot as S  # noqa: E402
from eval_ridges import predict, ridge_prf, thin  # noqa: E402
from ridge_data import top_sky_rows  # noqa: E402
from train_ridges import load_trained  # noqa: E402

SKY_ID = 2


def skyline_cols(pred_sky, gt_sky, ignore):
    gt_rows, pred_rows = top_sky_rows(gt_sky), top_sky_rows(pred_sky)
    h = gt_sky.shape[0]
    cols = np.arange(gt_sky.shape[1])
    ok = (gt_rows > 0) & (gt_rows < h)
    r = np.clip(gt_rows, 0, h - 1)
    ok &= ~ignore[r, cols] & ~ignore[np.maximum(r - 1, 0), cols]
    return np.abs(pred_rows - gt_rows)[ok]


def deg_per_px(image_id, width, cache):
    if image_id not in cache:
        a = S.get(f"/images/{image_id}/attributes")
        cache[image_id] = float(np.degrees(np.arctan(1.0 / (a["pose"]["focal"] * width / a["width"]))))
    return cache[image_id]


def summarize(px, deg, prf):
    e, d = np.concatenate(px), np.concatenate(deg)
    p, r = prf[0] / max(prf[1], 1), prf[2] / max(prf[3], 1)
    return {"columns": int(e.size), "median_px": float(np.median(e)), "mean_px": float(e.mean()),
            "p90_px": float(np.percentile(e, 90)), "le2px": float((e <= 2).mean()), "le5px": float((e <= 5).mean()),
            "median_deg": float(np.median(d)), "mean_deg": float(d.mean()), "p90_deg": float(np.percentile(d, 90)),
            "ridge_precision": p, "ridge_recall": r, "ridge_f1": 2 * p * r / max(p + r, 1e-9)}


def main(test_list, labels_dir, out_dir, *ckpts):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "deg_per_px.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    ids = [l.strip() for l in open(test_list) if l.strip()]
    models = {Path(c).parent.name: load_trained(c, "cuda") for c in ckpts}
    acc = {name: ([], [], np.zeros(4)) for name in list(models) + ["mask2former"]}
    for n, sid in enumerate(ids):
        z = np.load(Path(labels_dir) / f"{sid}.npz")
        image, gt_sky, ridge, ignore = z["image"], z["sky"], z["ridge"], z["ignore"]
        dpp = deg_per_px(sid, image.shape[1], cache)
        preds = {name: predict(m, image, "cuda") for name, m in models.items()}
        for name, prob in preds.items():
            e = skyline_cols(prob[0] > 0.5, gt_sky, ignore)
            acc[name][0].append(e)
            acc[name][1].append(e * dpp)
            r = ridge_prf(thin(prob[2]), ridge, ignore)
            if r is not None:
                acc[name][2][:] += r
        e = skyline_cols(z["seg"] == SKY_ID, gt_sky, ignore)
        acc["mask2former"][0].append(e)
        acc["mask2former"][1].append(e * dpp)
        if n % 200 == 0:
            print(f"{n}/{len(ids)}", flush=True)
            json.dump(cache, open(cache_path, "w"))
    json.dump(cache, open(cache_path, "w"))
    report = {name: summarize(*v) for name, v in acc.items()}
    report["mask2former"].update({"ridge_precision": None, "ridge_recall": None, "ridge_f1": None})
    report["n_images"] = len(ids)
    json.dump(report, open(out_dir / "report.json", "w"), indent=2)
    for name, r in report.items():
        if name != "n_images":
            print(f"{name:28s} med {r['median_px']:.1f}px ({r['median_deg']:.3f}°) mean {r['mean_px']:.1f} "
                  f"p90 {r['p90_px']:.0f} <=2px {r['le2px']:.3f} <=5px {r['le5px']:.3f} | crêtes F1 {r['ridge_f1']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
