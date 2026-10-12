"""Évalue le détecteur de crêtes sur le test GeoPose3K, contre la baseline Mask2Former (sans entraînement).

Usage : python scripts/eval_ridges.py <checkpoint.pt> <dossier_sortie> [split]

Mesures, sur les colonnes / pixels non masqués par un occultant :
- skyline : |ligne prédite - ligne MNT| par colonne, en pixels et en degrés ;
- crêtes internes : précision / rappel / F1 avec une tolérance de 3 px.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

sys.path.insert(0, str(Path(__file__).parent))
from ridge_data import load_meta, normalize, read_split, top_sky_rows  # noqa: E402
from train_ridges import load_trained  # noqa: E402

PREP, SPLITS = "data/geopose_prep", "data/geopose3k/splits"
RIDGE_TOL_PX = 3
SEG_REPO = "facebook/mask2former-swin-large-ade-semantic"


@torch.no_grad()
def predict(model, image, device):
    """Image (H, W, 3) uint8 -> probabilités (3, H, W) : ciel, skyline, crêtes."""
    h, w = image.shape[:2]
    x = torch.from_numpy(normalize(image)).permute(2, 0, 1)[None].to(device)
    ph, pw = (-h) % 32, (-w) % 32
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    return torch.sigmoid(logits.float())[0, :, :h, :w].cpu().numpy()


def skyline_errors(pred_sky, gt_sky, ignore):
    """Erreur absolue par colonne évaluable (ciel en haut, bord non masqué)."""
    gt_rows, pred_rows = top_sky_rows(gt_sky), top_sky_rows(pred_sky)
    h = gt_sky.shape[0]
    cols = np.arange(gt_sky.shape[1])
    ok = (gt_rows > 0) & (gt_rows < h)
    r = np.clip(gt_rows, 0, h - 1)
    ok &= ~ignore[r, cols] & ~ignore[np.maximum(r - 1, 0), cols]
    return np.abs(pred_rows - gt_rows)[ok]


def ridge_prf(pred, gt, ignore):
    pred, gt = pred & ~ignore, gt & ~ignore
    if not gt.any() and not pred.any():
        return None
    d_gt = ndimage.distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf)
    d_pred = ndimage.distance_transform_edt(~pred) if pred.any() else np.full(gt.shape, np.inf)
    tp_p = (d_gt[pred] <= RIDGE_TOL_PX).sum()
    tp_r = (d_pred[gt] <= RIDGE_TOL_PX).sum()
    return tp_p, pred.sum(), tp_r, gt.sum()


def thin(heat, thr=0.5):
    """Binarise une carte de lignes et l'amincit (maxima verticaux locaux)."""
    up = np.pad(heat, ((1, 0), (0, 0)), mode="edge")[:-1]
    down = np.pad(heat, ((0, 1), (0, 0)), mode="edge")[1:]
    return (heat > thr) & (heat >= up) & (heat >= down)


def summarize(errors_px, errors_deg):
    e, d = np.concatenate(errors_px), np.concatenate(errors_deg)
    return {"columns": int(e.size), "mean_px": float(e.mean()), "median_px": float(np.median(e)),
            "p90_px": float(np.percentile(e, 90)), "le2px": float((e <= 2).mean()),
            "le5px": float((e <= 5).mean()), "mean_deg": float(d.mean()), "median_deg": float(np.median(d)),
            "p90_deg": float(np.percentile(d, 90))}


def main(ckpt, out_dir, split="test"):
    out_dir = Path(out_dir)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model = load_trained(ckpt, device)
    processor = AutoImageProcessor.from_pretrained(SEG_REPO)
    seg_model = AutoModelForUniversalSegmentation.from_pretrained(SEG_REPO).to(device).eval()
    sky_id = {v: k for k, v in seg_model.config.id2label.items()}["sky"]

    meta = load_meta(PREP)
    ids = [i for i in read_split(SPLITS, split) if i in meta.index]
    res = {"model": ([], []), "mask2former": ([], [])}
    good = []  # photos bien recalées (sky_iou >= 0.9) : étiquettes plus fiables
    prf = np.zeros(4)
    per_image = []
    for n, sid in enumerate(ids):
        z = np.load(f"{PREP}/{sid}.npz")
        image, gt_sky, ridge, ignore = z["image"], z["sky"], z["ridge"], z["ignore"]
        deg_per_px = np.degrees(meta.loc[sid, "fov"]) / image.shape[1]
        prob = predict(model, image, device)
        with torch.no_grad():
            inputs = processor(images=Image.fromarray(image), return_tensors="pt").to(device)
            seg = processor.post_process_semantic_segmentation(
                seg_model(**inputs), target_sizes=[image.shape[:2]])[0].cpu().numpy()
        good.append(meta.loc[sid, "sky_iou"] >= 0.9)
        row = {"id": sid, "sky_iou": float(meta.loc[sid, "sky_iou"]), "mode": meta.loc[sid, "mode"]}
        for name, sky in (("model", prob[0] > 0.5), ("mask2former", seg == sky_id)):
            e = skyline_errors(sky, gt_sky, ignore)
            res[name][0].append(e)
            res[name][1].append(e * deg_per_px)
            row[f"{name}_median_px"] = float(np.median(e)) if e.size else None
        r = ridge_prf(thin(prob[2]), ridge, ignore)
        if r is not None:
            prf += r
        per_image.append(row)
        if n < 24:
            vis = image.copy()
            vis[ridge & ~ignore] = (255, 200, 0)
            vis[thin(prob[2])] = (0, 255, 255)
            rows = top_sky_rows(prob[0] > 0.5)
            gt_rows = top_sky_rows(gt_sky)
            cols = np.arange(image.shape[1])
            vis[np.clip(gt_rows, 0, image.shape[0] - 1), cols] = (255, 0, 0)
            vis[np.clip(rows, 0, image.shape[0] - 1), cols] = (0, 255, 0)
            vis[ignore] = (vis[ignore] * 0.5).astype(np.uint8)
            Image.fromarray(vis).save(out_dir / "vis" / f"{sid}.jpg", quality=90)
        if n % 100 == 0:
            print(f"{n}/{len(ids)}", flush=True)

    precision, recall = prf[0] / max(prf[1], 1), prf[2] / max(prf[3], 1)
    report = {name: summarize(*v) for name, v in res.items()}
    for name, (px, deg) in res.items():
        report[f"{name}_aligned"] = summarize([e for e, g in zip(px, good) if g],
                                              [e for e, g in zip(deg, good) if g])
    report["ridges"] = {"precision": precision, "recall": recall,
                        "f1": 2 * precision * recall / max(precision + recall, 1e-9), "tol_px": RIDGE_TOL_PX}
    report["n_images"] = len(ids)
    json.dump(report, open(out_dir / f"report_{split}.json", "w"), indent=2)
    json.dump(per_image, open(out_dir / f"per_image_{split}.json", "w"), indent=1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
