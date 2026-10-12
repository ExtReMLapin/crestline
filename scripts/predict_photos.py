"""Applique le détecteur de crêtes entraîné à des photos (local uniquement).

Usage : python scripts/predict_photos.py <checkpoint.pt> <dossier_photos> <dossier_sortie> [cote_long_px]

Sortie par photo : <stem>.npz (probabilités ciel / skyline / crêtes à la résolution
d'inférence, skyline par colonne) et <stem>.jpg (superposition + zooms).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from eval_ridges import predict, thin  # noqa: E402
from ridge_data import top_sky_rows  # noqa: E402
from train_ridges import load_trained  # noqa: E402


def draw(image, prob, out_path, zoom=(512, 288), n_zooms=4, ridge_thr=0.5):
    a = np.asarray(image).copy()
    h, w = a.shape[:2]
    ridges = thin(prob[2], ridge_thr)
    a[ridges] = (0, 255, 255)
    rows = top_sky_rows(prob[0] > 0.5)
    cols = np.arange(w)
    for dy in (-1, 0, 1):
        a[np.clip(rows + dy, 0, h - 1), cols] = (0, 255, 0)
    vis = Image.fromarray(a)
    overview = vis.copy()
    overview.thumbnail((1400, 1400))
    xs = np.linspace(zoom[0] // 2, w - zoom[0] // 2, n_zooms).astype(int)
    strip = Image.new("RGB", (zoom[0] * 2, zoom[1] * ((n_zooms + 1) // 2)), "white")
    for i, cx in enumerate(xs):
        cy = int(np.median(rows[max(cx - 30, 0):cx + 30]))
        top = max(0, min(cy - zoom[1] // 3, h - zoom[1]))
        crop = vis.crop((cx - zoom[0] // 2, top, cx + zoom[0] // 2, top + zoom[1]))
        strip.paste(crop, ((i % 2) * zoom[0], (i // 2) * zoom[1]))
    sheet = Image.new("RGB", (max(overview.width, strip.width), overview.height + strip.height), "white")
    sheet.paste(overview, (0, 0))
    sheet.paste(strip, (0, overview.height))
    sheet.save(out_path, quality=88)


def main(ckpt, photo_dir, out_dir, long_side=1280):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model = load_trained(ckpt, device)
    thr_file = Path(ckpt).parent / "ridge_threshold.json"       # seuil réglé sur la validation, s'il existe
    ridge_thr = json.load(open(thr_file))["best"] if thr_file.exists() else 0.5
    for path in sorted(Path(photo_dir).glob("*.[jJ][pP][gG]")):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        full_w = image.width
        image.thumbnail((int(long_side), int(long_side)), Image.LANCZOS)
        prob = predict(model, np.asarray(image), device)
        rows = top_sky_rows(prob[0] > 0.5)
        np.savez_compressed(out_dir / f"{path.stem}.npz", prob=prob.astype(np.float16), skyline_rows=rows,
                            scale_to_full=full_w / image.width)
        draw(image, prob, out_dir / f"{path.stem}.jpg", ridge_thr=float(ridge_thr))
        print(path.stem, image.size, "ciel %.2f" % (prob[0] > 0.5).mean(),
              "px crêtes %d" % thin(prob[2], float(ridge_thr)).sum())


if __name__ == "__main__":
    main(*sys.argv[1:5])
