"""Évalue la skyline sur CH1 (ETH, 203 photos des Alpes suisses, masques ciel corrigés à la main).

Usage : python scripts/eval_ch1.py <dossier_CH1_extrait> <sortie> <ckpt1> [<ckpt2> ...]

Vérité terrain = ligne de ciel du masque manuel ; erreur par colonne en px et en degrés (focale
fournie). Baseline : Mask2Former ADE20K sans entraînement.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

sys.path.insert(0, str(Path(__file__).parent))
from eval_ridges import predict  # noqa: E402
from ridge_data import top_sky_rows  # noqa: E402
from train_ridges import load_trained  # noqa: E402

SEG_REPO = "facebook/mask2former-swin-large-ade-semantic"


def load_ch1(root):
    for meta in sorted(Path(root).glob("*/*.png.txt")):
        lines = meta.read_text().split()
        img_path = meta.with_suffix("")                       # x.png.txt -> x.png
        mask_path = img_path.with_name(img_path.stem + "-mask.png")
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        if mask[0].mean() < 0.5:                              # polarité : le ciel est en haut
            mask = ~mask
        yield img_path.stem, np.array(Image.open(img_path).convert("RGB")), mask, float(lines[0])


def column_errors(pred_sky, gt_sky):
    gt_rows, pred_rows = top_sky_rows(gt_sky), top_sky_rows(pred_sky)
    ok = (gt_rows > 0) & (gt_rows < gt_sky.shape[0])
    return np.abs(pred_rows - gt_rows)[ok]


def main(root, out_dir, *ckpts):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {Path(c).parent.name: load_trained(c, "cuda") for c in ckpts}
    processor = AutoImageProcessor.from_pretrained(SEG_REPO)
    seg = AutoModelForUniversalSegmentation.from_pretrained(SEG_REPO).to("cuda").eval()
    acc = {n: ([], []) for n in list(models) + ["mask2former"]}
    per_image = []
    for name, image, gt_sky, focal in load_ch1(root):
        deg = np.degrees(np.arctan(1 / focal))
        row = {"id": name}
        for mname, m in models.items():
            e = column_errors(predict(m, image, "cuda")[0] > 0.5, gt_sky)
            acc[mname][0].append(e)
            acc[mname][1].append(e * deg)
            row[mname] = float(np.median(e)) if e.size else None
        with torch.no_grad():
            inputs = processor(images=Image.fromarray(image), return_tensors="pt").to("cuda")
            lab = processor.post_process_semantic_segmentation(seg(**inputs), target_sizes=[image.shape[:2]])[0]
        e = column_errors(lab.cpu().numpy() == 2, gt_sky)
        acc["mask2former"][0].append(e)
        acc["mask2former"][1].append(e * deg)
        row["mask2former"] = float(np.median(e)) if e.size else None
        per_image.append(row)
    report = {}
    for n, (px, dg) in acc.items():
        e, d = np.concatenate(px), np.concatenate(dg)
        report[n] = {"columns": int(e.size), "median_px": float(np.median(e)), "mean_px": float(e.mean()),
                     "p90_px": float(np.percentile(e, 90)), "le2px": float((e <= 2).mean()),
                     "le5px": float((e <= 5).mean()), "median_deg": float(np.median(d)),
                     "mean_deg": float(d.mean()), "p90_deg": float(np.percentile(d, 90))}
        r = report[n]
        print(f"{n:28s} med {r['median_px']:.1f}px ({r['median_deg']:.3f}°) mean {r['mean_px']:.1f} p90 {r['p90_px']:.0f} "
              f"<=2px {r['le2px']:.3f} <=5px {r['le5px']:.3f}")
    report["n_images"] = len(per_image)
    json.dump(report, open(out_dir / "report_ch1.json", "w"), indent=2)
    json.dump(per_image, open(out_dir / "per_image_ch1.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
