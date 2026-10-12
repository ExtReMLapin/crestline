"""Comparaison qualitative de plusieurs modèles de crêtes sur les photos personnelles (local uniquement).

Usage : python scripts/compare_fiz.py <dossier_photos> <sortie> <largeur> <ckpt1> [<ckpt2> ...]

Par photo, une planche : une colonne par méthode (Mask2Former + chaque modèle), skyline en vert,
crêtes internes en cyan, plus une ligne de zooms 1:1 sur la skyline.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

sys.path.insert(0, str(Path(__file__).parent))
from eval_ridges import predict, thin  # noqa: E402
from ridge_data import top_sky_rows  # noqa: E402
from train_ridges import load_trained  # noqa: E402

SEG_REPO = "facebook/mask2former-swin-large-ade-semantic"
COL_W = 640


def draw(image, sky, ridges):
    a = image.copy()
    if ridges is not None:
        a[thin(ridges)] = (0, 255, 255)
    rows = top_sky_rows(sky)
    cols = np.arange(a.shape[1])
    for dy in (-1, 0, 1):
        a[np.clip(rows + dy, 0, a.shape[0] - 1), cols] = (0, 255, 0)
    return a, rows


def main(photo_dir, out_dir, width, *ckpts):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {Path(c).parent.name: load_trained(c, "cuda") for c in ckpts}
    processor = AutoImageProcessor.from_pretrained(SEG_REPO)
    seg = AutoModelForUniversalSegmentation.from_pretrained(SEG_REPO).to("cuda").eval()
    for path in sorted(Path(photo_dir).glob("*.[jJ][pP][gG]")):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((int(width), int(width)), Image.LANCZOS)
        image = np.asarray(im)
        panels = []
        with torch.no_grad():
            inputs = processor(images=im, return_tensors="pt").to("cuda")
            lab = processor.post_process_semantic_segmentation(seg(**inputs), target_sizes=[image.shape[:2]])[0]
        panels.append(("mask2former", *draw(image, lab.cpu().numpy() == 2, None)))
        for name, m in models.items():
            prob = predict(m, image, "cuda")
            panels.append((name, *draw(image, prob[0] > 0.5, prob[2])))
        h, w = image.shape[:2]
        scale = COL_W / w
        ph = int(h * scale)
        zoom_w, zoom_h = COL_W, 220
        # zoom 1:1 centré sur la skyline de la première méthode, au milieu de l'image
        rows0 = panels[0][2]
        cx = w // 2
        cy = int(np.median(rows0[max(cx - 40, 0):cx + 40]))
        top = max(0, min(cy - zoom_h // 2, h - zoom_h))
        sheet = Image.new("RGB", (len(panels) * (COL_W + 8), ph + zoom_h + 30), "white")
        d = ImageDraw.Draw(sheet)
        for i, (name, vis, _) in enumerate(panels):
            x = i * (COL_W + 8)
            sheet.paste(Image.fromarray(vis).resize((COL_W, ph)), (x, 22))
            sheet.paste(Image.fromarray(vis).crop((cx - zoom_w // 2, top, cx + zoom_w // 2, top + zoom_h)), (x, ph + 26))
            d.text((x + 4, 5), name, fill="black")
        sheet.save(out_dir / f"{path.stem}.jpg", quality=88)
    print("planches :", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
