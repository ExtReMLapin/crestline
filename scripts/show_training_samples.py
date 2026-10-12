"""Planches montrant ce que reçoit le réseau à l'entraînement (avec la source MNT pour comprendre).

Usage : python scripts/show_training_samples.py <dossier_sortie> [n]

Par exemple : 1 ligne = 1 photo GeoPose3K.
  [photo d'origine] [distance MNT rendue] | [entrée réseau] [cible ciel] [cible skyline] [cible crêtes] [masque ignoré]
Les 5 dernières vignettes sortent du vrai DataLoader (augmentations comprises).
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from geopose_labels import read_pfm_gz  # noqa: E402
from ridge_data import RidgeDataset, load_meta, read_split  # noqa: E402

EXTRACTED = Path("data/geopose3k/extracted/geoPose3K_final_publish")
TILE = 300


def tile(img, title):
    img = img.convert("RGB")
    img.thumbnail((TILE, TILE))
    canvas = Image.new("RGB", (TILE, TILE + 22), "white")
    canvas.paste(img, ((TILE - img.width) // 2, 22 + (TILE - img.height) // 2))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def distance_vis(distance):
    """Distance en couleurs (proche = jaune, loin = violet), ciel en blanc."""
    sky = distance <= 0
    distance = np.nan_to_num(distance, nan=0.0)  # quelques pixels NaN dans certains rendus
    d = np.log(np.maximum(distance, 1.0))
    lo, hi = np.percentile(d[~sky], [1, 99]) if (~sky).any() else (0, 1)
    rgb = (matplotlib.colormaps["viridis_r"](np.clip((d - lo) / (hi - lo + 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[sky] = 255
    return Image.fromarray(rgb)


def superpose(image, sky, skyline, ridge, ignore):
    """Cibles dessinées sur l'entrée : bord du ciel en rouge, crêtes en jaune, zones ignorées assombries."""
    a = image.copy()
    a[ignore > 127] //= 3
    a[ridge > 160] = (255, 200, 0)
    a[skyline > 160] = (255, 0, 0)
    return a


def gray(a):
    return Image.fromarray(np.asarray(a, np.uint8))


def main(out_dir, n=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta("data/geopose_prep")
    ids = [i for i in read_split("data/geopose3k/splits", "train") if meta.sky_iou.get(i, 0) >= 0.9]
    rng = np.random.default_rng(3)
    chosen = list(rng.choice(ids, int(n), replace=False))
    ds = RidgeDataset("data/geopose_cache", chosen, train=True)
    np.random.seed(0)
    rows = []
    for k, sid in enumerate(chosen):
        photo = Image.open(next((EXTRACTED / sid).glob("photo.*")))
        distance = read_pfm_gz(EXTRACTED / sid / "pinhole/distance_crop.pfm.gz")
        image, targets = ds[k]
        image = image.permute(1, 2, 0).numpy()
        sky, skyline, ridge, ignore = targets.numpy()
        tiles = [
            tile(photo, "1. photo d'origine"),
            tile(distance_vis(distance), "2. distance MNT (ciel = blanc)"),
            tile(Image.fromarray(image), "3. ENTREE reseau (augmentee)"),
            tile(gray(sky), "4. cible ciel"),
            tile(gray(skyline), "5. cible skyline"),
            tile(gray(ridge), "6. cible cretes internes"),
            tile(gray(ignore), "7. ignore (blanc = hors perte)"),
            tile(Image.fromarray(superpose(image, sky, skyline, ridge, ignore)), "8. superposition sur l'entree"),
        ]
        row = Image.new("RGB", (TILE * len(tiles) + 20, TILE + 22), "white")
        for i, t in enumerate(tiles):
            row.paste(t, (i * TILE + (20 if i >= 2 else 0), 0))
        ImageDraw.Draw(row).line([(2 * TILE + 10, 0), (2 * TILE + 10, TILE + 22)], fill="red", width=3)
        row.save(out_dir / f"{k:02d}_{sid}.jpg", quality=90)
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out_dir / "planche.jpg", quality=88)
    print(out_dir / "planche.jpg", sheet.size)


if __name__ == "__main__":
    main(*sys.argv[1:3])
