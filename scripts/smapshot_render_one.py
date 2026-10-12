"""Rend la carte de distance d'une photo smapshot et superpose skyline / crêtes sur la photo.

Usage : python scripts/smapshot_render_one.py <image_id> <dossier_sortie> [largeur]
"""
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import dem_render as D  # noqa: E402
import smapshot as S  # noqa: E402
from geopose_labels import ridge_labels  # noqa: E402


def download_photo(attrs, width, out_path):
    url = attrs["media"]["image_url"].replace("/full/200,/", f"/full/{width},/")
    if not out_path.exists():
        urllib.request.urlretrieve(url, out_path)
    return Image.open(out_path).convert("RGB")


def main(image_id, out_dir, width=1024):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width = int(width)
    attrs = S.get(f"/images/{image_id}/attributes")
    cam = S.Camera.from_attributes(attrs)
    photo = download_photo(attrs, width, out_dir / f"{image_id}.jpg")
    t0 = time.time()
    dist, scene = D.render_camera(cam, photo.width, photo.height)
    t1 = time.time()
    print(f"{image_id} '{attrs['title']}' {photo.size} rendu {t1 - t0:.1f}s, niveaux MNT valides "
          f"{[round(L['valid'], 2) for L in scene.levels]}, ciel {np.mean(dist < 0):.2f}")
    skyline, ridge = ridge_labels(dist)
    a = np.asarray(photo).copy()
    a[ndimage.binary_dilation(ridge)] = (255, 200, 0)
    a[ndimage.binary_dilation(skyline)] = (255, 0, 0)
    Image.fromarray(a).save(out_dir / f"{image_id}_overlay.jpg", quality=90)
    np.savez_compressed(out_dir / f"{image_id}_distance.npz", distance=dist.astype(np.float32))


if __name__ == "__main__":
    main(*sys.argv[1:4])
