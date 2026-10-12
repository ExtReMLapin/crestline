"""Étiquettes de crêtes GeoPose3K à partir de la distance rendue depuis le MNT.

Pour une photo GeoPose3K (rendu « pinhole », aligné sur la photo) :
- skyline        : pixels de terrain dont un voisin est du ciel (distance < 0) ;
- crête interne  : pixels de terrain dont un voisin est nettement plus lointain
                   (saut de log-distance), c'est-à-dire une arête qui masque
                   du relief situé derrière.
"""
import gzip
from pathlib import Path

import numpy as np
from PIL import Image

RIDGE_LOG_JUMP = np.log(1.25)   # le relief masqué est au moins 25 % plus loin
RIDGE_MIN_JUMP_M = 150.0        # et au moins 150 m plus loin


def read_pfm_gz(path):
    with gzip.open(path, "rb") as f:
        header = f.readline().strip()
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
    channels = 3 if header == b"PF" else 1
    return np.flipud(data.reshape(height, width, channels))[..., 0].copy()


def read_info(path):
    """info.txt : mode d'alignement, lacet/tangage/roulis (rad), lat, lon, alt, fov (rad)."""
    lines = open(path).read().split()
    yaw, pitch, roll = map(float, lines[1:4])
    lat, lon, alt, fov = map(float, lines[4:8])
    return {"mode": lines[0], "yaw": yaw, "pitch": pitch, "roll": roll,
            "lat": lat, "lon": lon, "alt": alt, "fov": fov}


def neighbours(a, fill):
    """Voisins haut, bas, gauche, droite d'un tableau 2D (bord rempli par `fill`)."""
    p = np.pad(a, 1, constant_values=fill)
    return [p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]


def ridge_labels(distance):
    """Renvoie (skyline, ridge) en booléens, à la résolution du rendu."""
    sky = distance <= 0
    terrain = ~sky
    skyline = terrain & np.any(neighbours(sky, False), axis=0)
    d = np.where(sky, np.inf, distance)
    logd = np.log(np.maximum(d, 1.0))
    ridge = np.zeros_like(terrain)
    for nd, nlog in zip(neighbours(d, np.nan), neighbours(logd, np.nan)):
        with np.errstate(invalid="ignore"):  # inf - inf entre deux pixels de ciel
            farther = (nlog - logd > RIDGE_LOG_JUMP) & (nd - d > RIDGE_MIN_JUMP_M) & np.isfinite(nd)
        ridge |= farther
    return skyline, ridge & terrain & ~skyline


def load_sample(sample_dir):
    distance = read_pfm_gz(f"{sample_dir}/pinhole/distance_crop.pfm.gz")
    h, w = distance.shape
    photo_path = next(Path(sample_dir).glob("photo.*"))  # photo.jpeg ou photo.jpg selon la source
    photo = Image.open(photo_path).convert("RGB").resize((w, h), Image.BILINEAR)
    return photo, distance, read_info(f"{sample_dir}/info.txt")


def overlay(photo, skyline, ridge):
    a = np.asarray(photo).copy()
    a[ridge] = (255, 200, 0)
    a[skyline] = (255, 0, 0)
    return Image.fromarray(a)


if __name__ == "__main__":
    import sys
    sample, out = sys.argv[1], sys.argv[2]
    photo, distance, info = load_sample(sample)
    skyline, ridge = ridge_labels(distance)
    print(info, "skyline px:", int(skyline.sum()), "ridge px:", int(ridge.sum()))
    overlay(photo, skyline, ridge).save(out)
