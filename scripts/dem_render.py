"""Rendu de cartes de distance depuis un MNT suisse, par lancer de rayons sur GPU.

- Repère : LV95 (EPSG:2056, mètres) + altitudes LN02. Les poses smapshot utilisent la même
  convention d'altitude (vérifié sur les points de calage), donc aucune correction de géoïde.
- Trois niveaux de MNT selon la distance : swissALTI3D 2 m (proche), swissALTIRegio 10 m,
  swissALTIRegio sous-échantillonné 40 m (lointain).
- Courbure terrestre + réfraction : le terrain à la distance horizontale t descend de t²/(2R)·(1-k).
- Chaque rayon avance par pas croissants avec la distance ; le premier passage sous le terrain
  est affiné par dichotomie. Sortie : distance 3D (m) par pixel, -1 pour le ciel.
"""
import json
import math
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rasterio.merge
import rasterio.windows
import torch
import torch.nn.functional as F

R_EARTH, K_REFRACTION = 6371000.0, 0.13
REGIO = "data/dem/swissaltiregio_2056_5728.tif"
REGIO_URL = "https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio/swissaltiregio_2056_5728.tif"
ALTI3D_DIR = Path("data/dem/swissalti3d")
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
LEVELS = [  # (source, maille m, demi-côté de la fenêtre m, distance horizontale max servie par ce niveau)
    ("alti3d", 2.0, 6500.0, 6000.0),
    ("regio", 10.0, 31000.0, 30000.0),
    ("regio", 40.0, 121000.0, 120000.0),
]
TO_LV95 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)


# ---------- Téléchargement / lecture des MNT ----------

_INDEX = None


def alti3d_tiles(e0, n0, half):
    """Tuiles swissALTI3D 2 m couvrant le carré (index local, téléchargement si manquantes)."""
    global _INDEX
    ALTI3D_DIR.mkdir(parents=True, exist_ok=True)
    if _INDEX is None:
        _INDEX = json.loads(Path("data/dem/swissalti3d_index.json").read_text())
    out = []
    for ek in range(int((e0 - half) // 1000), int((e0 + half) // 1000) + 1):
        for nk in range(int((n0 - half) // 1000), int((n0 + half) // 1000) + 1):
            entry = _INDEX.get(f"{ek}-{nk}")
            if entry is None:
                continue                                    # hors Suisse : niveaux plus grossiers
            local = ALTI3D_DIR / entry["name"]
            if not local.exists():
                tmp = local.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
                urllib.request.urlretrieve(entry["href"], tmp)
                os.replace(tmp, local)
            out.append(local)
    return out


def read_level(source, cell, e0, n0, half):
    """Fenêtre carrée de MNT centrée sur (e0, n0) à la maille demandée. NaN hors données."""
    size = int(round(2 * half / cell))
    bounds = (e0 - half, n0 - half, e0 + half, n0 + half)
    if source == "alti3d":
        files = alti3d_tiles(e0, n0, half)
        if not files:
            return None
        srcs = [rasterio.open(f) for f in files]
        data, transform = rasterio.merge.merge(srcs, bounds=bounds, res=cell, nodata=np.nan)
        for s in srcs:
            s.close()
        z = data[0].astype(np.float32)
    else:
        path = REGIO if Path(REGIO).exists() and Path(REGIO).stat().st_size > 1e10 else f"/vsicurl/{REGIO_URL}"
        with rasterio.open(path) as d:
            win = rasterio.windows.from_bounds(*bounds, transform=d.transform)
            z = d.read(1, window=win, out_shape=(size, size), boundless=True, fill_value=np.nan,
                       resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            z[z > 1e30] = np.nan
    return z[:size, :size]


# ---------- Rendu ----------

def load_arrays(e0, n0):
    """Lit les niveaux de MNT (CPU) autour de (e0, n0) : liste de (altitudes, demi-côté, distance max)."""
    out = []
    for source, cell, half, max_t in LEVELS:
        z = read_level(source, cell, e0, n0, half)
        if z is not None:
            out.append((z, half, max_t))
    return out


class Scene:
    """Pyramide de MNT centrée sur la caméra, sur GPU."""

    def __init__(self, e0, n0, device="cuda", arrays=None):
        """arrays : résultat de load_arrays (préparé sur CPU, éventuellement dans un autre thread)."""
        self.e0, self.n0, self.levels = e0, n0, []
        for z, half, max_t in (arrays if arrays is not None else load_arrays(e0, n0)):
            self.levels.append({"z": torch.as_tensor(z, device=device)[None, None], "half": half,
                                "max_t": max_t, "valid": float(np.isfinite(z).mean())})

    def height(self, level, de, dn):
        """Altitude du MNT aux décalages (de, dn) depuis le centre de la scène, en partant du niveau
        `level` et en passant au niveau plus grossier là où il n'y a pas de données (hors Suisse...)."""
        z = torch.full_like(de, float("nan"))
        for L in self.levels[level:]:
            missing = torch.isnan(z)
            if not missing.any():
                break
            grid = torch.stack([de / L["half"], -dn / L["half"]], dim=-1)[None, None]  # y image vers le sud
            zl = F.grid_sample(L["z"], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0, 0]
            zl = torch.where((de.abs() > L["half"]) | (dn.abs() > L["half"]), torch.full_like(zl, float("nan")), zl)
            z = torch.where(missing, zl, z)
        return z


def distance_steps(t_max=120000.0, t0=2.0, rel=0.003, abs_step=0.5):
    t = [t0]
    while t[-1] < t_max:
        t.append(t[-1] * (1 + rel) + abs_step)
    return torch.tensor(t)


@torch.no_grad()
def render_distance(scene, cam_e, cam_n, cam_h, dirs_lv95, device="cuda", chunk=400_000):
    """dirs_lv95 : (N, 3) directions unitaires (Est, Nord, Haut) dans le repère LV95 local.
    Renvoie la distance 3D au premier point de terrain touché (m), -1 si ciel."""
    dirs = torch.as_tensor(dirs_lv95, dtype=torch.float32, device=device)
    horiz = dirs[:, :2].norm(dim=1).clamp(min=1e-6)
    ue, un = dirs[:, 0] / horiz, dirs[:, 1] / horiz          # direction horizontale unitaire
    slope = dirs[:, 2] / horiz                               # montée du rayon par mètre horizontal
    ts = distance_steps().to(device)
    drop_coef = (1 - K_REFRACTION) / (2 * R_EARTH)
    out = torch.full((len(dirs),), -1.0, device=device)
    max_relief = 4900.0                                       # au-dessus du plus haut sommet des Alpes

    def above(t, idx, level):
        de, dn = (t * ue[idx]), (t * un[idx])
        terrain = scene.height(level, de + (cam_e - scene.e0), dn + (cam_n - scene.n0)) - drop_coef * t * t
        return cam_h + t * slope[idx] - terrain                 # > 0 : rayon au-dessus du terrain (NaN : inconnu)

    for start in range(0, len(dirs), chunk):
        idx = torch.arange(start, min(start + chunk, len(dirs)), device=device)
        t_prev = torch.zeros(len(idx), device=device)
        for k, t in enumerate(ts):
            level = next((i for i, L in enumerate(scene.levels) if t <= L["max_t"]), None)
            if level is None or len(idx) == 0:
                break
            gap = above(t.expand(len(idx)), idx, level)
            hit = gap <= 0
            # rayon qui monte plus vite que le relief le plus haut possible : ciel, on l'abandonne
            ray_h = cam_h + slope[idx] * t - drop_coef * t * t
            gone = (ray_h > max_relief) & (slope[idx] - 2 * drop_coef * t > 0)   # au-dessus de tout et montant
            if hit.any():
                lo, hi = t_prev[hit], t.expand(int(hit.sum())).clone()
                hidx = idx[hit]
                for _ in range(12):                         # dichotomie entre le dernier pas libre et le choc
                    mid = (lo + hi) / 2
                    g = above(mid, hidx, level)
                    under = g <= 0
                    hi = torch.where(under, mid, hi)
                    lo = torch.where(under, lo, mid)
                d3 = hi * torch.sqrt(1 + slope[hidx] ** 2)
                out[hidx] = d3
            keep = ~(hit | gone)
            idx, t_prev = idx[keep], t.expand(int(keep.sum())).clone()
    return out


def lv95_rotation(lat, lon):
    """Azimut (rad, horaire depuis le nord de la grille LV95) de la direction du nord vrai au point donné,
    mesuré directement en transformant un point situé 1 km plus au nord."""
    e0, n0 = TO_LV95.transform(lon, lat)
    e1, n1 = TO_LV95.transform(lon, lat + 1000 / 111_320)
    return math.atan2(e1 - e0, n1 - n0)


def render_camera(cam, width, height=None, device="cuda", scene=None):
    """Carte de distance (height, width) pour une caméra smapshot.Camera, à la résolution demandée."""
    height = height or int(round(cam.height * width / cam.width))
    s = cam.width / width
    us = (np.arange(width) + 0.5) * s
    vs = (np.arange(height) + 0.5) * s
    uu, vv = np.meshgrid(us, vs)
    d_enu = cam.ray_directions_enu(uu.ravel(), vv.ravel())
    # ENU (nord vrai) -> axes de la grille LV95 : rotation de la convergence des méridiens
    a = lv95_rotation(cam.lat, cam.lon)    # azimut grille = azimut vrai + a
    c, s_ = math.cos(a), math.sin(a)
    d = np.stack([c * d_enu[:, 0] + s_ * d_enu[:, 1], c * d_enu[:, 1] - s_ * d_enu[:, 0], d_enu[:, 2]], axis=1)
    e0, n0 = TO_LV95.transform(cam.lon, cam.lat)
    scene = scene or Scene(e0, n0, device)
    dist = render_distance(scene, e0, n0, cam.alt, d, device)
    return dist.view(height, width).cpu().numpy(), scene
