"""Modèle numérique de terrain (MNT) sur GPU, dans n'importe quel système de coordonnées projeté (mètres).

Pyramide à 4 niveaux, choisis selon la distance au point de vue :

    distance   0 ─── 3 km ─── 10 km ─── 30 km ──────────── 120 km
    maille       c      2c        4c              8c          (c = maille du MNT fin)
    source     ────── MNT fin (zone + 30 km) ─────┤── MNT grossier (zone + 120 km) ──┤

Le MNT grossier est lu sous-échantillonné (les COG fournissent des aperçus) : on ne charge jamais
la zone + 120 km à pleine résolution.
"""
import math
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rasterio.crs
import rasterio.windows
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

R_EARTH, K_REFRACTION = 6371000.0, 0.13
LEVEL_MAX_T = (3000.0, 10000.0, 30000.0, 120000.0)   # distance max servie par chaque niveau (m)
FINE_MARGIN = LEVEL_MAX_T[2]
MAX_DIST = LEVEL_MAX_T[3]


def read_window(path, bounds, cell):
    """Lit la fenêtre `bounds` (xmin, ymin, xmax, ymax, dans le CRS du raster) à la maille `cell` (m).
    Renvoie (altitudes float32 avec NaN hors données, xmin, ymax)."""
    with rasterio.open(path) as d:
        win = rasterio.windows.from_bounds(*bounds, transform=d.transform)
        shape = (max(1, int(round((bounds[3] - bounds[1]) / cell))), max(1, int(round((bounds[2] - bounds[0]) / cell))))
        z = d.read(1, window=win, out_shape=shape, boundless=True, fill_value=np.nan,
                   resampling=Resampling.average if cell > abs(d.res[0]) * 1.5 else Resampling.bilinear)
        z = z.astype(np.float32)
        if d.nodata is not None and np.isfinite(d.nodata):
            z[z == np.float32(d.nodata)] = np.nan
    z[(z > 1e30) | (z < -1000)] = np.nan
    return z, bounds[0], bounds[3]


def raster_info(path):
    with rasterio.open(path) as d:
        return d.crs, abs(d.res[0]), d.bounds


def ensure_projected(path, out_path, target_crs=None):
    """Renvoie un MNT dans un CRS projeté (mètres) : le MNT tel quel s'il convient, sinon une copie reprojetée
    dans `target_crs`, ou, à défaut, dans la zone UTM du centre (cas d'un MNT en degrés, ex. Copernicus)."""
    with rasterio.open(path) as src:
        target = pyproj.CRS.from_user_input(target_crs) if target_crs else None
        if target is None and src.crs.is_projected:
            return Path(path)
        if target is not None and pyproj.CRS.from_user_input(src.crs.to_wkt()) == target:
            return Path(path)
        to_ll = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform((src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2)
        if target is None:
            target = pyproj.CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)
        cell = abs(src.res[0]) * (111_320 * math.cos(math.radians(lat)) if src.crs.is_geographic else 1.0)
        dst_crs = rasterio.crs.CRS.from_wkt(target.to_wkt())
        transform, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height,
                                                      *src.bounds, resolution=cell)
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=w, height=h, dtype="float32", nodata=np.nan,
                       compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1), resampling=Resampling.bilinear,
                      dst_nodata=np.nan, num_threads=8)
    print(f"MNT reprojeté ({target.name}) : {out_path}")
    return Path(out_path)


class Dem:
    """Pyramide de MNT sur GPU couvrant `bounds` (zone des points de vue) + marges."""

    def __init__(self, fine_path, bounds, coarse_path=None, device="cuda"):
        self.device = device
        crs, cell, _ = raster_info(fine_path)
        self.crs, self.cell = crs, cell
        x0, y0, x1, y1 = bounds
        m = FINE_MARGIN
        z, xmin, ymax = read_window(fine_path, (x0 - m, y0 - m, x1 + m, y1 + m), cell)
        levels = [(torch.from_numpy(z)[None, None], xmin, ymax, cell)]
        for _ in range(2):                               # niveaux 2c et 4c par moyenne (sur CPU : pics mémoire)
            prev, _, _, c = levels[-1]
            s = F.avg_pool2d(torch.nan_to_num(prev), 2, ceil_mode=True)
            k = F.avg_pool2d(torch.isfinite(prev).float(), 2, ceil_mode=True)
            levels.append((torch.where(k > 0.5, s / k.clamp(min=1e-6), torch.full_like(s, float("nan"))),
                           xmin, ymax, c * 2))
        m = MAX_DIST
        zc, xc, yc = read_window(coarse_path or fine_path, (x0 - m, y0 - m, x1 + m, y1 + m), cell * 8)
        levels.append((torch.from_numpy(zc)[None, None], xc, yc, cell * 8))
        self.levels = [{"z": t.to(device), "xmin": xm, "ymax": ym, "w": t.shape[-1] * c, "h": t.shape[-2] * c}
                       for t, xm, ym, c in levels]

    def sample(self, level, x, y):
        """x, y : tenseurs de même forme (coordonnées du CRS). Altitude bilinéaire, NaN hors données."""
        L = self.levels[level]
        shape = x.shape
        gx = (x.reshape(1, -1) - L["xmin"]) / L["w"] * 2 - 1
        gy = (L["ymax"] - y.reshape(1, -1)) / L["h"] * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        z = F.grid_sample(L["z"], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
        out = (gx.abs() > 1) | (gy.abs() > 1)
        return torch.where(out, torch.full_like(z, float("nan")), z).reshape(shape)

    def ground(self, x, y):
        x = torch.as_tensor(np.asarray(x, np.float64), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.asarray(y, np.float64), dtype=torch.float32, device=self.device)
        return self.sample(0, x, y)


def level_for(t):
    return next(i for i, m in enumerate(LEVEL_MAX_T) if t <= m)


def drop(t):
    """Abaissement apparent du terrain à la distance t : courbure terrestre corrigée de la réfraction."""
    return (1 - K_REFRACTION) / (2 * R_EARTH) * t * t


def true_north_offset(crs, x, y):
    """Angle (degrés, horaire) entre le nord de la grille du CRS et le nord vrai au point (x, y) :
    cap vrai = cap grille - offset."""
    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon, lat = to_ll.transform(x, y)
    x1, y1 = to_xy.transform(lon, lat + 0.01)
    return math.degrees(math.atan2(x1 - x, y1 - y))
