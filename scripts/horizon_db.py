"""Base de profils d'horizon : pour chaque point d'une grille (pas de 250 m, caméra à 1,7 m du sol),
élévation angulaire de l'horizon sur 360° et distance de l'occultant, calculées sur GPU depuis swissALTIRegio.

Usage : python scripts/horizon_db.py <dossier_sortie> <E_min> <N_min> <E_max> <N_max> [pas_m=250]

Sortie : points.npy (E, N, sol), elev.npy (float16, deg), dist.npy (float16, km), en .npy pour le mmap.

- Repère LV95 (m), altitudes LN02. Azimut 0 = nord de la grille LV95, sens horaire (la convergence
  des méridiens est corrigée au moment du matching, pas ici).
- Horizon = max sur la distance t de atan((z(t) - chute(t) - h_cam) / t), avec courbure + réfraction.
- MNT en pyramide (10, 20, 40, 80 m, moyenne) ; le niveau est choisi selon la distance pour que le pas
  d'échantillonnage reste de l'ordre de la maille.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
import torch
import torch.nn.functional as F

# MNT source (grille LV95 10 m). Variable d'environnement MARCO_DEM pour en essayer un autre (ex. Copernicus).
REGIO = os.environ.get("MARCO_DEM", "data/dem/swissaltiregio_2056_5728.tif")
R_EARTH, K_REFRACTION = 6371000.0, 0.13
CAM_HEIGHT = 1.7
MAX_DIST = 120_000.0
LEVEL_MAX_T = [3000.0, 10000.0, 30000.0, MAX_DIST]     # niveau i (maille 10·2^i m) jusqu'à cette distance


def distance_steps(t0=15.0, rel=0.006, abs_step=6.0):
    t = [t0]
    while t[-1] * (1 + rel) + abs_step <= MAX_DIST:
        t.append(t[-1] * (1 + rel) + abs_step)
    return np.array(t, np.float32)


class Dem:
    """MNT 10 m sur une emprise, sur GPU, avec pyramide par moyenne (NaN = pas de données)."""

    def __init__(self, e_min, n_min, e_max, n_max, device="cuda"):
        self.e_min, self.n_max, self.device = e_min, n_max, device
        with rasterio.open(REGIO) as d:
            win = rasterio.windows.from_bounds(e_min, n_min, e_max, n_max, transform=d.transform)
            z = d.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        z[z > 1e30] = np.nan
        z[z < -1000] = np.nan
        self.shape = z.shape
        levels = [torch.from_numpy(z)[None, None]]               # pyramide construite sur CPU (pics mémoire)
        for _ in range(len(LEVEL_MAX_T) - 1):
            prev = levels[-1]
            s = F.avg_pool2d(torch.nan_to_num(prev), 2, ceil_mode=True)
            c = F.avg_pool2d(torch.isfinite(prev).float(), 2, ceil_mode=True)
            levels.append(torch.where(c > 0.5, s / c.clamp(min=1e-6), torch.full_like(s, float("nan"))))
        self.levels = [lv.to(device) for lv in levels]
        # coordonnées normalisées de grid_sample : x = -1 au bord ouest, +1 au bord est ; y = -1 au nord
        self.width_m, self.height_m = self.shape[1] * 10.0, self.shape[0] * 10.0

    def grid(self, e, n):
        return torch.stack([(e - self.e_min) / self.width_m * 2 - 1, (self.n_max - n) / self.height_m * 2 - 1], -1)

    def sample(self, level, e, n):
        """e, n : tenseurs (B, A). Altitude bilinéaire, NaN hors données."""
        return F.grid_sample(self.levels[level], self.grid(e, n)[None], mode="bilinear",
                             padding_mode="border", align_corners=False)[0, 0]

    def ground(self, e, n):
        return self.sample(0, torch.as_tensor(e, device=self.device), torch.as_tensor(n, device=self.device))


@torch.no_grad()
def horizon(dem, e, n, h, n_az=1440):
    """e, n, h : (B,) positions et altitudes caméra. Renvoie élévation (deg) et distance (m) de l'horizon, (B, n_az)."""
    dev = dem.device
    az = torch.arange(n_az, device=dev) * (2 * math.pi / n_az)
    ue, un = torch.sin(az), torch.cos(az)
    best = torch.full((len(e), n_az), -90.0, device=dev)
    best_t = torch.zeros_like(best)
    drop = (1 - K_REFRACTION) / (2 * R_EARTH)
    e, n, h = e[:, None], n[:, None], h[:, None]
    for t in distance_steps().tolist():
        level = next(i for i, m in enumerate(LEVEL_MAX_T) if t <= m)
        z = dem.sample(level, e + t * ue, n + t * un)
        ang = torch.rad2deg(torch.atan((z - drop * t * t - h) / t))          # t > 0 : atan2(y, t) = atan(y / t)
        better = ang > best                                  # NaN (hors données) : jamais meilleur
        best = torch.where(better, ang, best)
        best_t = torch.where(better, torch.full_like(best_t, t), best_t)
    return best, best_t


def build(out, e_min, n_min, e_max, n_max, step=250.0, n_az=1440, batch=512, device="cuda"):
    margin = MAX_DIST
    dem = Dem(e_min - margin, n_min - margin, e_max + margin, n_max + margin, device)
    es = np.arange(e_min + step / 2, e_max, step)
    ns = np.arange(n_min + step / 2, n_max, step)
    ee, nn = np.meshgrid(es, ns)
    ee, nn = ee.ravel().astype(np.float64), nn.ravel().astype(np.float64)
    ground = np.concatenate([dem.ground(torch.tensor(ee[i:i + 100000], dtype=torch.float32)[None],
                                        torch.tensor(nn[i:i + 100000], dtype=torch.float32)[None])[0].cpu().numpy()
                             for i in range(0, len(ee), 100000)])
    keep = np.isfinite(ground) & (ground > 0)
    ee, nn, ground = ee[keep], nn[keep], ground[keep]
    print(f"{len(ee)} points de vue, MNT {dem.shape}", flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "points.npy", np.stack([ee, nn, ground], 1))
    elev = np.lib.format.open_memmap(out / "elev.npy", "w+", np.float16, (len(ee), n_az))
    dist = np.lib.format.open_memmap(out / "dist.npy", "w+", np.float16, (len(ee), n_az))   # km
    t0 = time.time()
    for i in range(0, len(ee), batch):
        sl = slice(i, i + batch)
        f = lambda a: torch.tensor(a[sl], dtype=torch.float32, device=device)  # noqa: E731
        b, bt = horizon(dem, f(ee), f(nn), f(ground) + CAM_HEIGHT, n_az)
        elev[sl] = b.cpu().numpy()
        dist[sl] = (bt / 1000).cpu().numpy()
        if (i // batch) % 20 == 0:
            done = min(i + batch, len(ee))
            rate = done / (time.time() - t0)
            print(f"{done}/{len(ee)}  {rate:.0f} points/s  reste {(len(ee) - done) / rate / 60:.1f} min", flush=True)
    elev.flush()
    dist.flush()
    print("écrit", out, f"{(time.time() - t0) / 60:.1f} min")


class Rows:
    """Vue en lignes sur plusieurs tableaux mmap mis bout à bout (bases découpées en bandes)."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.starts = np.cumsum([0] + [len(a) for a in arrays])
        self.shape = (int(self.starts[-1]), arrays[0].shape[1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        out = np.empty((len(rows), self.shape[1]), self.arrays[0].dtype)
        which = np.searchsorted(self.starts, rows, side="right") - 1
        for k, a in enumerate(self.arrays):
            sel = which == k
            if sel.any():
                out[sel] = a[rows[sel] - self.starts[k]]
        return out


def load_many(db_dirs):
    """Plusieurs bases (bandes) : points concaténés et profils en vue Rows (mmap)."""
    parts = [load(d) for d in db_dirs]
    return np.concatenate([p[0] for p in parts]), Rows([p[1] for p in parts])


def load(db_dir):
    """Base en mmap : points (N, 3) E/N/sol, elev (N, A) deg, dist (N, A) km."""
    d = Path(db_dir)
    return (np.load(d / "points.npy"), np.load(d / "elev.npy", mmap_mode="r"),
            np.load(d / "dist.npy", mmap_mode="r"))


if __name__ == "__main__":
    a = sys.argv[1:]
    build(a[0], *map(float, a[1:5]), step=float(a[5]) if len(a) > 5 else 250.0)

