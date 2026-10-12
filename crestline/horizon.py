"""Horizons et rendus calculés depuis le MNT, sur GPU.

Horizon d'un point de vue = pour chaque direction, l'angle le plus haut sous lequel on voit le relief :

        angle max = horizon
             ╲                      ▲ sommet à 40 km
              ╲   ▲ 8 km      ___/ ╲___
      ●────────╲_/ ╲_________/          ╲____
    caméra (1,7 m au-dessus du sol)

Azimut 0 = nord de la grille du CRS de la carte, sens horaire.
"""
import math

import numpy as np
import torch

from .dem import MAX_DIST, drop, level_for

CAM_HEIGHT = 1.7


def distance_steps(t0=15.0, rel=0.006, abs_step=6.0):
    """Pas de marche le long d'un rayon : serrés près de la caméra, de plus en plus larges au loin."""
    t = [t0]
    while t[-1] * (1 + rel) + abs_step <= MAX_DIST:
        t.append(t[-1] * (1 + rel) + abs_step)
    return t


@torch.no_grad()
def horizon(dem, x, y, h, n_az=1440):
    """x, y, h : tenseurs (B,) positions et altitudes caméra. Renvoie élévation (deg) et distance (m), (B, n_az)."""
    dev = dem.device
    az = torch.arange(n_az, device=dev) * (2 * math.pi / n_az)
    ux, uy = torch.sin(az), torch.cos(az)
    best = torch.full((len(x), n_az), -90.0, device=dev)
    best_t = torch.zeros_like(best)
    x, y, h = x[:, None], y[:, None], h[:, None]
    for t in distance_steps():
        z = dem.sample(level_for(t), x + t * ux, y + t * uy)
        ang = torch.rad2deg(torch.atan((z - drop(t) - h) / t))
        better = ang > best                                    # NaN (hors données) : jamais meilleur
        best = torch.where(better, ang, best)
        best_t = torch.where(better, torch.full_like(best_t, t), best_t)
    return best, best_t


def camera_rays(width, height, focal, heading, pitch, roll=0.0):
    """Directions (H*W, 3) en (est, nord, haut) dans la grille du CRS, pour une caméra sténopé.
    heading : cap grille (deg), pitch : tangage (deg, + vers le haut), roll : roulis (deg, + horaire)."""
    h, p, r = map(math.radians, (heading, pitch, roll))
    fwd = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    right0 = np.array([math.cos(h), -math.sin(h), 0.0])
    up0 = np.cross(right0, fwd)
    right = math.cos(r) * right0 - math.sin(r) * up0
    up = np.cross(right, fwd)
    u = np.arange(width) + 0.5 - width / 2
    v = height / 2 - (np.arange(height) + 0.5)
    uu, vv = np.meshgrid(u, v)
    d = fwd[None] * focal + right[None] * uu.reshape(-1, 1) + up[None] * vv.reshape(-1, 1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@torch.no_grad()
def render_depth(dem, x, y, h, rays, chunk=200_000):
    """Distance horizontale (m) du premier relief touché par chaque rayon, NaN pour le ciel."""
    dev = dem.device
    rays = torch.as_tensor(rays, dtype=torch.float32, device=dev)
    horiz = rays[:, :2].norm(dim=1).clamp(min=1e-6)
    ux, uy, slope = rays[:, 0] / horiz, rays[:, 1] / horiz, rays[:, 2] / horiz
    out = torch.full((len(rays),), float("nan"), device=dev)
    for s in range(0, len(rays), chunk):
        idx = torch.arange(s, min(s + chunk, len(rays)), device=dev)
        for t in distance_steps():
            if len(idx) == 0:
                break
            z = dem.sample(level_for(t), x + t * ux[idx], y + t * uy[idx])
            hit = (h + t * slope[idx]) <= (z - drop(t))
            out[idx[hit]] = t
            gone = (h + t * slope[idx] > 4900) & (slope[idx] > 0)   # au-dessus de tout relief, et montant
            idx = idx[~(hit | gone)]
    return out
