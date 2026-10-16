"""Lancer de rayons de l'horizon en un seul noyau GPU (Triton), même calcul que horizon_db.horizon.

La version PyTorch lance ~8 petites opérations par pas de distance (800 pas) et écrit à chaque fois des tableaux
intermédiaires (coordonnées, altitudes, angles) : sur un raffinement, ~37 000 lancements pour ~120 ms de calcul.
Ici chaque rayon fait tous ses pas d'un coup, en registres ; seule la lecture du MNT touche la mémoire.

- Interpolation de F.grid_sample(mode="bilinear", padding_mode="border", align_corners=False) : mêmes formules,
  un NaN voisin donne NaN.
- Positions comptées depuis le coin du MNT (~500 km au plus au lieu de ~2,6e6 m en LV95) : en float32, le point
  lu sur le MNT est juste à ~3 cm près au lieu de ~25 cm. La version PyTorch en souffre près de la caméra : un
  occultant à 15 m sur une pente raide bouge alors de jusqu'à ~0,7°.
- atan est croissante : on garde la plus grande pente (z - chute - h) / t et on ne calcule l'angle qu'à la fin.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _march(x0_ptr, y0_ptr, h_ptr, ux_ptr, uy_ptr, best_ptr, best_t_ptr, n_rays,
           t_ptr, d_ptr, s_lo, s_hi, dem_ptr, W, H, sx, sy, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    live = r < n_rays
    x0 = tl.load(x0_ptr + r, mask=live, other=0.0)
    y0 = tl.load(y0_ptr + r, mask=live, other=0.0)
    h = tl.load(h_ptr + r, mask=live, other=0.0)
    ux = tl.load(ux_ptr + r, mask=live, other=0.0)
    uy = tl.load(uy_ptr + r, mask=live, other=0.0)
    best = tl.load(best_ptr + r, mask=live, other=0.0)
    best_t = tl.load(best_t_ptr + r, mask=live, other=0.0)
    Wf = W.to(tl.float32)
    Hf = H.to(tl.float32)
    for s in range(s_lo, s_hi):
        t = tl.load(t_ptr + s)
        d = tl.load(d_ptr + s)
        ix = (x0 + t * ux) * sx - 0.5            # x0 : distance au bord ouest (m), sx : pixels par mètre
        iy = (y0 - t * uy) * sy - 0.5            # y0 : distance au bord nord
        ix = tl.minimum(Wf - 1.0, tl.maximum(ix, 0.0))
        iy = tl.minimum(Hf - 1.0, tl.maximum(iy, 0.0))
        fx = tl.floor(ix)
        fy = tl.floor(iy)
        x_nw = fx.to(tl.int32)
        y_nw = fy.to(tl.int32)
        nw = (fx + 1.0 - ix) * (fy + 1.0 - iy)
        ne = (ix - fx) * (fy + 1.0 - iy)
        sw = (fx + 1.0 - ix) * (iy - fy)
        se = (ix - fx) * (iy - fy)
        in_e = (x_nw + 1) < W
        in_s = (y_nw + 1) < H
        base = y_nw.to(tl.int64) * W + x_nw
        v_nw = tl.load(dem_ptr + base, mask=live, other=0.0)
        v_ne = tl.load(dem_ptr + base + 1, mask=live & in_e, other=0.0)
        v_sw = tl.load(dem_ptr + base + W, mask=live & in_s, other=0.0)
        v_se = tl.load(dem_ptr + base + W + 1, mask=live & in_e & in_s, other=0.0)
        z = v_nw * nw
        z = tl.where(in_e, z + v_ne * ne, z)
        z = tl.where(in_s, z + v_sw * sw, z)
        z = tl.where(in_e & in_s, z + v_se * se, z)
        ratio = (z - d - h) / t
        better = ratio > best                      # NaN (hors données) : jamais meilleur
        best = tl.where(better, ratio, best)
        best_t = tl.where(better, t, best_t)
    tl.store(best_ptr + r, best, mask=live)
    tl.store(best_t_ptr + r, best_t, mask=live)


_steps_cache = {}


def _steps(device, distance_steps, level_max_t, drop):
    key = str(device)
    if key not in _steps_cache:
        t = torch.as_tensor(distance_steps(), dtype=torch.float32)
        d = torch.tensor([drop * float(x) * float(x) for x in t.tolist()], dtype=torch.float32)
        lv = [next(i for i, m in enumerate(level_max_t) if float(x) <= m) for x in t.tolist()]
        ranges = []
        for i in range(len(level_max_t)):
            idx = [k for k, v in enumerate(lv) if v == i]
            if idx:
                ranges.append((i, idx[0], idx[-1] + 1))
        _steps_cache[key] = (t.to(device), d.to(device), ranges)
    return _steps_cache[key]


@torch.no_grad()
def horizon_fused(dem, e, n, h, az, distance_steps, level_max_t, drop, block=256):
    """e, n, h : (B,) (e, n de préférence en float64) ; az : (B, A) ou (1, A) azimuts (rad).
    Renvoie élévation (deg) et distance (m), (B, A)."""
    dev = dem.device
    B, A = len(e), az.shape[1]
    ux, uy = torch.sin(az).expand(B, A), torch.cos(az).expand(B, A)
    # décalage depuis le coin calculé en float64 (e, n peuvent arriver en float64 : position exacte de la caméra)
    x0 = (e.double() - dem.e_min).float()[:, None].expand(B, A).contiguous().view(-1)
    y0 = (dem.n_max - n.double()).float()[:, None].expand(B, A).contiguous().view(-1)
    hh = h.float()[:, None].expand(B, A).contiguous().view(-1)
    ux, uy = ux.contiguous().view(-1), uy.contiguous().view(-1)
    best = torch.full((B * A,), float("-inf"), device=dev)
    best_t = torch.zeros_like(best)
    t, d, ranges = _steps(dev, distance_steps, level_max_t, drop)
    grid = (triton.cdiv(B * A, block),)
    for level, s_lo, s_hi in ranges:
        lv = dem.levels[level][0, 0]
        assert lv.is_contiguous()
        _march[grid](x0, y0, hh, ux, uy, best, best_t, B * A, t, d, s_lo, s_hi, lv, lv.shape[1], lv.shape[0],
                     lv.shape[1] / float(dem.width_m), lv.shape[0] / float(dem.height_m), BLOCK=block)
    elev = torch.rad2deg(torch.atan(best)).view(B, A)
    return elev, best_t.view(B, A)
