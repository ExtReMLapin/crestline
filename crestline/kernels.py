"""Noyaux GPU (Triton) des deux calculs les plus coûteux. Mêmes résultats que les versions PyTorch, qui restent
utilisées quand Triton n'est pas disponible (CPU, macOS, Windows sans Triton).

1. Lancer de rayons de l'horizon (horizon.horizon). En PyTorch, chaque pas de distance (~800 par rayon) lance
   une dizaine de petites opérations et écrit des tableaux intermédiaires. Ici chaque rayon fait tous ses pas
   d'un coup, en registres : ~8x plus rapide. Deux détails :
   - positions comptées depuis le coin de chaque niveau du MNT : en float32, une coordonnée de carte
     (~5e6 m en UTM) n'est juste qu'à ~0,5 m près, ce qui faisait bouger un relief tout proche de la caméra
     jusqu'à ~0,8° ; ici, quelques centimètres ;
   - atan est croissante : on garde la plus grande pente (z - chute - h) / t, l'angle n'est calculé qu'à la fin.

2. Vote de l'index de mots de crête (index.GpuIndex). Un seul noyau lit les entrées, calcule la case de cap et
   vote ; les votes sont traités par tuiles de lieux qui tiennent dans le cache L2 : ~3,5x plus rapide.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                           # pragma: no cover
    HAVE_TRITON = False


def available(device):
    """Noyaux utilisables : Triton présent, calcul sur GPU, et CRESTLINE_NO_TRITON non défini (secours)."""
    return HAVE_TRITON and str(device).startswith("cuda") and not os.environ.get("CRESTLINE_NO_TRITON")


if HAVE_TRITON:
    @triton.jit
    def _march(x0_ptr, y0_ptr, h_ptr, ux_ptr, uy_ptr, best_ptr, best_t_ptr, n_rays, t_ptr, d_ptr, s_lo, s_hi,
               dem_ptr, W, H, dx, dy, sx, sy, w_m, h_m, BLOCK: tl.constexpr):
        """Pas s_lo..s_hi d'un niveau du MNT. x0, y0 : distance de la caméra au bord ouest / nord du niveau de
        référence ; dx, dy : décalage de ce niveau ; sx, sy : pixels par mètre ; hors du niveau : NaN."""
        r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = r < n_rays
        x0 = tl.load(x0_ptr + r, mask=live, other=0.0) + dx
        y0 = tl.load(y0_ptr + r, mask=live, other=0.0) + dy
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
            xr = x0 + t * ux                                   # mètres depuis le bord ouest du niveau
            yr = y0 - t * uy                                   # mètres depuis le bord nord
            # grid_sample(bilinear, border, align_corners=False)
            ix = tl.minimum(Wf - 1.0, tl.maximum(xr * sx - 0.5, 0.0))
            iy = tl.minimum(Hf - 1.0, tl.maximum(yr * sy - 0.5, 0.0))
            fx = tl.floor(ix)
            fy = tl.floor(iy)
            xi = fx.to(tl.int32)
            yi = fy.to(tl.int32)
            in_e = (xi + 1) < W
            in_s = (yi + 1) < H
            base = yi.to(tl.int64) * W + xi
            z = tl.load(dem_ptr + base, mask=live, other=0.0) * (fx + 1.0 - ix) * (fy + 1.0 - iy)
            z = tl.where(in_e, z + tl.load(dem_ptr + base + 1, mask=live & in_e, other=0.0)
                         * (ix - fx) * (fy + 1.0 - iy), z)
            z = tl.where(in_s, z + tl.load(dem_ptr + base + W, mask=live & in_s, other=0.0)
                         * (fx + 1.0 - ix) * (iy - fy), z)
            z = tl.where(in_e & in_s, z + tl.load(dem_ptr + base + W + 1, mask=live & in_e & in_s, other=0.0)
                         * (ix - fx) * (iy - fy), z)
            z = tl.where((xr < 0.0) | (xr > w_m) | (yr < 0.0) | (yr > h_m), float("nan"), z)
            ratio = (z - d - h) / t
            better = ratio > best                              # NaN (hors données) : jamais meilleur
            best = tl.where(better, ratio, best)
            best_t = tl.where(better, t, best_t)
        tl.store(best_ptr + r, best, mask=live)
        tl.store(best_t_ptr + r, best_t, mask=live)

    @triton.jit
    def _vote(packed_ptr, ends_ptr, start_ptr, dq_ptr, w_ptr, votes_ptr, n_slots, total, offset,
              N_HEAD: tl.constexpr, DIR_BITS: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
        """Un vote par entrée : l'entrée retrouve son segment (dichotomie sur les fins cumulées), lit lieu et
        direction, calcule la case de cap (comme torch.remainder) et l'ajoute aux votes."""
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = i < total
        lo = tl.zeros([BLOCK], tl.int32)
        hi = tl.zeros([BLOCK], tl.int32) + n_slots
        for _ in range(STEPS):
            mid = (lo + hi) // 2
            go = tl.load(ends_ptr + mid, mask=live & (lo < hi), other=0) <= i
            lo = tl.where((lo < hi) & go, mid + 1, lo)
            hi = tl.where((lo < hi) & ~go, mid, hi)
        k = tl.minimum(lo, n_slots - 1)
        e = tl.load(packed_ptr + tl.load(start_ptr + k, mask=live, other=0) + i, mask=live, other=0)
        pt = (e >> DIR_BITS).to(tl.int64)
        a = (e & ((1 << DIR_BITS) - 1)).to(tl.float32) - (tl.load(dq_ptr + k, mask=live, other=0.0) - offset)
        m = a % 360.0
        m = tl.where((m != 0.0) & (m < 0.0), m + 360.0, m)
        hb = m.to(tl.int64) % N_HEAD
        tl.atomic_add(votes_ptr + pt * N_HEAD + hb, tl.load(w_ptr + k, mask=live, other=0.0), mask=live,
                      sem="relaxed")


_steps = {}


def _step_tables(device, distance_steps, level_for, drop):
    if str(device) not in _steps:
        ts = list(distance_steps())
        t = torch.tensor(ts, dtype=torch.float32)
        d = torch.tensor([drop(float(x)) for x in t.tolist()], dtype=torch.float32)
        lv = [level_for(float(x)) for x in t.tolist()]
        ranges = []
        for i in sorted(set(lv)):
            idx = [k for k, v in enumerate(lv) if v == i]
            ranges.append((i, idx[0], idx[-1] + 1))
        _steps[str(device)] = (t.to(device), d.to(device), ranges)
    return _steps[str(device)]


@torch.no_grad()
def horizon_march(dem, x, y, h, az, distance_steps, level_for, drop, block=256):
    """dem : crestline.dem.Dem ; x, y, h : (B,) (x, y de préférence en float64) ; az : (B, A) ou (1, A) en rad.
    Renvoie élévation (deg) et distance (m) de l'horizon, (B, A)."""
    dev = dem.device
    B, A = len(x), az.shape[1]
    ref = dem.levels[0]
    x0 = (x.double() - ref["xmin"]).float()[:, None].expand(B, A).contiguous().view(-1)
    y0 = (ref["ymax"] - y.double()).float()[:, None].expand(B, A).contiguous().view(-1)
    hh = h.float()[:, None].expand(B, A).contiguous().view(-1)
    ux = torch.sin(az).expand(B, A).contiguous().view(-1)
    uy = torch.cos(az).expand(B, A).contiguous().view(-1)
    best = torch.full((B * A,), float("-inf"), device=dev)
    best_t = torch.zeros_like(best)
    t, d, ranges = _step_tables(dev, distance_steps, level_for, drop)
    grid = (triton.cdiv(B * A, block),)
    for level, s_lo, s_hi in ranges:
        L = dem.levels[level]
        z = L["z"][0, 0]
        assert z.is_contiguous()
        _march[grid](x0, y0, hh, ux, uy, best, best_t, B * A, t, d, s_lo, s_hi, z, z.shape[1], z.shape[0],
                     float(ref["xmin"] - L["xmin"]), float(L["ymax"] - ref["ymax"]),
                     z.shape[1] / float(L["w"]), z.shape[0] / float(L["h"]), float(L["w"]), float(L["h"]),
                     BLOCK=block)
    return torch.rad2deg(torch.atan(best)).view(B, A), best_t.view(B, A)


def vote(packed, ends, start, dq, weight, votes, total, offset, n_head, dir_bits, block=1024):
    """Lance le noyau de vote sur `total` entrées réparties en segments (fins cumulées `ends`)."""
    _vote[(triton.cdiv(total, block),)](packed, ends, start, dq.contiguous(), weight.contiguous().float(), votes,
                                        len(ends), total, float(offset), N_HEAD=n_head, DIR_BITS=dir_bits,
                                        STEPS=24, BLOCK=block)        # 24 pas : jusqu'à 16 M de segments
