"""Comparaison de la silhouette de la photo avec les horizons de la carte.

On fait glisser la silhouette de la photo le long du panorama de chaque lieu ; l'écart le plus faible donne
le cap. Tangage (11 hypothèses + décalage vertical libre) et roulis (pente bornée) sont absorbés.

    panorama du lieu   ___╱╲____╱╲╱╲_____╱╲___________╱╲__
    photo                        ╱╲╱╲_     ← glisse sur 360°
    écart :            gros   gros  PETIT  gros  ...

Toutes les positions de glissement sont testées d'un coup par corrélation circulaire (FFT).
"""
import math

import numpy as np
import torch

from .horizon import CAM_HEIGHT, horizon

PITCHES = np.arange(-12.0, 19.0, 3.0)   # degrés ; le reste est absorbé par le décalage vertical
MIN_BINS = 20                            # en dessous, silhouette trop courte pour être fiable
MAX_ROLL = 6.0                           # degrés
MIN_SKYLINE_CONF = 0.6                   # réglé sur le val smapshot


def skyline_rows(sky):
    """Nombre de pixels de ciel consécutifs depuis le haut, par colonne."""
    return np.argmin(np.vstack([sky, np.zeros((1, sky.shape[1]), bool)]), axis=0)


def reliable_columns(prob, rows, min_conf=MIN_SKYLINE_CONF, smooth=15):
    """Colonnes où la skyline est fiable : sortie « skyline » de Crestline près de la ligne trouvée, lissée.
    Elle est basse au bord des nuages ou entre neige et ciel blanc."""
    h, w = prob.shape[1:]
    cols = np.arange(w)
    r = np.clip(rows, 0, h - 1)
    conf = np.stack([prob[1][np.clip(r + d, 0, h - 1), cols] for d in range(-3, 4)]).max(0)
    return np.convolve(conf, np.ones(smooth) / smooth, "same") >= min_conf     # min_conf = 0 : tout est gardé


def photo_profiles(rows, height, focal, n_az=1440, col_ok=None):
    """Skyline (ligne par colonne) -> P, M (n_pitch, n_az) : élévation (deg) et masque sur des cases
    d'azimut de 360/n_az degrés, case 0 = bord gauche de la photo ; et la case de l'axe optique."""
    width = len(rows)
    ok = (rows > 0) & (rows < height)                     # ciel en haut et terrain en bas
    if col_ok is not None:
        ok &= col_ok
    u = np.arange(width) + 0.5 - width / 2
    y = height / 2 - rows.astype(np.float64)
    delta = 360.0 / n_az
    k_center = math.ceil((math.degrees(math.atan(width / 2 / focal)) + 2 * delta) / delta)
    P = np.zeros((len(PITCHES), n_az), np.float32)
    M = np.zeros((len(PITCHES), n_az), np.float32)
    if ok.sum() < 2:
        return P, M, k_center
    centers = (np.arange(2 * k_center + 1) - k_center) * delta
    for i, p in enumerate(np.radians(PITCHES)):
        fwd = focal * math.cos(p) - y * math.sin(p)
        up = focal * math.sin(p) + y * math.cos(p)
        a = np.degrees(np.arctan2(u, fwd))[ok]
        e = np.degrees(np.arctan2(up, np.hypot(u, fwd)))[ok]
        order = np.argsort(a)
        a, e = a[order], e[order]
        idx = np.clip(np.searchsorted(a, centers), 1, len(a) - 1)
        valid = np.minimum(np.abs(a[idx] - centers), np.abs(a[idx - 1] - centers)) <= delta / 2
        P[i, :len(centers)] = np.where(valid, np.interp(centers, a, e), 0)
        M[i, :len(centers)] = valid
    return P, M, k_center


def profile_ffts(elev):
    """elev : tenseur (N, A) d'élévations (deg). FFT du profil centré et de son carré."""
    b = elev.float().clamp(min=-15.0)
    b = b - b.mean(1, keepdim=True)
    return torch.fft.rfft(b), torch.fft.rfft(b * b)


class Query:
    """Silhouette de la photo, prête à être corrélée avec des horizons de la carte (un masque commun à
    tous les tangages : seule la corrélation avec la photo dépend du tangage)."""

    @torch.no_grad()
    def __init__(self, P, M, device="cuda"):
        P = torch.as_tensor(P, device=device)
        M = torch.as_tensor(M, device=device)
        self.n_az = P.shape[1]
        usable = M.sum(1) >= MIN_BINS
        self.pitch_idx = torch.nonzero(usable)[:, 0].tolist()
        Mc = M[usable].amin(0) if self.pitch_idx else torch.zeros(self.n_az, device=device)
        self.n = float(Mc.sum().clamp(min=1))
        if Mc.sum() < MIN_BINS:
            self.pitch_idx = []
        Pc = (P - (P * Mc).sum(1, keepdim=True) / self.n) * Mc
        k = torch.arange(self.n_az, device=device, dtype=torch.float32)
        X = (k - (k * Mc).sum() / self.n) * Mc
        self.S_pp, self.S_xp = (Pc * Pc).sum(1), (X * Pc).sum(1)
        self.S_xx = float((X * X).sum().clamp(min=1e-6))
        self.g_max = math.radians(MAX_ROLL) * 360.0 / self.n_az        # pente max (deg/case) ≈ roulis × case
        self.FM, self.FX = torch.fft.rfft(Mc).conj(), torch.fft.rfft(X).conj()
        self.FMP = torch.fft.rfft(Pc).conj()

    @torch.no_grad()
    def score(self, B, B2, shift_center=None, shift_tol=0):
        """B, B2 : profile_ffts. Renvoie écart rms (deg), décalage (cases), indice de tangage.
        shift_center (B,) : si donné, seuls les décalages à moins de shift_tol cases de celui-ci sont testés."""
        rms = torch.full((len(B),), float("inf"), device=B.device)
        sh = torch.zeros(len(B), dtype=torch.long, device=B.device)
        pi = torch.zeros_like(sh)
        if not self.pitch_idx:
            return rms, sh, pi
        n_az, n = self.n_az, self.n
        s_b = torch.fft.irfft(self.FM * B, n=n_az)
        base = torch.fft.irfft(self.FM * B2, n=n_az) - s_b * s_b / n   # Σ m b² - (Σ m b)²/n
        s_xb = torch.fft.irfft(self.FX * B, n=n_az)
        del s_b
        banned = None
        if shift_center is not None:
            dk = (torch.arange(n_az, device=B.device)[None] - shift_center.to(B.device)[:, None] + n_az // 2) % n_az
            banned = (dk - n_az // 2).abs() > shift_tol
        for j in self.pitch_idx:
            sse = base - 2 * torch.fft.irfft(self.FMP[j] * B, n=n_az) + self.S_pp[j]
            s_xr = self.S_xp[j] - s_xb                                 # pente du résidu (roulis), bornée
            g = (s_xr / self.S_xx).clamp(-self.g_max, self.g_max)
            sse = sse - 2 * g * s_xr + g * g * self.S_xx
            if banned is not None:
                sse = sse.masked_fill(banned, float("inf"))
            v, s = sse.min(1)
            v = (v.clamp(min=0) / n).sqrt()
            better = v < rms
            rms, sh, pi = torch.where(better, v, rms), torch.where(better, s, sh), torch.where(better, j, pi)
        return rms, sh, pi


def score_all(query_args, shards, chunk=100_000):
    """Recherche exhaustive. shards : horizons de la carte (float16), dans l'ordre des points ; chaque morceau est
    soit un tenseur déjà sur un GPU, soit un couple (tableau en mémoire centrale, device) lu en flux par paquets
    (plus lent, quand la carte ne tient pas sur les GPU). Les GPU calculent en parallèle."""
    parts = [(e, e.device) if torch.is_tensor(e) else e for e in shards]
    queries = {}
    for _, dev in parts:
        queries.setdefault(str(dev), Query(*query_args, dev))
    out = [[] for _ in parts]
    for i in range(0, max(len(e) for e, _ in parts), chunk):           # paquets entrelacés entre morceaux
        for (e, dev), o in zip(parts, out):
            if i < len(e):
                block = e[i:i + chunk]
                if not torch.is_tensor(block):
                    block = torch.from_numpy(np.ascontiguousarray(block)).to(dev, non_blocking=True)
                o.append(queries[str(dev)].score(*profile_ffts(block)))
    res = [tuple(torch.cat(t) for t in zip(*o)) for o in out]
    return tuple(np.concatenate([r[k].cpu().numpy() for r in res]) for k in range(3))


@torch.no_grad()
def refine(dem, query, centers, cam_h=CAM_HEIGHT, radius=100.0, step=25.0, shifts=None, width=None, tol_deg=15.0):
    """Horizons recalculés sur une grille fine autour de chaque candidat : le relief proche change vite.

        candidat (grille 125 m)          raffinement ±100 m, pas 25 m
               ●                          · · · · ● · · · ·   → meilleur point

    shifts (K,) : décalage (cap) déjà trouvé pour chaque candidat ; width : cases couvertes par la photo.
    Si donnés, l'horizon n'est calculé que dans la fenêtre de la photo (± tol_deg) et le cap n'est cherché
    qu'à ± tol_deg : 3 à 5 fois moins de rayons qu'un tour complet. Se déplacer de 100 m ne change le cap
    que de quelques degrés ; hors de la fenêtre, une valeur fixe (le score ignore un décalage vertical).

    Renvoie, par candidat : meilleure position (K, 2), écart, décalage, tangage."""
    off = np.arange(-radius, radius + 1e-6, step)
    ox, oy = np.meshgrid(off, off)
    pos = (centers[:, None, :] + np.stack([ox.ravel(), oy.ravel()], 1)[None]).reshape(-1, 2)
    x = torch.tensor(pos[:, 0], dtype=torch.float32, device=dem.device)
    y = torch.tensor(pos[:, 1], dtype=torch.float32, device=dem.device)
    ground = dem.sample(0, x, y)
    n_az = query.n_az
    windowed = shifts is not None and width is not None
    if windowed:
        tol = int(math.ceil(tol_deg * n_az / 360))
        sc = torch.as_tensor(np.repeat(np.asarray(shifts), len(off) ** 2), device=dem.device).long()
        bins = (sc[:, None] - tol + torch.arange(width + 2 * tol, device=dem.device)[None]) % n_az
    out = []
    for i in range(0, len(pos), 4096):
        sl = slice(i, i + 4096)
        if windowed:
            part, _ = horizon(dem, x[sl], y[sl], ground[sl] + cam_h, n_az, az_bins=bins[sl])
            elev = torch.full((len(part), n_az), -15.0, device=dem.device)
            elev.scatter_(1, bins[sl], torch.nan_to_num(part, nan=-15.0))
            out.append(query.score(*profile_ffts(elev.to(query.FM.device)), sc[sl].to(query.FM.device), tol))
        else:
            elev, _ = horizon(dem, x[sl], y[sl], ground[sl] + cam_h, n_az)
            out.append(query.score(*profile_ffts(torch.nan_to_num(elev, nan=-15.0).to(query.FM.device))))
    rms, sh, pi = (torch.cat(o).cpu().numpy().reshape(len(centers), -1) for o in zip(*out))
    rms = np.where(np.isfinite(ground.cpu().numpy().reshape(len(centers), -1)), rms, np.inf)
    j = rms.argmin(1)
    r = np.arange(len(centers))
    return pos.reshape(len(centers), -1, 2)[r, j], rms[r, j], sh[r, j], pi[r, j]


def top_candidates(points, rms, k=20, radius=1000.0, pool=50_000):
    """Meilleurs points, avec suppression des non-maxima dans un rayon donné (m)."""
    order = np.argpartition(rms, pool)[:pool] if len(rms) > pool else np.arange(len(rms))
    order = order[np.argsort(rms[order])]
    order = order[np.isfinite(rms[order])]
    xy = points[order, :2]
    alive = np.ones(len(order), bool)
    chosen = []
    for i in range(len(order)):
        if not alive[i]:
            continue
        chosen.append(order[i])
        if len(chosen) == k:
            break
        alive &= np.hypot(*(xy - xy[i]).T) > radius
    return np.array(chosen, int)


def fit_pose(dem, P, M, k_center, cand, cam_h=CAM_HEIGHT):
    """Tangage et roulis réels d'un candidat (pour le rendu) : on recalcule son horizon et on ajuste
    décalage vertical et pente sur la silhouette de la photo, au cap trouvé."""
    x = torch.tensor([cand["x"]], dtype=torch.float32, device=dem.device)
    y = torch.tensor([cand["y"]], dtype=torch.float32, device=dem.device)
    elev, _ = horizon(dem, x, y, dem.sample(0, x, y) + cam_h, P.shape[1])
    b = torch.nan_to_num(elev, nan=-15.0)[0].cpu().numpy()
    j = int(np.argmin(np.abs(PITCHES - cand["pitch_hyp"])))
    k = np.nonzero(M[j])[0]
    shift = int(round(cand["heading_grid"] / (360 / P.shape[1]))) - k_center
    bb = b[(shift + k) % len(b)]
    r = P[j, k] - bb
    xk = k - k.mean()
    g_max = math.radians(MAX_ROLL) * 360.0 / P.shape[1]
    g = float(np.clip((xk * (r - r.mean())).sum() / max((xk * xk).sum(), 1e-6), -g_max, g_max))
    c = float((r - g * xk).mean())
    # photo calculée avec le tangage supposé p : élévations trop hautes de c -> tangage réel p - c ;
    # pente g (deg/case) ≈ roulis (rad) × largeur de case (deg)
    return PITCHES[j] - c, math.degrees(g / (360 / P.shape[1]))
