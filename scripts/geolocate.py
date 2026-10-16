"""Géolocalisation par la skyline : profil d'horizon de la photo contre la base de profils MNT.

Baseline sans apprentissage :
- skyline de la photo (Crestline) -> profil angulaire (azimut relatif, élévation), pour plusieurs tangages ;
- pour chaque point de la base et chaque cap (pas de 0,25°) : écart quadratique moyen entre les deux profils,
  après retrait du meilleur décalage vertical (absorbe le reste de tangage et l'altitude approximative) ;
- le roulis (petit) incline le profil : on retire aussi la meilleure pente, bornée à ±MAX_ROLL ;
- tout est calculé par corrélation circulaire FFT sur GPU, pour tous les caps à la fois.
"""
import math
import os

import numpy as np
import torch

from horizon_db import load as load_db

PITCHES = np.arange(-12.0, 19.0, 3.0)        # degrés ; le reste est absorbé par le décalage vertical
MIN_BINS = 20                                # en dessous, profil photo trop court pour être fiable
MAX_ROLL = 6.0                               # degrés : le roulis est absorbé par une pente bornée du résidu


def skyline_rows(sky):
    """Nombre de pixels de ciel consécutifs depuis le haut, par colonne."""
    return np.argmin(np.vstack([sky, np.zeros((1, sky.shape[1]), bool)]), axis=0)


MIN_SKYLINE_CONF = 0.6                       # réglé sur le val smapshot (0,5 : 72,0 %, 0,6 : 73,6 %, 0,65 : 72,0 %)


def reliable_columns(prob, rows, min_conf=MIN_SKYLINE_CONF, min_sharp=0.0, smooth=15):
    """Colonnes où la skyline est fiable, d'après les sorties de Crestline (prob : ciel, skyline, crêtes) :
    - confiance : sortie « skyline » au voisinage de la ligne trouvée (basse au bord des nuages, neige/ciel...) ;
    - netteté : écart de probabilité de ciel juste au-dessus et juste au-dessous de la ligne.
    Les deux sont lissées sur `smooth` colonnes pour ne pas trouer le profil au moindre pixel douteux."""
    h, w = prob.shape[1:]
    cols = np.arange(w)
    r = np.clip(rows, 0, h - 1)
    conf = np.stack([prob[1][np.clip(r + d, 0, h - 1), cols] for d in range(-3, 4)]).max(0)
    sharp = np.abs(prob[0][np.clip(r - 4, 0, h - 1), cols] - prob[0][np.clip(r + 4, 0, h - 1), cols])
    k = np.ones(smooth) / smooth
    conf, sharp = np.convolve(conf, k, "same"), np.convolve(sharp, k, "same")
    return (conf >= min_conf) & (sharp >= min_sharp)


def photo_profiles(rows, height, focal, n_az=1440, pitches=PITCHES, col_ok=None):
    """rows : ligne de skyline par colonne. Renvoie P, M (n_pitch, n_az) : élévation (deg) et masque sur des
    cases d'azimut relatif de 360/n_az degrés, la case 0 étant le bord gauche de la photo ; et le décalage
    (en cases) entre la case 0 et l'axe optique."""
    width = len(rows)
    ok = (rows > 0) & (rows < height)                     # ciel en haut et terrain en bas
    if col_ok is not None:
        ok &= col_ok
    u = np.arange(width) + 0.5 - width / 2
    y = height / 2 - rows.astype(np.float64)              # bord du pixel, vers le haut
    delta = 360.0 / n_az
    half = math.degrees(math.atan(width / 2 / focal)) + 2 * delta
    k_center = math.ceil(half / delta)
    P = np.zeros((len(pitches), n_az), np.float32)
    M = np.zeros((len(pitches), n_az), np.float32)
    for i, p in enumerate(np.radians(pitches)):
        fwd = focal * math.cos(p) - y * math.sin(p)
        up = focal * math.sin(p) + y * math.cos(p)
        az = np.degrees(np.arctan2(u, fwd))
        el = np.degrees(np.arctan2(up, np.hypot(u, fwd)))
        if ok.sum() < 2:
            continue
        a, e = az[ok], el[ok]
        order = np.argsort(a)
        a, e = a[order], e[order]
        centers = (np.arange(2 * k_center + 1) - k_center) * delta
        vals = np.interp(centers, a, e)
        # case valide si une colonne valide tombe à moins d'une demi-case
        idx = np.clip(np.searchsorted(a, centers), 1, len(a) - 1)
        gap = np.minimum(np.abs(a[idx] - centers), np.abs(a[idx - 1] - centers))
        valid = gap <= delta / 2
        P[i, :len(centers)] = np.where(valid, vals, 0)
        M[i, :len(centers)] = valid
    return P, M, k_center


class Matcher:
    """Base de profils répartie sur tous les GPU visibles (float16). Les FFT sont recalculées par paquet à
    chaque requête (une base à 125 m ne tient pas en mémoire avec ses FFT précalculées) ; les GPU
    calculent en parallèle (aucune synchronisation avant la fin)."""

    def __init__(self, db_dirs, devices=None, chunk=100_000):
        pts, elev = [], []
        for d in db_dirs:
            p, e, _ = load_db(d)
            pts.append(p)
            elev.append(e)
        self.points = np.concatenate(pts)
        elev = np.concatenate(elev)
        devices = devices or [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        bounds = np.linspace(0, len(elev), len(devices) + 1).astype(int)
        self.shards = [(dev, torch.as_tensor(elev[bounds[i]:bounds[i + 1]], device=dev)) for i, dev in enumerate(devices)]
        self.chunk = chunk
        self.n_az = elev.shape[1]
        print(f"base : {len(self.points)} points, {self.n_az} azimuts, {len(devices)} GPU", flush=True)

    def scores(self, P, M):
        """P, M : (n_pitch, n_az). Renvoie, par point : rms (deg), case de décalage, indice de tangage."""
        queries = {dev: Query(P, M, dev) for dev, _ in self.shards}
        out = {dev: [] for dev, _ in self.shards}
        n_max = max(len(e) for _, e in self.shards)
        for i in range(0, n_max, self.chunk):                 # paquets entrelacés entre GPU
            for dev, elev in self.shards:
                if i < len(elev):
                    out[dev].append(queries[dev].score(*profile_ffts(elev[i:i + self.chunk])))
        res = [tuple(torch.cat(o) for o in zip(*out[dev])) for dev, _ in self.shards]
        return tuple(np.concatenate([r[k].cpu().numpy() for r in res]) for k in range(3))


def profile_ffts(elev):
    """elev : tenseur (N, A) d'élévations (deg). FFT du profil centré et de son carré."""
    b = elev.float().clamp(min=-15.0)
    b = b - b.mean(1, keepdim=True)
    return torch.fft.rfft(b), torch.fft.rfft(b * b)


class Query:
    """Profil photo (tous les tangages) prêt à être corrélé avec des profils de la base.

    Un seul masque de cases (commun à tous les tangages) : les corrélations qui ne dépendent que du masque
    et du profil MNT sont calculées une fois, seule celle avec le profil photo dépend du tangage."""

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
        Pc = (P - (P * Mc).sum(1, keepdim=True) / self.n) * Mc         # centré sur les cases valides
        k = torch.arange(self.n_az, device=device, dtype=torch.float32)
        X = (k - (k * Mc).sum() / self.n) * Mc                          # abscisse centrée (cases)
        self.S_pp, self.S_xp = (Pc * Pc).sum(1), (X * Pc).sum(1)
        self.S_xx = float((X * X).sum().clamp(min=1e-6))
        self.g_max = math.radians(MAX_ROLL) * 360.0 / self.n_az        # pente max (deg/case) ≈ roulis × case
        self.FM, self.FX = torch.fft.rfft(Mc).conj(), torch.fft.rfft(X).conj()
        self.FMP = torch.fft.rfft(Pc).conj()

    @torch.no_grad()
    def score(self, B, B2, shift_center=None, shift_tol=0):
        """B, B2 : profile_ffts. Renvoie rms (deg), décalage (cases), indice de tangage.
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
            k = torch.arange(n_az, device=B.device)
            dk = (k[None] - shift_center.to(B.device)[:, None] + n_az // 2) % n_az - n_az // 2
            banned = dk.abs() > shift_tol
        for j in self.pitch_idx:
            s_pb = torch.fft.irfft(self.FMP[j] * B, n=n_az)
            sse = base - 2 * s_pb + self.S_pp[j]                        # Σ m (P - b - c*)², somme(Pc) = 0
            # pente g du résidu (roulis), bornée : sse(g) = sse - 2 g S_xr + g² S_xx
            s_xr = self.S_xp[j] - s_xb
            g = (s_xr / self.S_xx).clamp(-self.g_max, self.g_max)
            sse = sse - 2 * g * s_xr + g * g * self.S_xx
            if banned is not None:
                sse = sse.masked_fill(banned, float("inf"))
            v, s = sse.min(1)
            v = (v.clamp(min=0) / n).sqrt()
            better = v < rms
            rms, sh, pi = torch.where(better, v, rms), torch.where(better, s, sh), torch.where(better, j, pi)
        return rms, sh, pi


class Refiner:
    """Raffinement local : horizons recalculés sur une grille fine autour des candidats (MNT sur un autre GPU)."""

    def __init__(self, bounds, device="cuda:0", radius=100.0, step=25.0):
        from horizon_db import Dem
        e_min, n_min, e_max, n_max = bounds
        m = 120_000.0
        self.dem, self.device = Dem(e_min - m, n_min - m, e_max + m, n_max + m, device), device
        self.radius, self.step = radius, step

    @torch.no_grad()
    def refine(self, query, centers, n_az=1440, shifts=None, width=None, tol_deg=15.0):
        """centers : (K, 2) E/N. Renvoie, par candidat, meilleure position (K, 2), rms, décalage, tangage.

        shifts (K,) : décalage (cap) déjà trouvé pour chaque candidat ; width : cases couvertes par la photo.
        Si donnés, l'horizon n'est calculé que dans la fenêtre de la photo (± tol_deg) et le cap n'est
        cherché qu'à ± tol_deg : 3 à 5 fois moins de rayons qu'un tour complet."""
        from horizon_db import CAM_HEIGHT, horizon
        off = np.arange(-self.radius, self.radius + 1e-6, self.step)
        oe, on = np.meshgrid(off, off)
        per = len(off) ** 2
        pos = (centers[:, None, :] + np.stack([oe.ravel(), on.ravel()], 1)[None]).reshape(-1, 2)
        e = torch.tensor(pos[:, 0], dtype=torch.float32, device=self.device)
        n = torch.tensor(pos[:, 1], dtype=torch.float32, device=self.device)
        ground = self.dem.ground(e[None], n[None])[0]
        windowed = shifts is not None and width is not None and os.environ.get("REFINE_FULL") != "1"
        tol_deg = float(os.environ.get("REFINE_TOL", tol_deg))
        if windowed:
            tol = int(math.ceil(tol_deg * n_az / 360))
            sc = torch.as_tensor(np.repeat(np.asarray(shifts), per), device=self.device).long()
            bins = (sc[:, None] - tol + torch.arange(width + 2 * tol, device=self.device)[None]) % n_az
        out = []
        for i in range(0, len(pos), 4096):
            sl = slice(i, i + 4096)
            if windowed:
                part, _ = horizon(self.dem, e[sl], n[sl], ground[sl] + CAM_HEIGHT, n_az, az_bins=bins[sl])
                elev = torch.full((len(part), n_az), -15.0, device=self.device)
                elev.scatter_(1, bins[sl], torch.nan_to_num(part, nan=-15.0))
                out.append(query.score(*profile_ffts(elev.to(query.FM.device)), sc[sl], tol))
            else:
                elev, _ = horizon(self.dem, e[sl], n[sl], ground[sl] + CAM_HEIGHT, n_az)
                out.append(query.score(*profile_ffts(torch.nan_to_num(elev, nan=-15.0).to(query.FM.device))))
        rms, sh, pi = (torch.cat(o).cpu().numpy().reshape(len(centers), -1) for o in zip(*out))
        rms = np.where(np.isfinite(ground.cpu().numpy().reshape(len(centers), -1)), rms, np.inf)
        j = rms.argmin(1)
        r = np.arange(len(centers))
        return pos.reshape(len(centers), -1, 2)[r, j], rms[r, j], sh[r, j], pi[r, j]


def top_candidates(points, rms, k=20, radius=1000.0, pool=50_000):
    """Meilleurs points, avec suppression des non-maxima dans un rayon donné (m), parmi les `pool` meilleurs."""
    order = np.argpartition(rms, pool)[:pool] if len(rms) > pool else np.arange(len(rms))
    order = order[np.argsort(rms[order])]
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
    return np.array(chosen)


def localize(matcher, refiner, sky, focal, n_refine=100, k=20, truth=None):
    """Masque de ciel (H, W) + focale (px) -> k candidats (non-maxima à 1 km), du meilleur au moins bon :
    dicts e, n, rms (deg), heading_grid (deg, depuis le nord de la grille LV95), pitch (deg).
    truth (E, N) optionnel : ajoute le score à la vraie position (diagnostic, hors classement)."""
    P, M, k_center = photo_profiles(skyline_rows(sky), sky.shape[0], focal, matcher.n_az)
    rms, shift, pitch = matcher.scores(P, M)
    pos = matcher.points[:, :2]
    diag = {"bins": int(M.max(0).sum())}
    if truth is not None:
        d = np.hypot(*(pos - np.asarray(truth)).T)
        diag["true_rms_grid"] = float(rms[d < 300].min()) if d.min() < 300 else None
    if refiner is not None and n_refine:
        cand = top_candidates(matcher.points, rms, n_refine, radius=500.0)
        centers = pos[cand]
        q = Query(P, M)
        cols = np.nonzero(M.max(0))[0]
        width = int(cols.max()) + 1 if len(cols) else None      # cases couvertes par la photo (None : tour complet)
        pos, rms, shift, pitch = refiner.refine(q, centers, shifts=shift[cand], width=width)
        if truth is not None:
            diag["true_rms"] = float(refiner.refine(q, np.asarray([truth], float))[1][0])
    elif truth is not None:
        diag["true_rms"] = diag["true_rms_grid"]
    delta = 360.0 / matcher.n_az
    best = top_candidates(pos, rms, k)
    return [{"e": float(pos[i, 0]), "n": float(pos[i, 1]), "rms": float(rms[i]),
             "heading_grid": float(((shift[i] + k_center) * delta) % 360), "pitch": float(PITCHES[pitch[i]])}
            for i in best], diag


def localize_indexed(index, elev, points, refiner, sky, focal, n_places=1000, n_refine=100, k=20,
                     truth=None, device="cuda", col_ok=None):
    """Comme localize, mais la recherche exhaustive est remplacée par l'index de mots de crête :
    index (ShardedIndex) -> n_places lieux -> score exact sur ces lieux -> raffinement des n_refine meilleurs.
    elev : profils de la base (mmap), points : (N, 3) ; indices globaux communs avec l'index."""
    P, M, k_center = photo_profiles(skyline_rows(sky), sky.shape[0], focal, elev.shape[1], col_ok=col_ok)
    j = int(np.argmin(np.abs(PITCHES)))
    diag = {"bins": int(M.max(0).sum())}
    if M[j].sum() < MIN_BINS:
        return [], diag
    places = index.query(P[j], M[j], k_center, n_places, device=device)
    q = Query(P, M, device)
    rows = np.sort(places)
    B, B2 = profile_ffts(torch.as_tensor(np.asarray(elev[rows]), device=device))
    rms, shift, pitch = (t.cpu().numpy() for t in q.score(B, B2))
    pos = points[rows, :2]
    if truth is not None:
        d = np.hypot(*(pos - np.asarray(truth)).T)
        diag["in_shortlist"] = bool(d.min() < 1000)
    cand = top_candidates(pos, rms, n_refine, radius=500.0)
    centers = pos[cand]
    cols = np.nonzero(M.max(0))[0]
    width = int(cols.max()) + 1 if len(cols) else None          # cases couvertes par la photo (None : tour complet)
    pos, rms, shift, pitch = refiner.refine(q, centers, shifts=shift[cand], width=width)
    if truth is not None:
        diag["true_rms"] = float(refiner.refine(q, np.asarray([truth], float))[1][0])
    delta = 360.0 / elev.shape[1]
    best = top_candidates(pos, rms, k)
    return [{"e": float(pos[i, 0]), "n": float(pos[i, 1]), "rms": float(rms[i]),
             "heading_grid": float(((shift[i] + k_center) * delta) % 360), "pitch": float(PITCHES[pitch[i]])}
            for i in best], diag
