"""Planche de diagnostic d'une géolocalisation : photo + skyline, et profil photo contre profils MNT
(meilleur candidat et vraie position), recalés comme dans le matching (décalage + pente bornée).

Usage : python scripts/viz_geoloc.py <image> <focale_px> <E_vrai|-> <N_vrai|-> <sortie.png> <ckpt> <base1> [...]
"""
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_ridges import predict  # noqa: E402
from geolocate import MAX_ROLL, PITCHES, Matcher, Refiner, localize, photo_profiles, skyline_rows  # noqa: E402
from horizon_db import CAM_HEIGHT, horizon  # noqa: E402
from train_ridges import load_trained  # noqa: E402


def fit(P, M, b, shift):
    """Profil MNT b aligné sur le profil photo : retourne (azimuts relatifs des cases, photo, MNT recalé)."""
    k = np.nonzero(M)[0]
    bb = b[(shift + k) % len(b)]
    r = P[k] - bb
    x = k - k.mean()
    g = np.clip((x * (r - r.mean())).sum() / max((x * x).sum(), 1e-6), *(np.array([-1, 1]) * math.radians(MAX_ROLL) * 360 / len(b)))
    c = (r - g * x).mean()
    return k, P[k], bb + c + g * x


def profile_at(dem, e, n):
    t = lambda v: torch.tensor([v], dtype=torch.float32, device=dem.device)  # noqa: E731
    g = dem.ground(t(e)[None], t(n)[None])[0]
    elev, _ = horizon(dem, t(e), t(n), g + CAM_HEIGHT)
    return elev[0].cpu().numpy()


def main(image_path, focal, e_true, n_true, out, ckpt, *dbs):
    focal = float(focal)
    truth = None if e_true == "-" else (float(e_true), float(n_true))
    image = np.asarray(Image.open(image_path).convert("RGB"))
    sky = predict(load_trained(ckpt, "cuda"), image, "cuda")[0] > 0.5
    matcher = Matcher(dbs)
    lo, hi = matcher.points[:, :2].min(0), matcher.points[:, :2].max(0)
    refiner = Refiner((*lo, *hi))
    cands, diag = localize(matcher, refiner, sky, focal, truth=truth)
    P, M, k_center = photo_profiles(skyline_rows(sky), sky.shape[0], focal, matcher.n_az)
    delta = 360.0 / matcher.n_az
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]})
    ax[0].imshow(image)
    rows = skyline_rows(sky)
    ax[0].plot(np.arange(len(rows)), rows, "lime", lw=1)
    ax[0].set_axis_off()
    c0 = cands[0]
    lines = [("meilleur candidat", c0["e"], c0["n"], c0, "tab:red")]
    if truth is not None:
        lines.append(("vraie position", truth[0], truth[1], None, "tab:blue"))
    for label, e, n, c, color in lines:
        b = profile_at(refiner.dem, e, n)
        # tangage et cap : ceux du candidat, sinon meilleurs pour ce profil
        best = None
        for j in range(len(PITCHES)):
            if M[j].sum() < 20:
                continue
            for s in range(matcher.n_az) if c is None else [int(round(c["heading_grid"] / delta)) - k_center]:
                k, p, f = fit(P[j], M[j], b, s % matcher.n_az)
                err = np.sqrt(np.mean((p - f) ** 2))
                if best is None or err < best[0]:
                    best = (err, k, p, f)
        err, k, p, f = best
        # le profil photo dépend du tangage retenu pour chaque position : un tracé photo par position
        ax[1].plot((k - k_center) * delta, f, color=color, label=f"MNT, {label} ({err:.3f}°)")
        ax[1].plot((k - k_center) * delta, p, color=color, lw=1, ls=":", label=f"photo, tangage de « {label} »")
    d = f" — erreur {math.hypot(c0['e'] - truth[0], c0['n'] - truth[1]) / 1000:.2f} km" if truth else ""
    ax[1].set_title(f"meilleur candidat E {c0['e']:.0f} N {c0['n']:.0f}, cap {c0['heading_grid']:.1f}°{d}")
    ax[1].set_xlabel("azimut relatif (°)")
    ax[1].set_ylabel("élévation (°)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(out, cands[:3], diag)


if __name__ == "__main__":
    main(*sys.argv[1:])
