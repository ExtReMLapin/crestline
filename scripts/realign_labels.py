"""Recale les étiquettes MNT de GeoPose3K sur les photos, et mesure la qualité d'alignement.

Idée : une petite erreur d'orientation / focale dans la pose GeoPose3K équivaut, pour du relief
lointain, à une similitude 2D du rendu (cap -> dx, tangage -> dy, roulis -> rotation,
focale -> échelle). On cherche par photo la similitude qui fait coïncider la skyline du MNT
avec la skyline vue par Mask2Former sur la photo (colonnes non occultées), puis on
l'applique à la carte de distance : toutes les étiquettes (skyline ET crêtes internes) suivent.

Usage : python scripts/realign_labels.py <extrait> <prep> <photo_seg> <sortie> [n_procs]
Écrit <sortie>/<id>.npz (mêmes champs que prepare_geopose.py, étiquettes recalées)
et <sortie>/meta_realign.csv (paramètres, erreurs de skyline avant / après).
"""
import csv
import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage, optimize  # noqa: E402

cv2.setNumThreads(1)  # un seul thread par processus (le parallélisme vient du Pool)

sys.path.insert(0, str(Path(__file__).parent))
from geopose_labels import read_pfm_gz, ridge_labels  # noqa: E402
from ridge_data import top_sky_rows  # noqa: E402

SKY_ID = 2          # « sky » dans ADE20K (Mask2Former)
CLIP_PX = 0.03      # erreur par colonne plafonnée à 3 % de la hauteur (robustesse aux colonnes aberrantes)
MIN_COLS = 0.15     # il faut au moins 15 % de colonnes exploitables pour recaler


def similarity(params, w, h):
    """dx, dy (fraction de la largeur), angle (deg), log-échelle -> matrice 2x3 autour du centre."""
    dx, dy, angle, log_s = params
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, float(np.exp(log_s)))
    M[:, 2] += (dx * w, dy * w)
    return M


def warp_distance(distance, M):
    h, w = distance.shape
    d = np.nan_to_num(distance, nan=-1.0)
    # le hors-cadre devient « inconnu » (-2), traité comme à ignorer
    return cv2.warpAffine(d, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-2.0)


def photo_skyline(seg, occluder):
    """Skyline photo par colonne + colonnes exploitables (ciel en haut, bord non occulté)."""
    sky = seg == SKY_ID
    rows = top_sky_rows(sky)
    h = seg.shape[0]
    cols = np.arange(seg.shape[1])
    r = np.clip(rows, 0, h - 1)
    usable = (rows > 0) & (rows < h - 1) & ~occluder[r, cols] & ~occluder[np.maximum(r - 1, 0), cols]
    return rows, usable


def column_errors(distance_w, photo_rows, usable):
    """Écart par colonne entre la skyline MNT et la skyline photo.

    Toutes les colonnes où la photo voit du ciel comptent, y compris celles où le MNT met du
    terrain dès le haut (erreur énorme, c'est voulu : pose fausse). Le hors-cadre (-2) compte
    comme « pas de terrain » ; seules les colonnes entièrement hors cadre sont exclues.
    """
    dem_rows = top_sky_rows((distance_w == -1.0) | (distance_w == -2.0))
    h = distance_w.shape[0]
    cols = np.arange(distance_w.shape[1])
    # colonnes où le MNT ne sait rien à l'endroit de la skyline photo (hors cadre après décalage)
    unknown = distance_w[np.clip(photo_rows, 0, h - 1), cols] == -2.0
    ok = usable & ~unknown & (dem_rows < h)
    return np.abs(dem_rows - photo_rows)[ok], ok


def objective(params, distance, photo_rows, usable, clip):
    h, w = distance.shape
    e, ok = column_errors(warp_distance(distance, similarity(params, w, h)), photo_rows, usable)
    if ok.mean() < MIN_COLS * 0.5:
        return clip * 10
    reg = 0.5 * (abs(params[2]) / 3 + abs(params[3]) / 0.05)  # préférence pour les petites corrections
    return np.minimum(e, clip).mean() + reg


def finalize(prep_file, distance, seg, occ_ids, best, out_file):
    """Applique la similitude `best`, recalcule les étiquettes, mesure l'écart avant / après."""
    h, w = distance.shape
    z = dict(np.load(prep_file))
    photo_rows, usable = photo_skyline(seg, np.isin(seg, occ_ids))
    row = {"id": prep_file.stem, "usable_cols": round(float(usable.mean()), 3)}
    e0, _ = column_errors(warp_distance(distance, similarity(np.zeros(4), w, h)), photo_rows, usable)
    d_w = warp_distance(distance, similarity(best, w, h))
    e1, _ = column_errors(d_w, photo_rows, usable)
    row.update({"dx": best[0], "dy": best[1], "angle": best[2], "scale": float(np.exp(best[3])),
                "err_before_med": float(np.median(e0)) if e0.size else np.nan,
                "err_before_p90": float(np.percentile(e0, 90)) if e0.size else np.nan,
                "err_after_med": float(np.median(e1)) if e1.size else np.nan,
                "err_after_p90": float(np.percentile(e1, 90)) if e1.size else np.nan})
    outside = d_w == -2.0
    d_w[outside] = 1e9  # hors-cadre : ni ciel ni saut de profondeur, et ignoré ci-dessous
    skyline, ridge = ridge_labels(np.where(d_w == -1.0, -1.0, d_w))
    z["sky"] = d_w == -1.0
    near_border = ndimage.binary_dilation(outside, iterations=3)  # le bord du cadre n'est pas une crête
    z["skyline"] = skyline & ~near_border
    z["ridge"] = ridge & ~near_border
    z["ignore"] = z["ignore"] | near_border
    np.savez_compressed(out_file, **z)
    return row


def realign(args):
    sample_dir, prep_file, seg_file, out_file, occ_ids = args
    seg = np.load(seg_file)["labels"]
    distance = read_pfm_gz(sample_dir / "pinhole/distance_crop.pfm.gz")
    h, w = distance.shape
    photo_rows, usable = photo_skyline(seg, np.isin(seg, occ_ids))
    clip = CLIP_PX * h
    identity = np.zeros(4)
    best = identity
    if usable.mean() >= MIN_COLS:
        # 1) recherche exhaustive grossière sur les 4 paramètres, à demi-résolution
        d_half = distance[::2, ::2]
        rows_half, usable_half = photo_rows[::2] // 2, usable[::2]
        f_half = lambda p: objective(p, d_half, rows_half, usable_half, clip / 2)  # noqa: E731
        grid = itertools.product(np.linspace(-0.08, 0.08, 17), np.linspace(-0.08, 0.08, 17),
                                 np.linspace(-4, 4, 9), np.log(np.linspace(0.94, 1.06, 7)))
        start = min((np.array(g) for g in grid), key=f_half)
        # 2) affinage local à pleine résolution
        f = lambda p: objective(p, distance, photo_rows, usable, clip)  # noqa: E731
        res = optimize.minimize(f, start, method="Powell",
                                bounds=[(-0.1, 0.1), (-0.1, 0.1), (-5, 5), (-0.08, 0.08)],
                                options={"xtol": 1e-3, "ftol": 1e-3, "maxfev": 400})
        best = min([identity, start, res.x], key=f)
    return finalize(prep_file, distance, seg, occ_ids, best, out_file)


def main(extracted, prep, photo_seg, out, n_procs=40):
    from transformers import AutoConfig
    id2label = AutoConfig.from_pretrained("facebook/mask2former-swin-large-ade-semantic").id2label
    from prepare_geopose import OCCLUDERS
    occ_ids = [k for k, v in id2label.items() if v in OCCLUDERS]
    assert id2label[SKY_ID] == "sky"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(extracted) / f.stem, f, Path(photo_seg) / f.name, out / f.name, occ_ids)
            for f in sorted(Path(prep).glob("*.npz")) if (Path(photo_seg) / f.name).exists()]
    rows = []
    with Pool(int(n_procs)) as pool:
        for i, row in enumerate(pool.imap_unordered(realign, jobs, chunksize=4)):
            rows.append(row)
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    with open(out / "meta_realign.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:6])
