"""Géolocalise des photos sans GPS (local uniquement) : focale EXIF -> skyline Crestline -> base d'horizons.

Usage : python scripts/geolocate_photos.py <dossier_photos> <sortie> <ckpt> <largeur_capteur_mm> <bases,...> <index,...>

Pipeline rapide : index de mots de crête -> vérification exacte -> raffinement ; colonnes de skyline peu
fiables (bord des nuages...) ignorées.

Sortie : planche par photo (photo + skyline, profils photo/MNT du meilleur candidat), carte des
5 meilleurs candidats de chaque photo, et results.json (lat/lon, cap, score).
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyproj
from PIL import Image, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from dem_render import lv95_rotation  # noqa: E402
from eval_ridges import predict  # noqa: E402
from geolocate import Refiner, localize_indexed, reliable_columns, skyline_rows  # noqa: E402
from horizon_db import load_many  # noqa: E402
from ridge_index import ShardedIndex  # noqa: E402
from train_ridges import load_trained  # noqa: E402

TO_WGS84 = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
LONG_SIDE = 1536


def main(photo_dir, out_dir, ckpt, sensor_mm, dbs, indexes):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_trained(ckpt, "cuda")
    points, elev = load_many(dbs.split(","))
    index = ShardedIndex(indexes.split(","))
    refiner = Refiner((*points[:, :2].min(0), *points[:, :2].max(0)))
    results = {}
    for path in sorted(Path(photo_dir).glob("*.[jJ][pP][gG]")):
        raw = Image.open(path)
        focal_mm = float(raw.getexif().get_ifd(0x8769)[37386])
        img = ImageOps.exif_transpose(raw).convert("RGB")
        img.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
        image = np.asarray(img)
        # focale en pixels : le grand côté du capteur correspond au grand côté de l'image
        focal = focal_mm / float(sensor_mm) * max(image.shape[:2])
        prob = predict(model, image, "cuda")
        sky = prob[0] > 0.5
        ok = reliable_columns(prob, skyline_rows(sky))
        cands, diag = localize_indexed(index, elev, points, refiner, sky, focal, col_ok=ok)
        if len(cands) < 2:
            print(f"{path.name}: skyline fiable trop courte", flush=True)
            continue
        for c in cands:
            c["lon"], c["lat"] = TO_WGS84.transform(c["e"], c["n"])
            c["heading_true"] = (c["heading_grid"] - math.degrees(lv95_rotation(c["lat"], c["lon"]))) % 360
        results[path.name] = {"focal_px": focal, "hfov_deg": 2 * math.degrees(math.atan(image.shape[1] / 2 / focal)),
                              **diag, "top": cands[:5]}
        c = cands[0]
        print(f"{path.name}: {c['lat']:.5f}, {c['lon']:.5f}  cap {c['heading_true']:.0f}°  rms {c['rms']:.3f}° "
              f"(2e : {cands[1]['rms']:.3f}° à {math.hypot(cands[1]['e'] - c['e'], cands[1]['n'] - c['n']) / 1000:.1f} km)",
              flush=True)
        fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.4, 1]})
        ax[0].imshow(image)
        rows = skyline_rows(sky).astype(float)
        ax[0].plot(np.arange(image.shape[1]), np.where(ok, rows, np.nan), "lime", lw=1.5)
        ax[0].plot(np.arange(image.shape[1]), np.where(ok, np.nan, rows), "red", lw=1)
        ax[0].set_title(path.name)
        ax[0].set_axis_off()
        es, ns = [x["e"] for x in cands[:5]], [x["n"] for x in cands[:5]]
        ax[1].scatter(es, ns, c=range(len(es)), cmap="autumn")
        for k, x in enumerate(cands[:5]):
            ax[1].annotate(f"{k + 1}: {x['rms']:.3f}°", (x["e"], x["n"]))
            ax[1].arrow(x["e"], x["n"], 2000 * math.sin(math.radians(x["heading_grid"])),
                        2000 * math.cos(math.radians(x["heading_grid"])), width=150, alpha=0.4)
        ax[1].set_aspect("equal")
        ax[1].set_title("5 meilleurs candidats (LV95), flèche = cap")
        fig.tight_layout()
        fig.savefig(out_dir / f"{path.stem}.png", dpi=80)
        plt.close(fig)
    json.dump(results, open(out_dir / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
