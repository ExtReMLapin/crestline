"""Évalue la géolocalisation sur les photos smapshot du test (zones géographiques jamais vues à l'entraînement).

Usage : python scripts/eval_geoloc_smapshot.py <liste_ids> <dossier_photos> <sortie> <ckpt> <n_raffines> <base1> [...]

La pose smapshot (recalée à la main) donne la position et le cap vrais. Seules les photos prises du sol sont
gardées (caméra à moins de 30 m au-dessus du MNT, à l'intérieur de la base) : la base suppose une caméra
à 1,7 m du sol.
Mesures : recall@1/5/20 à 1 km, erreur de position du meilleur candidat, erreur de cap quand il est bon.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import smapshot as S  # noqa: E402
from dem_render import TO_LV95, lv95_rotation  # noqa: E402
from eval_ridges import predict  # noqa: E402
from geolocate import Matcher, Refiner, localize  # noqa: E402
from horizon_db import REGIO  # noqa: E402
from train_ridges import load_trained  # noqa: E402

POSES = Path("data/smapshot/test_poses.json")
MAX_ABOVE_GROUND = 30.0


def poses(ids):
    cache = json.loads(POSES.read_text()) if POSES.exists() else {}
    for i in ids:
        if i not in cache:
            a = S.get(f"/images/{i}/attributes")
            cache[i] = {"pose": a["pose"], "width": a["width"], "height": a["height"]}
    POSES.write_text(json.dumps(cache))
    return cache


def main(id_list, photo_dir, out_dir, ckpt, n_refine, *dbs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = Path(id_list).read_text().split()
    cams = poses(ids)
    model = load_trained(ckpt, "cuda")
    matcher = Matcher(dbs)
    lo, hi = matcher.points[:, :2].min(0), matcher.points[:, :2].max(0)
    refiner = Refiner((*lo, *hi)) if int(n_refine) else None
    regio = rasterio.open(REGIO)
    results, skipped = [], {"aerial": 0, "outside": 0, "no_photo": 0}
    for i in ids:
        p = cams[i]["pose"]
        e, n = TO_LV95.transform(p["longitude"], p["latitude"])
        if not (lo[0] < e < hi[0] and lo[1] < n < hi[1]):
            skipped["outside"] += 1
            continue
        g = float(next(regio.sample([(e, n)]))[0])
        if not p["altitude"] - g < MAX_ABOVE_GROUND:
            skipped["aerial"] += 1
            continue
        path = Path(photo_dir) / f"{i}.jpg"
        if not path.exists():
            skipped["no_photo"] += 1
            continue
        image = np.asarray(Image.open(path).convert("RGB"))
        focal = p["focal"] * image.shape[1] / cams[i]["width"]
        sky = predict(model, image, "cuda")[0] > 0.5
        cands, diag = localize(matcher, refiner, sky, focal, int(n_refine), truth=(e, n))
        d = [np.hypot(c["e"] - e, c["n"] - n) for c in cands]
        hit = next((k + 1 for k, x in enumerate(d) if x < 1000), None)
        conv = np.degrees(lv95_rotation(p["latitude"], p["longitude"]))
        head_err = abs((cands[0]["heading_grid"] - conv - p["azimuth"] + 180) % 360 - 180)
        r = {"id": i, "top1_m": float(d[0]), "first_hit": hit, "heading_err": float(head_err),
             "best_rms": cands[0]["rms"], **diag, "top": cands}
        results.append(r)
        print(f"{i}: top1 {d[0] / 1000:.1f} km, 1er bon {hit}, cap {head_err:.1f}°, rms vrai "
              f"{diag.get('true_rms', float('nan')):.3f} meilleur {cands[0]['rms']:.3f}", flush=True)
    n_img = len(results)
    first = [r["first_hit"] for r in results]
    ok = [r for r in results if r["first_hit"] == 1]
    summary = {"n_images": n_img, "skipped": skipped,
               **{f"recall@{k}": sum(1 for f in first if f is not None and f <= k) / n_img for k in (1, 5, 20)},
               "median_top1_km": float(np.median([r["top1_m"] for r in results]) / 1000),
               "median_top1_m_when_correct": float(np.median([r["top1_m"] for r in ok])) if ok else None,
               "search_misses": sum(1 for r in results if r["first_hit"] != 1 and r.get("true_rms") is not None
                                    and r["true_rms"] < r["best_rms"]),
               "median_heading_err_when_correct": float(np.median([r["heading_err"] for r in ok])) if ok else None}
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "images": results}, open(out_dir / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
