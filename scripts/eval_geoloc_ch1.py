"""Évalue la géolocalisation par la skyline sur CH1 (203 photos, lat/lon vraies, focale connue).

Usage : python scripts/eval_geoloc_ch1.py <dossier_CH1_extrait> <sortie> <ckpt> <n_raffines> <base1> [<base2> ...]

n_raffines > 0 : les n meilleurs candidats (non-maxima à 500 m) sont raffinés sur une grille fine
(horizons recalculés, MNT sur cuda:1).

Mesures (comme Baatz et al.) : une photo est bien localisée si un candidat est à moins de 1 km de la vérité.
- recall@1/5/20 sur les candidats après suppression des non-maxima (rayon 1 km) ;
- erreur du meilleur candidat (m) ;
- ratés de recherche : échecs où la vraie position (raffinée) aurait battu le meilleur candidat.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dem_render import TO_LV95  # noqa: E402
from eval_ch1 import load_ch1  # noqa: E402
from eval_ridges import predict  # noqa: E402
from geolocate import Matcher, Refiner, localize  # noqa: E402
from train_ridges import load_trained  # noqa: E402


def main(root, out_dir, ckpt, n_refine, *dbs):
    n_refine = int(n_refine)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_trained(ckpt, "cuda")
    matcher = Matcher(dbs)
    pts = matcher.points
    refiner = Refiner((*pts[:, :2].min(0), *pts[:, :2].max(0))) if n_refine else None
    metas = {m.with_suffix("").stem: m.read_text().split() for m in Path(root).glob("*/*.png.txt")}
    results = []
    for name, image, _, focal in load_ch1(root):
        lat, lon = float(metas[name][1]), float(metas[name][2])
        e, n = TO_LV95.transform(lon, lat)
        sky = predict(model, image, "cuda")[0] > 0.5
        cands, diag = localize(matcher, refiner, sky, focal, n_refine, truth=(e, n))
        d = [np.hypot(c["e"] - e, c["n"] - n) for c in cands]
        hit = next((k + 1 for k, x in enumerate(d) if x < 1000), None)
        r = {"id": name, "top1_m": float(d[0]), "first_hit": hit, "best_rms": cands[0]["rms"], **diag, "top": cands}
        results.append(r)
        print(f"{name}: top1 {d[0] / 1000:.1f} km, 1er bon {hit}, rms vrai {diag.get('true_rms') or float('nan'):.3f} "
              f"meilleur {cands[0]['rms']:.3f}", flush=True)
    n_img = len(results)
    first = [r["first_hit"] for r in results]
    summary = {"n_images": n_img,
               **{f"recall@{k}": sum(1 for f in first if f is not None and f <= k) / n_img for k in (1, 5, 20)},
               "median_top1_km": float(np.median([r["top1_m"] for r in results]) / 1000),
               "search_misses": sum(1 for r in results if r["first_hit"] != 1 and r.get("true_rms") is not None
                                    and r["true_rms"] < r["best_rms"])}
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "images": results}, open(out_dir / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
