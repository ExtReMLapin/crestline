"""Liste les photos du test où un modèle fait les plus grosses erreurs de skyline (et les dessine).

Usage : python scripts/worst_cases.py <liste_test> <dossier_labels> <sortie> <ckpt_a> <ckpt_b> [n]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_ridges import predict  # noqa: E402
from eval_smapshot import skyline_cols  # noqa: E402
from ridge_data import top_sky_rows  # noqa: E402
from train_ridges import load_trained  # noqa: E402


def main(test_list, labels_dir, out_dir, ckpt_a, ckpt_b, n=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ma, mb = load_trained(ckpt_a, "cuda"), load_trained(ckpt_b, "cuda")
    rows = []
    for sid in [l.strip() for l in open(test_list) if l.strip()]:
        z = np.load(Path(labels_dir) / f"{sid}.npz")
        pa, pb = predict(ma, z["image"], "cuda"), predict(mb, z["image"], "cuda")
        ea = skyline_cols(pa[0] > 0.5, z["sky"], z["ignore"])
        eb = skyline_cols(pb[0] > 0.5, z["sky"], z["ignore"])
        if ea.size:
            rows.append((sid, ea.mean(), eb.mean(), ea.size, pa, pb, z))
    total = sum(r[1] * r[3] for r in rows)
    rows.sort(key=lambda r: -r[1] * r[3])
    print(f"part de l'erreur totale de A due aux {n} pires photos : "
          f"{sum(r[1] * r[3] for r in rows[:int(n)]) / total:.0%}")
    for sid, ea, eb, ncol, pa, pb, z in rows[:int(n)]:
        print(f"{sid}: erreur moyenne A {ea:.1f} px, B {eb:.1f} px ({ncol} colonnes)")
        img = z["image"]
        tiles = []
        for p, color in ((pa, (255, 0, 0)), (pb, (0, 128, 255))):
            a = img.copy()
            r = top_sky_rows(p[0] > 0.5)
            g = top_sky_rows(z["sky"])
            cols = np.arange(a.shape[1])
            a[np.clip(g, 0, a.shape[0] - 1), cols] = (0, 255, 0)
            a[np.clip(r, 0, a.shape[0] - 1), cols] = color
            a[z["ignore"]] = (a[z["ignore"]] * 0.5).astype(np.uint8)
            tiles.append(a)
        Image.fromarray(np.concatenate(tiles, axis=1)).save(out_dir / f"{sid}.jpg", quality=85)


if __name__ == "__main__":
    main(*sys.argv[1:7])
