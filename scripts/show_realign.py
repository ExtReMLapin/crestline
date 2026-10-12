"""Planche avant / après recalage des annotations GeoPose3K.

Usage : python scripts/show_realign.py <dossier_sortie> [n_plus_grosses] [n_hasard]
Chaque ligne : [avant] [après], skyline MNT en rouge, crêtes internes en jaune,
skyline vue par Mask2Former sur la photo en pointillés cyan (référence du recalage).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from ridge_data import top_sky_rows  # noqa: E402

PREP, REALIGNED, SEG = Path("data/geopose_prep"), Path("data/geopose_realigned"), Path("data/photo_seg")
WIDTH = 700


def draw(z, seg, title):
    a = z["image"].copy()
    thick = lambda m: ndimage.binary_dilation(m, iterations=1)  # noqa: E731
    a[thick(z["ridge"])] = (255, 200, 0)
    a[thick(z["skyline"])] = (255, 0, 0)
    rows = top_sky_rows(seg == 2)
    cols = np.arange(a.shape[1])
    dots = (cols // 6) % 2 == 0
    a[np.clip(rows[dots], 0, a.shape[0] - 1), cols[dots]] = (0, 255, 255)
    im = Image.fromarray(a)
    im = im.resize((WIDTH, int(im.height * WIDTH / im.width)))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, WIDTH, 18], fill="white")
    d.text((4, 3), title, fill="black")
    return im


def main(out_dir, n_worst=6, n_random=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # écarts calculés par realign_labels.py (score qui compte aussi le terrain MNT là où la photo voit du ciel)
    merged = REALIGNED / "meta_realign.csv"
    parts = [merged] if merged.exists() else sorted(REALIGNED.glob("meta_realign_*.csv"))  # en cours : par GPU
    meta = pd.concat([pd.read_csv(f) for f in parts]).dropna(subset=["err_before_p90", "err_after_p90"])
    meta = meta[[(REALIGNED / f"{i}.npz").exists() for i in meta.id]]
    rows = [(f"{r.id}.npz", r.err_before_p90, r.err_after_p90) for r in meta.itertuples()]
    gain = sorted(rows, key=lambda r: r[2] - r[1])
    rng = np.random.default_rng(1)
    chosen = gain[:int(n_worst)] + [rows[i] for i in rng.choice(len(rows), int(n_random), replace=False)]
    sheets = []
    for k, (name, e0, e1) in enumerate(chosen):
        seg = np.load(SEG / name)["labels"]
        kind = "plus grosse correction" if k < int(n_worst) else "au hasard"
        left = draw(np.load(PREP / name), seg, f"AVANT  (p90 skyline {e0:.0f} px)  - {kind}")
        right = draw(np.load(REALIGNED / name), seg, f"APRES  (p90 skyline {e1:.0f} px)")
        pair = Image.new("RGB", (2 * WIDTH + 10, max(left.height, right.height)), "white")
        pair.paste(left, (0, 0))
        pair.paste(right, (WIDTH + 10, 0))
        pair.save(out_dir / f"{k:02d}_{Path(name).stem[:40]}.jpg", quality=90)
        sheets.append(pair)
    print(f"{len(rows)} photos comparées ; p90 médian avant {np.median([r[1] for r in rows]):.1f} px, "
          f"après {np.median([r[2] for r in rows]):.1f} px")


if __name__ == "__main__":
    main(*sys.argv[1:4])
