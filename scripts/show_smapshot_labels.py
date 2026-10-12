"""Planches de revue des étiquettes smapshot.

Usage : python scripts/show_smapshot_labels.py <dossier_out> <dossier_planches>
Par photo : [photo] [lignes MNT avant affinage en cyan, après en rouge (skyline) / jaune (crêtes)]
[étiquettes finales : zones ignorées assombries, rouge / jaune = lignes gardées].
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from smapshot_labels import depth_lines  # noqa: E402


def main(out_dir, sheet_dir, id_list=None):
    out_dir, sheet_dir = Path(out_dir), Path(sheet_dir)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    metas = {}
    for f in out_dir.glob("meta_*.jsonl"):
        for l in open(f):
            m = json.loads(l)
            metas[str(m["id"])] = m
    files = sorted((out_dir / "labels").glob("*.npz"))
    if id_list:
        wanted = {l.strip() for l in open(id_list) if l.strip()}
        files = [f for f in files if f.stem in wanted]
    for npz in files:
        z = np.load(npz)
        img = z["image"]
        thick = lambda m: ndimage.binary_dilation(m, iterations=1)  # noqa: E731
        _, sk0, rd0 = depth_lines(np.where(z["distance_before"] == -2, 1e9, z["distance_before"]))
        a = img.copy()
        a[thick(sk0 | rd0)] = (0, 255, 255)
        a[thick(z["ridge"])] = (255, 200, 0)
        a[thick(z["skyline"])] = (255, 0, 0)
        b = img.copy()
        b[z["ignore"]] = (b[z["ignore"]] * 0.35).astype(np.uint8)
        b[thick(z["ridge"] & ~z["ignore"])] = (255, 200, 0)
        b[thick(z["skyline"] & ~z["ignore"])] = (255, 0, 0)
        m = metas.get(npz.stem, {})
        h, w = img.shape[:2]
        sheet = Image.new("RGB", (3 * w + 20, h + 24), "white")
        for i, (t, im) in enumerate([("photo", img), ("cyan = MNT avant affinage, rouge/jaune = apres", a),
                                     ("etiquettes finales (sombre = ignore)", b)]):
            sheet.paste(Image.fromarray(im), (i * (w + 10), 24))
            ImageDraw.Draw(sheet).text((i * (w + 10) + 4, 5), t, fill="black")
        title = (f"{npz.stem} {m.get('date', '')} {str(m.get('title', '')).strip()[:50]} | rot ({m.get('yaw', 0):+.2f},"
                 f"{m.get('pitch', 0):+.2f},{m.get('roll', 0):+.2f}) ech {m.get('scale', 1):.3f}")
        ImageDraw.Draw(sheet).text((2 * (w + 10) + 4, 13), title, fill="gray")
        sheet.save(sheet_dir / f"{npz.stem}.jpg", quality=88)
    print("planches :", len(list(sheet_dir.glob("*.jpg"))))


if __name__ == "__main__":
    main(*sys.argv[1:4])
