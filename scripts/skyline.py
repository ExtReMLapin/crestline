"""Extraction de la skyline (ligne ciel/relief) en pleine résolution.

Entrées : les photos et les cartes de labels ADE20K de deux modèles
(outputs/seg/mask2former, outputs/seg/segformer, produites par segment_ade.py).

Étapes :
1. Masque ciel grossier = ciel connecté au bord haut (Mask2Former), nettoyé.
2. Ligne grossière par colonne = hauteur du ciel continu depuis le haut.
3. Validité par colonne : la ligne doit reposer sur du relief, pas sur un
   occultant (arbre, personne, objet), et les deux modèles doivent être d'accord.
4. Raffinement pleine résolution : programmation dynamique (Viterbi) dans une
   bande autour de la ligne grossière, sur le gradient vertical de l'image.

Sortie : outputs/skyline/<stem>.npz (y par colonne en pixels pleine résolution,
masque de validité) et une visualisation avec zooms.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from transformers import AutoConfig

OCCLUDERS = {
    "tree", "plant", "flower", "person", "apparel", "bag", "pole", "fence", "signboard",
    "column", "car", "chair", "tent", "building", "house", "palm", "pillow", "cushion",
}
BAND_PX = 16          # demi-largeur de la bande de recherche, en pixels pleine résolution
MAX_STEP_PX = 3       # saut vertical maximal entre deux colonnes voisines
DISAGREE_PX = 12      # écart toléré entre les deux modèles, en pixels pleine résolution
SMOOTH_COST = 0.15    # pénalité par pixel de saut (lissage)
MIN_EDGE = 0.35       # netteté minimale de l'arête (0..1) : en dessous, bord flou (nuage, brume)


def sky_height(labels, sky_id):
    """Hauteur du ciel continu depuis le haut, par colonne (résolution des labels)."""
    sky = labels == sky_id
    # fermeture sur une image bordée par réplication, sinon l'érosion efface les bords
    pad = 3
    sky = ndimage.binary_closing(np.pad(sky, pad, mode="edge"), iterations=2)[pad:-pad, pad:-pad]
    lab, _ = ndimage.label(sky)
    top_ids = set(np.unique(lab[0])) - {0}
    sky = np.isin(lab, list(top_ids))
    # nombre de pixels de ciel consécutifs depuis le haut
    return np.argmin(np.vstack([sky, np.zeros((1, sky.shape[1]), bool)]), axis=0)


def occluded_columns(labels, heights, occluder_ids, window=4):
    h = labels.shape[0]
    occ = np.zeros(labels.shape[1], bool)
    for x, y in enumerate(heights):
        below = labels[min(y, h - 1):min(y + window, h), x]
        occ[x] = np.isin(below, occluder_ids).mean() > 0.5
    return occ


def vertical_edge_cost(gray):
    """Coût faible là où l'image a une forte transition verticale."""
    g = ndimage.gaussian_filter(gray, 1.2)
    dy = np.abs(ndimage.sobel(g, axis=0))
    return 1.0 - dy / (np.percentile(dy, 99.5) + 1e-6)


def viterbi_refine(cost, coarse_y):
    """Chemin de coût minimal dans une bande autour de coarse_y (une valeur y par colonne)."""
    h, w = cost.shape
    offsets = np.arange(-BAND_PX, BAND_PX + 1)
    ys = np.clip(coarse_y[None, :] + offsets[:, None], 0, h - 1)       # (n_off, w)
    local = cost[ys, np.arange(w)[None, :]]                            # (n_off, w)
    n = len(offsets)
    jump = np.abs(offsets[:, None] - offsets[None, :]).astype(float)
    trans = np.where(jump <= MAX_STEP_PX, SMOOTH_COST * jump, np.inf)
    acc = local[:, 0].copy()
    back = np.zeros((n, w), np.int16)
    for x in range(1, w):
        prev = acc[:, None] + trans
        back[:, x] = np.argmin(prev, axis=0)
        acc = prev[back[:, x], np.arange(n)] + local[:, x]
    path = np.zeros(w, int)
    path[-1] = int(np.argmin(acc))
    for x in range(w - 1, 0, -1):
        path[x - 1] = back[path[x], x]
    return ys[path, np.arange(w)]


def extract(photo_path, seg_dir, id2label):
    name2id = {v: k for k, v in id2label.items()}
    sky_id = name2id["sky"]
    occ_ids = [name2id[n] for n in OCCLUDERS if n in name2id]
    image = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
    W, H = image.size
    m2f = np.load(Path(seg_dir) / "mask2former" / f"{photo_path.stem}.npz")["labels"]
    sgf = np.load(Path(seg_dir) / "segformer" / f"{photo_path.stem}.npz")["labels"]
    scale = W / m2f.shape[1]

    h_m2f = sky_height(m2f, sky_id)
    h_sgf = sky_height(sgf, sky_id)
    occ = occluded_columns(m2f, h_m2f, occ_ids)
    no_sky = h_m2f == 0
    disagree = np.abs(h_m2f - h_sgf) * scale > DISAGREE_PX
    touches_bottom = h_m2f >= m2f.shape[0] - 1

    # passage pleine résolution
    cols = np.clip((np.arange(W) / scale).astype(int), 0, m2f.shape[1] - 1)
    # interpolation linéaire de la ligne grossière (évite l'effet d'escalier)
    x_lo = (np.arange(m2f.shape[1]) + 0.5) * scale
    coarse_y = np.round(np.interp(np.arange(W), x_lo, h_m2f * scale)).astype(int)
    gray = np.asarray(image.convert("L"), np.float32) / 255.0
    cost = vertical_edge_cost(gray)
    y = viterbi_refine(cost, np.clip(coarse_y, 0, H - 1))
    edge = ndimage.uniform_filter1d(1.0 - cost[y, np.arange(W)], 25)
    blurry = edge < MIN_EDGE
    bad = (occ | no_sky | disagree | touches_bottom)[cols] | blurry
    valid = ndimage.binary_opening(~bad, iterations=15)  # retire les petits îlots valides
    reasons = {
        "occluder": float(occ.mean()), "no_sky": float(no_sky.mean()),
        "model_disagreement": float(disagree.mean()), "blurry_edge": float(blurry.mean()),
        "valid": float(valid.mean()),
    }
    return image, y, coarse_y, valid, reasons


def draw(image, y, valid, out_path, n_zooms=4, zoom=(640, 360)):
    W, H = image.size
    vis = image.copy()
    d = ImageDraw.Draw(vis)
    for x in range(W - 1):
        color = (0, 255, 0) if valid[x] else (255, 0, 0)
        d.line([(x, y[x]), (x + 1, y[x + 1])], fill=color, width=3)
    overview = vis.copy()
    overview.thumbnail((1400, 1400))
    # zooms 1:1 répartis sur la largeur, centrés sur la ligne
    xs = np.linspace(zoom[0] // 2, W - zoom[0] // 2, n_zooms).astype(int)
    crops = []
    for cx in xs:
        cy = int(np.median(y[max(cx - 50, 0):cx + 50]))
        box = (cx - zoom[0] // 2, max(0, min(cy - zoom[1] // 2, H - zoom[1])))
        crops.append(vis.crop((*box, box[0] + zoom[0], box[1] + zoom[1])))
    strip = Image.new("RGB", (zoom[0] * 2, zoom[1] * ((n_zooms + 1) // 2)), "white")
    for i, c in enumerate(crops):
        strip.paste(c, ((i % 2) * zoom[0], (i // 2) * zoom[1]))
    sheet = Image.new("RGB", (max(overview.width, strip.width), overview.height + strip.height), "white")
    sheet.paste(overview, (0, 0))
    sheet.paste(strip, (0, overview.height))
    sheet.save(out_path, quality=88)


def main(photo_dir, seg_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    id2label = AutoConfig.from_pretrained("facebook/mask2former-swin-large-ade-semantic").id2label
    for path in sorted(Path(photo_dir).glob("*.[jJ][pP][gG]")):
        image, y, coarse_y, valid, reasons = extract(path, seg_dir, id2label)
        np.savez_compressed(out_dir / f"{path.stem}.npz", y=y, coarse_y=coarse_y, valid=valid,
                            width=image.width, height=image.height)
        draw(image, y, valid, out_dir / f"{path.stem}.jpg")
        shift = np.abs(y - coarse_y)[valid]
        print(path.stem, {k: round(v, 2) for k, v in reasons.items()},
              "raffinement médian %.1f px" % (np.median(shift) if shift.size else np.nan))


if __name__ == "__main__":
    main(*sys.argv[1:4])
