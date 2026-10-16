"""Recouvrement entre un jeu de test et les photos d'entraînement du modèle.

1. Présélection par empreinte visuelle (dHash 256 bits, Hamming <= MAX_HAMMING) : rapide, mais les photos de
   montagne se ressemblent toutes à petite échelle (ciel clair en haut, sombre en bas).
2. Vérification des paires suspectes : corrélation des miniatures 64 × 64 en niveaux de gris, la meilleure sur
   ±SHIFT pixels de décalage (tolère un léger recadrage). >= MIN_CORR : à regarder à l'œil.
   Une copie recadrée à 2 % et recompressée score 0,76-0,99 ; deux photos différentes au même cadrage
   (ciel uni en haut, relief sombre en bas) peuvent aussi dépasser 0,95. Le verdict final est visuel.

Usage : python scripts/overlap_check.py <dossier_test> <liste_ids_entraînement> [dossier_photos_entraînement]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

MAX_HAMMING = 64
MIN_CORR = 0.9
SHIFT = 4
HASH = 16


def dhash(path, size=HASH):
    try:
        im = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR), np.int16)
    except OSError:
        return None
    return np.packbits((im[:, 1:] > im[:, :-1]).ravel())


def thumb(path):
    return np.asarray(Image.open(path).convert("L").resize((64, 64), Image.BILINEAR), np.float32)


def best_corr(a, b, shift=SHIFT):
    """Corrélation normalisée du centre de a avec b, maximisée sur les décalages."""
    c = a[shift:-shift, shift:-shift].ravel()
    c = (c - c.mean()) / (np.linalg.norm(c - c.mean()) + 1e-6)
    best = -1.0
    for dy in range(2 * shift + 1):
        for dx in range(2 * shift + 1):
            w = b[dy:dy + 64 - 2 * shift, dx:dx + 64 - 2 * shift].ravel()
            w = w - w.mean()
            best = max(best, float(c @ w / (np.linalg.norm(w) + 1e-6)))
    return best


def hashes(paths):
    with ProcessPoolExecutor(16) as ex:
        return list(ex.map(dhash, paths, chunksize=64))


def main(test_dir, train_list, train_dir="data/smapshot/out/photos"):
    test = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    train = [Path(train_dir) / f"{i}.jpg" for i in Path(train_list).read_text().split()]
    ht, hr = hashes(test), hashes(train)
    ok_t = [(p, h) for p, h in zip(test, ht) if h is not None]
    ok_r = [(p, h) for p, h in zip(train, hr) if h is not None]
    print(f"{len(ok_t)} photos de test, {len(ok_r)} photos d'entraînement lues")
    a = np.stack([h for _, h in ok_t])
    b = np.stack([h for _, h in ok_r])
    pairs = []
    for i in range(0, len(a), 64):
        d = np.unpackbits(a[i:i + 64, None] ^ b[None], axis=-1).sum(-1)
        pairs += [(i + r, c) for r, c in zip(*np.nonzero(d <= MAX_HAMMING))]
    print(f"paires présélectionnées (Hamming <= {MAX_HAMMING} / {HASH * HASH}) : {len(pairs)}")
    found = []
    for r, c in pairs:
        corr = best_corr(thumb(ok_t[r][0]), thumb(ok_r[c][0]))
        if corr >= MIN_CORR:
            found.append((ok_t[r][0].name, ok_r[c][0].name, corr))
    print(f"même photo (corrélation >= {MIN_CORR}) : {len(found)}")
    for t, r, corr in sorted(found, key=lambda x: -x[2])[:40]:
        print(f"  {t}  ~  {r}  ({corr:.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
