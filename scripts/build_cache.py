"""Construit un cache d'entraînement brut (uint8, non compressé) à partir de data/geopose_prep.

Chaque photo devient <cache>/<id>.npy de forme (H, W, 7) uint8 :
  0-2 image RGB | 3 ciel | 4 skyline adoucie | 5 crêtes adoucies | 6 à ignorer   (0..255)
Les cibles adoucies sont calculées ici une fois pour toutes. Les fichiers sont lus
en mmap pendant l'entraînement : une seule copie en RAM (cache de pages), partagée
par tous les processus.

Usage : python scripts/build_cache.py <prep> <cache> [n_procs]
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from ridge_data import soft_line  # noqa: E402


def build(args):
    src, dst = args
    if dst.exists():
        return
    z = np.load(src)
    out = np.empty((*z["sky"].shape, 7), np.uint8)
    out[..., :3] = z["image"]
    out[..., 3] = z["sky"] * 255
    out[..., 4] = np.round(soft_line(z["skyline"]) * 255)
    out[..., 5] = np.round(soft_line(z["ridge"]) * 255)
    out[..., 6] = z["ignore"] * 255
    tmp = dst.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.rename(dst)


def main(prep, cache, n_procs=16):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(p, cache / f"{p.stem}.npy") for p in sorted(Path(prep).glob("*.npz"))]
    with Pool(int(n_procs)) as pool:
        for i, _ in enumerate(pool.imap_unordered(build, jobs, chunksize=8)):
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
