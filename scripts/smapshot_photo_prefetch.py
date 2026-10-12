"""Pré-télécharge en parallèle les photos smapshot (IIIF ETH, ~7 s par photo côté serveur).

Usage : python scripts/smapshot_photo_prefetch.py <selection.csv> <dossier_out> [threads] [largeur]
Les photos vont dans <dossier_out>/photos/<id>.jpg (réutilisées par smapshot_labels.py).
"""
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import smapshot as S  # noqa: E402


def main(selection_csv, out_dir, threads=12, width=1024):
    photos = Path(out_dir) / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(selection_csv)
    todo = [(r.id, r.image_url) for r in sel.itertuples() if not (photos / f"{r.id}.jpg").exists()]
    print(f"{len(todo)} photos à télécharger", flush=True)

    def fetch(job):
        image_id, url = job
        dst = photos / f"{image_id}.jpg"
        tmp = dst.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part.jpg")
        try:
            S.download_photo(url, tmp, int(width))
            os.replace(tmp, dst)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            return False

    ok = 0
    with ThreadPoolExecutor(int(threads)) as pool:
        for k, r in enumerate(pool.map(fetch, todo)):
            ok += r
            if k % 500 == 0:
                print(f"{k}/{len(todo)} ({ok} ok)", flush=True)
    print(f"terminé : {ok}/{len(todo)}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
