"""Index global swissALTI3D (2 m) et pré-téléchargement parallèle des tuiles autour des stations.

Usage :
  python scripts/alti3d_prefetch.py index                      -> data/dem/swissalti3d_index.json
  python scripts/alti3d_prefetch.py fetch <selection.csv> [n]  -> tuiles dans data/dem/swissalti3d/

L'index associe chaque tuile de 1 km « EEEE-NNNN » à son URL la plus récente (maille 2 m).
"""
import json
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items?limit=100"
INDEX = Path("data/dem/swissalti3d_index.json")
TILE_DIR = Path("data/dem/swissalti3d")
HALF_KM = 7  # couvre la fenêtre de 6,5 km autour de chaque caméra


def build_index():
    tiles, url, pages = {}, STAC, 0
    while url:
        with urllib.request.urlopen(url, timeout=120) as r:
            page = json.load(r)
        for item in page["features"]:
            for name, asset in item["assets"].items():
                if name.endswith("_2_2056_5728.tif"):
                    tile, year = name.split("_")[2], int(name.split("_")[1])
                    if tile not in tiles or year > tiles[tile][0]:
                        tiles[tile] = (year, asset["href"], name)
        url = next((l["href"] for l in page.get("links", []) if l["rel"] == "next"), None)
        pages += 1
        if pages % 50 == 0:
            print(f"{pages} pages, {len(tiles)} tuiles", flush=True)
    INDEX.write_text(json.dumps({t: {"year": y, "href": h, "name": n} for t, (y, h, n) in tiles.items()}))
    print(f"index : {len(tiles)} tuiles")


def download(entry):
    local = TILE_DIR / entry["name"]
    if local.exists():
        return 0
    tmp = local.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
    with urllib.request.urlopen(entry["href"], timeout=60) as r:   # délai : évite un blocage infini
        tmp.write_bytes(r.read())
    os.replace(tmp, local)
    return local.stat().st_size


def fetch(selection_csv, n_threads=32):
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text())
    sel = pd.read_csv(selection_csv)
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    e, n = tr.transform(sel.longitude.values, sel.latitude.values)
    needed = set()
    for ek, nk in {(int(a // 1000), int(b // 1000)) for a, b in zip(e, n)}:
        for de in range(-HALF_KM, HALF_KM + 1):
            for dn in range(-HALF_KM, HALF_KM + 1):
                t = f"{ek + de}-{nk + dn}"
                if t in index:
                    needed.add(t)
    todo = [index[t] for t in needed if not (TILE_DIR / index[t]["name"]).exists()]
    print(f"tuiles nécessaires : {len(needed)}, à télécharger : {len(todo)}", flush=True)
    done = 0
    with ThreadPoolExecutor(int(n_threads)) as pool:
        for k, size in enumerate(pool.map(download, todo)):
            done += size
            if k % 1000 == 0:
                print(f"{k}/{len(todo)} ({done / 1e9:.1f} Go)", flush=True)
    print("terminé")


if __name__ == "__main__":
    if sys.argv[1] == "index":
        build_index()
    else:
        fetch(*sys.argv[2:4])
