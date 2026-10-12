"""Démo : une carte de 60 × 60 km autour du Mont-Blanc et des Fiz, et une photo du lac d'Anterne.

Le MNT (swissALTIRegio, swisstopo, 10 m, couvre aussi les régions voisines) est un COG de 10 Go : on n'en
lit que deux fenêtres à distance (~quelques centaines de Mo) :
- zone + 30 km à 10 m (relief proche et moyen) ;
- zone + 120 km à 80 m (horizon lointain).
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import rasterio

from .dem import FINE_MARGIN, MAX_DIST, read_window
from .maps import Map
from .net import cache_root, user_agent

REGIO_URL = ("/vsicurl/https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio/"
             "swissaltiregio_2056_5728.tif")
DEMO_BOUNDS = (2530000.0, 1070000.0, 2590000.0, 1130000.0)   # LV95 : 60 × 60 km, Mont-Blanc, Fiz, Sixt
HERE = Path(__file__).parent / "demo"


def cache_dir(override=None):
    """--cache, sinon $CRESTLINE_CACHE, sinon $XDG_CACHE_HOME/crestline, sinon ~/.cache/crestline."""
    return Path(override) if override else cache_root()


def fetch_dem(out_dir):
    """Télécharge les deux fenêtres du MNT de démo (une seule fois)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fine, coarse = out_dir / "dem_10m.tif", out_dir / "dem_80m.tif"
    x0, y0, x1, y1 = DEMO_BOUNDS
    for path, margin, cell in ((fine, FINE_MARGIN, 10.0), (coarse, MAX_DIST, 80.0)):
        if path.exists():
            continue
        print(f"téléchargement du MNT ({cell:.0f} m, zone + {margin / 1000:.0f} km)...", flush=True)
        b = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        with rasterio.Env(GDAL_HTTP_USERAGENT=user_agent()):
            z, xmin, ymax = read_window(REGIO_URL, b, cell)
        profile = {"driver": "GTiff", "dtype": "float32", "width": z.shape[1], "height": z.shape[0], "count": 1,
                   "crs": "EPSG:2056", "transform": rasterio.transform.from_origin(xmin, ymax, cell, cell),
                   "nodata": np.nan, "compress": "deflate", "predictor": 3, "tiled": True}
        tmp = path.with_suffix(".part.tif")
        with rasterio.open(tmp, "w", **profile) as f:
            f.write(z, 1)
        shutil.move(tmp, path)
    return fine, coarse


def demo_map(step=125.0, cache=None, devices=None):
    """Carte de démo (construite une fois, puis en cache) : horizons + index."""
    root = cache_dir(cache)
    m_dir = root / f"demo-map-{int(step)}m"
    if (m_dir / "elev.npy").exists() and (m_dir / "index").exists():
        return Map(m_dir)
    fine, coarse = fetch_dem(root / "demo-dem")
    m = Map.create(m_dir, fine, bounds=DEMO_BOUNDS, step=step, dem_coarse=coarse)
    print("calcul des horizons de la carte de démo...", flush=True)
    m.build(devices)
    print("construction de l'index...", flush=True)
    m.build_index()
    return m


def expected():
    info = json.loads((HERE / "lac_anterne.json").read_text())
    return HERE / "lac_anterne.jpg", info


def distance_m(lat0, lon0, lat1, lon1):
    k = 111_320.0
    return math.hypot((lat1 - lat0) * k, (lon1 - lon0) * k * math.cos(math.radians(lat0)))
