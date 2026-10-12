"""Télécharge le RGE ALTI de l'IGN (Géoplateforme, WMS-R en valeurs brutes float32) sur une zone, en Lambert-93,
et complète les trous (hors de France) avec swissALTIRegio reprojeté.

Usage : python scripts/ign_dem.py <sortie.tif> <lon0,lat0,lon1,lat1> [maille_m=5] [marge_km=30] [regio.tif]

Licence des données IGN : Licence Ouverte Etalab 2.0.
"""
import sys
import time
import urllib.request

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

WMS = ("https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
       "&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES&STYLES=&CRS=EPSG:2154&FORMAT=image/x-bil;bits=32"
       "&BBOX={x0},{y0},{x1},{y1}&WIDTH={w}&HEIGHT={h}")
TILE = 2000                       # pixels par côté de tuile


def fetch_tile(x0, y0, x1, y1, w, h):
    for k in range(4):
        try:
            with urllib.request.urlopen(WMS.format(x0=x0, y0=y0, x1=x1, y1=y1, w=w, h=h), timeout=120) as r:
                return np.frombuffer(r.read(), "<f4").reshape(h, w)
        except Exception:
            if k == 3:
                raise
            time.sleep(2 ** k)


def main(out, bbox, cell=5.0, margin_km=30.0, regio="data/dem/swissaltiregio_2056_5728.tif"):
    cell, margin = float(cell), float(margin_km) * 1000
    lo0, la0, lo1, la1 = (float(v) for v in bbox.split(","))
    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    xs, ys = t.transform([lo0, lo1, lo0, lo1], [la0, la0, la1, la1])
    x0, y0 = (np.floor((min(xs) - margin) / 1000) * 1000, np.floor((min(ys) - margin) / 1000) * 1000)
    x1, y1 = (np.ceil((max(xs) + margin) / 1000) * 1000, np.ceil((max(ys) + margin) / 1000) * 1000)
    W, H = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    z = np.full((H, W), np.nan, np.float32)
    span = TILE * cell
    n = 0
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            w, h = min(TILE, W - tx), min(TILE, H - ty)
            bx0, by1 = x0 + tx * cell, y1 - ty * cell
            z[ty:ty + h, tx:tx + w] = fetch_tile(bx0, by1 - h * cell, bx0 + w * cell, by1, w, h)
            n += 1
            print(f"tuile {n} ({tx // TILE}, {ty // TILE}) : {span / 1000:.0f} km", flush=True)
    z[(z < -1000) | (z > 9000)] = np.nan                     # hors France : valeur de remplissage
    print(f"couverture IGN : {np.isfinite(z).mean():.1%}")
    transform = from_origin(x0, y1, cell, cell)
    if not np.isfinite(z).all():
        fill = np.full_like(z, np.nan)
        with rasterio.open(regio) as src:
            reproject(rasterio.band(src, 1), fill, dst_transform=transform, dst_crs="EPSG:2154",
                      dst_nodata=np.nan, resampling=Resampling.bilinear, num_threads=8)
        holes = ~np.isfinite(z)
        z[holes] = fill[holes]
        print(f"trous complétés par swissALTIRegio : {holes.mean():.1%}")
    profile = {"driver": "GTiff", "dtype": "float32", "width": W, "height": H, "count": 1, "crs": "EPSG:2154",
               "transform": transform, "nodata": np.nan, "compress": "deflate", "predictor": 3, "tiled": True,
               "BIGTIFF": "YES"}
    with rasterio.open(out, "w", **profile) as f:
        f.write(z, 1)
    print("écrit", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
