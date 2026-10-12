"""MNT mondiaux reprojetés sur la même grille que swissALTIRegio (LV95, 10 m), pour les comparer avec
exactement le même code.

Usage : python scripts/copernicus_dem.py <sortie.tif> [copernicus|fabdem]

- copernicus : GLO-30 (30 m, bucket public AWS), modèle de surface (arbres et bâtiments compris), EGM2008.
- fabdem : Copernicus dont forêts et bâtiments ont été retirés (Université de Bristol, CC BY-NC-SA 4.0).
  Les archives 10°×10° doivent être dans data/dem/fabdem/ (N40E000-N50E010.zip, N40E010-N50E020.zip).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rasterio.merge
from rasterio.warp import Resampling, reproject

URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
TILES = Path("data/dem/copernicus")
REGIO = "data/dem/swissaltiregio_2056_5728.tif"


def tile_name(lat, lon):
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def download(bounds_wgs84):
    TILES.mkdir(parents=True, exist_ok=True)
    lon0, lat0, lon1, lat1 = bounds_wgs84
    out = []
    for lat in range(int(np.floor(lat0)), int(np.floor(lat1)) + 1):
        for lon in range(int(np.floor(lon0)), int(np.floor(lon1)) + 1):
            name = tile_name(lat, lon)
            local = TILES / f"{name}.tif"
            if not local.exists():
                tmp = local.with_suffix(".part")
                urllib.request.urlretrieve(URL.format(n=name), tmp)
                tmp.rename(local)
            out.append(local)
    return out


def fabdem_tiles(bounds_wgs84, root=Path("data/dem/fabdem")):
    """Extrait des archives FABDEM les tuiles 1°×1° couvrant l'emprise."""
    lon0, lat0, lon1, lat1 = bounds_wgs84
    wanted = {f"N{lat:02d}E{lon:03d}_FABDEM_V1-2.tif"
              for lat in range(int(np.floor(lat0)), int(np.floor(lat1)) + 1)
              for lon in range(int(np.floor(lon0)), int(np.floor(lon1)) + 1)}
    (root / "tiles").mkdir(parents=True, exist_ok=True)
    out = []
    for z in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(z) as f:
            for member in f.namelist():
                name = Path(member).name
                if name in wanted:
                    local = root / "tiles" / name
                    if not local.exists():
                        local.write_bytes(f.read(member))
                    out.append(local)
    return out


def main(out_path, source="copernicus"):
    with rasterio.open(REGIO) as ref:
        profile, bounds = ref.profile, ref.bounds
    to_wgs = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    xs, ys = to_wgs.transform([bounds.left, bounds.right, bounds.left, bounds.right],
                              [bounds.bottom, bounds.bottom, bounds.top, bounds.top])
    bbox = (min(xs), min(ys), max(xs), max(ys))
    tiles = download(bbox) if source == "copernicus" else fabdem_tiles(bbox)
    print(len(tiles), "tuiles")
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, src_transform = rasterio.merge.merge(srcs, nodata=np.nan)
    src_crs = srcs[0].crs
    for s in srcs:
        s.close()
    dst = np.full((profile["height"], profile["width"]), np.nan, np.float32)
    reproject(mosaic[0].astype(np.float32), dst, src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=profile["transform"], dst_crs="EPSG:2056", dst_nodata=np.nan,
              resampling=Resampling.bilinear, num_threads=16)
    profile.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                   blockxsize=512, blockysize=512, BIGTIFF="YES")
    with rasterio.open(out_path, "w", **profile) as f:
        f.write(dst, 1)
    print("écrit", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
