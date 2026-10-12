"""Dégrade swissALTIRegio (10 m, sol nu) à une maille plus grossière, sur la même grille 10 m (moyenne par blocs
puis interpolation bilinéaire). Sert à séparer l'effet de la résolution de celui des arbres/bâtiments
quand on compare avec Copernicus (30 m, surface).

Usage : python scripts/degrade_dem.py <sortie.tif> [facteur=3]
"""
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

REGIO = "data/dem/swissaltiregio_2056_5728.tif"


def main(out_path, factor=3):
    factor = int(factor)
    with rasterio.open(REGIO) as src:
        profile = src.profile
        coarse = src.read(1, out_shape=(src.height // factor, src.width // factor),
                          resampling=Resampling.average).astype(np.float32)
        coarse[coarse > 1e30] = np.nan
        coarse_transform = src.transform * src.transform.scale(src.width / coarse.shape[1], src.height / coarse.shape[0])
    dst = np.full((profile["height"], profile["width"]), np.nan, np.float32)
    reproject(coarse, dst, src_transform=coarse_transform, src_crs="EPSG:2056", src_nodata=np.nan,
              dst_transform=profile["transform"], dst_crs="EPSG:2056", dst_nodata=np.nan,
              resampling=Resampling.bilinear, num_threads=16)
    profile.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                   blockxsize=512, blockysize=512, BIGTIFF="YES")
    with rasterio.open(out_path, "w", **profile) as f:
        f.write(dst, 1)
    print("écrit", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
