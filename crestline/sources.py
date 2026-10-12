"""Téléchargement du relief (MNT) depuis des sources publiques, pour une zone donnée en lon/lat.

    zone demandée (--bbox)
    ┌──────────────────────────────┐
    │  zone + 30 km  → maille fine (relief proche : c'est là que la précision compte)
    │  ┌────────────────────────┐  │
    │  │   points de vue        │  │
    │  └────────────────────────┘  │
    └──────────────────────────────┘
      zone + 120 km → maille 80 m  (horizon lointain)

Les trous (hors de la couverture de la source) sont comblés par la meilleure source suivante :
    ign (France, lidar 1-5 m) → swisstopo (Suisse + ~100 km, 10 m) → copernicus (monde, 30 m, radar)

| source     | couverture          | maille | licence                          |
|------------|---------------------|--------|----------------------------------|
| ign        | France              | 5 m    | Licence Ouverte Etalab 2.0 (IGN) |
| swisstopo  | Suisse et voisins   | 10 m   | libre, attribution « © swisstopo » |
| copernicus | monde               | 30 m   | libre (Copernicus DEM, ESA)      |

Copernicus est un MNT radar : 20 à 40 m d'erreur sur les arêtes, la localisation y est nettement moins bonne.
"""
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from .dem import FINE_MARGIN, MAX_DIST
from .net import country_at, get, user_agent

SWISSTOPO_URL = ("/vsicurl/https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio/"
                 "swissaltiregio_2056_5728.tif")
COPERNICUS_URL = "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
IGN_WMS = ("https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
           "&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES&STYLES=&CRS=EPSG:2154&FORMAT=image/x-bil;bits=32"
           "&BBOX={x0},{y0},{x1},{y1}&WIDTH={w}&HEIGHT={h}")
COARSE_CELL = 80.0
SOURCES = {   # source : (CRS natif, maille par défaut, sources de repli)
    "ign": ("EPSG:2154", 5.0, ("swisstopo", "copernicus")),
    "swisstopo": ("EPSG:2056", 10.0, ("copernicus",)),
    "copernicus": (None, 30.0, ()),
}
ATTRIBUTION = {"ign": "IGN, RGE ALTI (Licence Ouverte Etalab 2.0)", "swisstopo": "© swisstopo, swissALTIRegio",
               "copernicus": "Copernicus DEM GLO-30, © DLR e.V. 2010-2014 et © Airbus 2014-2018, ESA"}


class Grid:
    def __init__(self, crs, x0, y0, x1, y1, cell):
        self.crs, self.cell = crs, cell
        self.x0 = math.floor(x0 / cell) * cell
        self.y1 = math.ceil(y1 / cell) * cell
        self.w = int(math.ceil((x1 - self.x0) / cell))
        self.h = int(math.ceil((self.y1 - y0) / cell))
        self.transform = from_origin(self.x0, self.y1, cell, cell)

    def lonlat_bounds(self):
        t = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        xs = [self.x0, self.x0 + self.w * self.cell]
        ys = [self.y1 - self.h * self.cell, self.y1]
        lon, lat = t.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        return min(lon), min(lat), max(lon), max(lat)


def utm_crs(lon, lat):
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"


# ---------- lecteurs : remplissent une grille ----------

def read_cog(url, grid, tiles_dir=None):
    """Lit un GeoTIFF distant (COG) sur la grille, en prenant l'aperçu dont la maille convient."""
    try:
        with rasterio.Env(GDAL_HTTP_USERAGENT=user_agent()), rasterio.open(url) as full:
            res = abs(full.res[0]) if full.crs.is_projected else abs(full.res[0]) * 111_320
            factors = full.overviews(1)
            level = None
            for k, f in enumerate(factors):
                if res * f <= grid.cell:
                    level = k
        kw = {"overview_level": level} if level is not None else {}
        with rasterio.Env(GDAL_HTTP_USERAGENT=user_agent()), rasterio.open(url, **kw) as src, WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.w,
                                                         height=grid.h, resampling=Resampling.bilinear,
                                                         src_nodata=src.nodata, nodata=np.nan) as vrt:
            z = vrt.read(1).astype(np.float32)
    except rasterio.errors.RasterioIOError:
        return np.full((grid.h, grid.w), np.nan, np.float32)        # hors couverture (tuile absente)
    z[(z < -1000) | (z > 9000)] = np.nan
    return z


def read_swisstopo(grid, tiles_dir=None):
    return read_cog(SWISSTOPO_URL, grid)


def read_copernicus(grid, tiles_dir=None):
    lo0, la0, lo1, la1 = grid.lonlat_bounds()
    z = np.full((grid.h, grid.w), np.nan, np.float32)
    for lat in range(math.floor(la0), math.floor(la1) + 1):
        for lon in range(math.floor(lo0), math.floor(lo1) + 1):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            t = read_cog(COPERNICUS_URL.format(n=name), grid)
            holes = ~np.isfinite(z)
            z[holes] = t[holes]
    return z


def read_ign(grid, tiles_dir=None, tile=2000, workers=4):
    """Service WMS-R de la Géoplateforme IGN, valeurs brutes float32, par tuiles de 2000 px (Lambert-93).
    Chaque tuile est gardée dans `tiles_dir` : un téléchargement interrompu reprend où il s'était arrêté."""
    if pyproj.CRS.from_user_input(grid.crs) != pyproj.CRS.from_epsg(2154):
        raise ValueError("la source ign s'utilise en Lambert-93 (EPSG:2154)")
    z = np.full((grid.h, grid.w), np.nan, np.float32)
    jobs = [(ty, tx) for ty in range(0, grid.h, tile) for tx in range(0, grid.w, tile)]

    if tiles_dir is not None:
        Path(tiles_dir).mkdir(parents=True, exist_ok=True)

    def fetch(job):
        ty, tx = job
        w, h = min(tile, grid.w - tx), min(tile, grid.h - ty)
        x0, y1 = grid.x0 + tx * grid.cell, grid.y1 - ty * grid.cell
        cached = Path(tiles_dir) / f"{x0:.0f}_{y1:.0f}_{grid.cell:g}_{w}x{h}.npy" if tiles_dir else None
        if cached is not None and cached.exists():
            return job, np.load(cached)
        url = IGN_WMS.format(x0=x0, y0=y1 - h * grid.cell, x1=x0 + w * grid.cell, y1=y1, w=w, h=h)
        for k in range(5):
            try:
                t = np.frombuffer(get(url, timeout=120), "<f4").reshape(h, w)
                break
            except Exception:
                if k == 4:
                    raise
                time.sleep(2 ** k)
        if cached is not None:
            np.save(cached.with_suffix(".part.npy"), t)
            cached.with_suffix(".part.npy").rename(cached)
        return job, t

    with ThreadPoolExecutor(workers) as pool:
        for n, ((ty, tx), t) in enumerate(pool.map(fetch, jobs), 1):
            z[ty:ty + t.shape[0], tx:tx + t.shape[1]] = t
            print(f"\r  IGN : tuile {n}/{len(jobs)}", end="", flush=True)
    print()
    z[(z < -1000) | (z > 9000)] = np.nan                               # hors France : valeur de remplissage
    return z


READERS = {"ign": read_ign, "swisstopo": read_swisstopo, "copernicus": read_copernicus}


# ---------- assemblage ----------

def build_raster(grid, chain, path):
    tiles_dir = Path(str(path) + ".tuiles")                          # reprise après interruption
    z = np.full((grid.h, grid.w), np.nan, np.float32)
    shares = {}
    for name in chain:
        holes = ~np.isfinite(z)
        if not holes.any():
            break
        print(f"  {name} ({grid.cell:g} m, {grid.w}×{grid.h} px)...", flush=True)
        part = READERS[name](grid, tiles_dir / name)
        filled = holes & np.isfinite(part)
        z[filled] = part[filled]
        shares[name] = float(filled.mean())
    profile = {"driver": "GTiff", "dtype": "float32", "width": grid.w, "height": grid.h, "count": 1,
               "crs": grid.crs, "transform": grid.transform, "nodata": np.nan, "compress": "deflate",
               "predictor": 3, "tiled": True, "BIGTIFF": "IF_SAFER"}
    with rasterio.open(path, "w", **profile) as f:
        f.write(z, 1)
    shutil.rmtree(tiles_dir, ignore_errors=True)                     # fichier complet : tuiles inutiles
    shares["vide"] = float((~np.isfinite(z)).mean())
    return shares


def already_fetched(out_dir, source, bbox_lonlat, cell):
    """Relief déjà téléchargé dans out_dir pour la même source et la même maille, sur une zone qui contient
    celle demandée (les marges sont alors couvertes aussi)."""
    out_dir = Path(out_dir)
    prov_path = out_dir / "dem_provenance.json"
    if not (prov_path.exists() and (out_dir / "dem_fine.tif").exists() and (out_dir / "dem_coarse.tif").exists()):
        return None
    prov = json.loads(prov_path.read_text())
    a0, b0, a1, b1 = prov["bbox_lonlat"]
    lo0, la0, lo1, la1 = bbox_lonlat
    if prov["source"] == source and prov["cell"] == cell and a0 <= lo0 and b0 <= la0 and lo1 <= a1 and la1 <= b1:
        return prov
    return None


def fetch(source, bbox_lonlat, out_dir, cell=None, force=False):
    """Télécharge les deux MNT d'une carte (fin : zone + 30 km ; grossier : zone + 120 km) dans out_dir,
    ou réutilise ceux déjà présents (même source, même maille, zone contenue). Renvoie (fin, grossier, provenance)."""
    crs, default_cell, fallbacks = SOURCES[source]
    lo0, la0, lo1, la1 = bbox_lonlat
    crs = crs or utm_crs((lo0 + lo1) / 2, (la0 + la1) / 2)
    cell = float(cell or default_cell)
    prov = None if force else already_fetched(out_dir, source, bbox_lonlat, cell)
    if prov:
        print(f"relief déjà téléchargé ({source}, {cell:g} m, {prov['date']}) : réutilisé (--force pour retélécharger)")
        return Path(out_dir) / "dem_fine.tif", Path(out_dir) / "dem_coarse.tif", prov
    t = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = t.transform([lo0, lo1, lo0, lo1], [la0, la0, la1, la1])
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    fine_px = (x1 - x0 + 2 * FINE_MARGIN) * (y1 - y0 + 2 * FINE_MARGIN) / cell ** 2
    if fine_px > 1.5e9:
        raise ValueError(f"zone trop grande pour une maille de {cell:g} m ({fine_px / 1e9:.1f} milliards de pixels, "
                         "trop pour la mémoire du GPU) : augmentez --cell ou découpez la zone")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chain = (source,) + fallbacks
    fine, coarse = out_dir / "dem_fine.tif", out_dir / "dem_coarse.tif"
    m = FINE_MARGIN
    print(f"MNT fin : zone + {m / 1000:.0f} km", flush=True)
    s_fine = build_raster(Grid(crs, x0 - m, y0 - m, x1 + m, y1 + m, cell), chain, fine)
    m = MAX_DIST
    print(f"MNT grossier : zone + {m / 1000:.0f} km", flush=True)
    s_coarse = build_raster(Grid(crs, x0 - m, y0 - m, x1 + m, y1 + m, max(COARSE_CELL, cell)), chain, coarse)
    prov = {"source": source, "cell": cell, "crs": crs, "bbox_lonlat": list(bbox_lonlat),
            "date": time.strftime("%Y-%m-%d"), "fine_shares": s_fine, "coarse_shares": s_coarse,
            "attribution": [ATTRIBUTION[s] for s in chain if s_fine.get(s, 0) + s_coarse.get(s, 0) > 0]}
    (out_dir / "dem_provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    if s_fine.get("copernicus", 0) > 0.05:
        print("attention : une partie du relief proche vient de Copernicus (radar, imprécis en montagne)")
    return fine, coarse, prov


# ---------- choix automatique et estimation ----------

SWISSTOPO_EXTENT = (2385000.0, 974000.0, 2935000.0, 1404000.0)       # emprise de swissALTIRegio (LV95)


def bbox_around(lat, lon, radius_km):
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def choose_source(bbox_lonlat, country=None):
    """Meilleure source pour la zone : IGN en France, swisstopo en Suisse et autour, Copernicus ailleurs."""
    lo0, la0, lo1, la1 = bbox_lonlat
    country = country or country_at((la0 + la1) / 2, (lo0 + lo1) / 2)
    if country in ("fr", "mc"):
        return "ign"
    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    xs, ys = t.transform([lo0, lo1, lo0, lo1], [la0, la0, la1, la1])
    x0, y0, x1, y1 = SWISSTOPO_EXTENT
    if country in ("ch", "li") or (min(xs) > x0 and max(xs) < x1 and min(ys) > y0 and max(ys) < y1):
        return "swisstopo"
    print("attention : pas de relief précis connu pour cette zone, Copernicus (radar, 30 m) sera utilisé ; "
          "la localisation y est nettement moins fiable. Fournissez un MNT lidar avec --dem si vous en avez un.")
    return "copernicus"


def estimate(source, bbox_lonlat, cell=None, step=125.0, download=True, build=True, index=True):
    """Ordres de grandeur avant de lancer : téléchargement, disque, mémoire GPU, temps."""
    cell = float(cell or SOURCES[source][1])
    lo0, la0, lo1, la1 = bbox_lonlat
    w = (lo1 - lo0) * 111_320 * math.cos(math.radians((la0 + la1) / 2))
    h = (la1 - la0) * 111_320
    fine = (w + 2 * FINE_MARGIN) * (h + 2 * FINE_MARGIN) / cell ** 2
    coarse = (w + 2 * MAX_DIST) * (h + 2 * MAX_DIST) / max(COARSE_CELL, cell) ** 2
    points = w * h / step ** 2
    est = {"zone_km": (w / 1000, h / 1000), "points": points, "download_gb": 0.0, "disk_gb": 0.0,
           "gpu_gb": (fine * 4 * 1.33 + coarse * 4) / 1e9, "minutes": 0.0}
    if download:
        bytes_px = 4.0 if source == "ign" else 1.5                     # IGN : valeurs brutes ; COG : compressé
        est["download_gb"] += (fine + coarse) * bytes_px / 1e9
        est["disk_gb"] += (fine + coarse) * 2.1 / 1e9
        seconds_per_tile = 2.3 if source == "ign" else 1.2               # tuile de 4 Mpx, mesuré
        est["minutes"] += (fine + coarse) / 4e6 * seconds_per_tile / 60
    if build:
        est["disk_gb"] += points * (1440 * 2 + 24) / 1e9
        est["minutes"] += points / 4000 / 60
    if index:
        est["disk_gb"] += points * 3700 / 1e9
        est["minutes"] += points / 650_000
    return est
