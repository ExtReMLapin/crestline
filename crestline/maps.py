"""Une « carte » = un dossier autonome : MNT + base d'horizons (+ index optionnel).

    ma_carte/
    ├── map.json        zone, CRS, pas de grille, chemins des MNT
    ├── points.npy      (N, 3) x, y, altitude du sol de chaque point de vue
    ├── elev.npy        (N, 1440) horizon de chaque point, float16, un azimut tous les 0,25°
    └── index/          optionnel (mode index) : index de mots de crête, en morceaux

    Grille (pas 125 m par défaut)            pour chaque point : un panorama
    ┌─┬─┬─┬─┬─┬─┐
    ├─┼─┼─●─┼─┼─┤ ──────▶   élévation   ___╱╲__╱╲╱╲___╱╲__
    └─┴─┴─┴─┴─┴─┘           (degrés)     0°  N   90° E  180° ... 360°
"""
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pyproj
import torch

from .dem import Dem, ensure_projected, raster_info
from .horizon import CAM_HEIGHT, horizon
from .index import RidgeIndex, ShardedIndex

N_AZ = 1440
INDEX_SHARD = 2_000_000          # points par morceau d'index (borne la mémoire de construction)


def gpu_devices():
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] if n else ["cpu"]


class Map:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "map.json").read_text())
        self.crs = pyproj.CRS.from_user_input(self.meta["crs"])
        self._dem, self._index = None, None

    # ---------- création ----------

    @classmethod
    def create(cls, path, dem, bounds=None, bbox_lonlat=None, step=125.0, dem_coarse=None, crs=None,
               camera_height=CAM_HEIGHT, force=False):
        """Nouvelle carte. Zone : `bounds` (xmin, ymin, xmax, ymax dans le CRS de la carte) ou `bbox_lonlat`
        (lon_min, lat_min, lon_max, lat_max), sinon tout le MNT. CRS de la carte : `crs`, sinon celui du MNT
        (un MNT en degrés est reprojeté en UTM). Les reprojections sont écrites dans le dossier de la carte."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        dem = ensure_projected(dem, path / "dem_projected.tif", crs).resolve()
        if dem_coarse:
            dem_coarse = ensure_projected(dem_coarse, path / "dem_coarse_projected.tif", raster_info(dem)[0])
        crs, cell, dem_bounds = raster_info(dem)
        if bounds is None:
            if bbox_lonlat is None:
                bounds = tuple(dem_bounds)
            else:
                t = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                lo0, la0, lo1, la1 = bbox_lonlat
                xs, ys = t.transform([lo0, lo1, lo0, lo1], [la0, la0, la1, la1])
                bounds = (min(xs), min(ys), max(xs), max(ys))
        meta = {"crs": crs.to_wkt(), "bounds": [float(b) for b in bounds], "step": float(step), "n_az": N_AZ,
                "dem": str(dem), "dem_coarse": str(Path(dem_coarse).resolve()) if dem_coarse else None,
                "dem_cell": cell, "camera_height": float(camera_height)}
        prov = dem.parent / "dem_provenance.json"
        if prov.exists():
            meta["dem_provenance"] = json.loads(prov.read_text())
        old_path = path / "map.json"
        if old_path.exists():
            old = json.loads(old_path.read_text())
            same = all(old.get(k) == meta[k] for k in ("crs", "bounds", "step", "n_az", "dem", "dem_coarse",
                                                         "camera_height"))
            if same and not force:
                return cls(path)                          # mêmes réglages : horizons et index réutilisables
            for stale in ("points.npy", "elev.npy"):      # réglages changés : l'ancienne carte n'est plus valable
                (path / stale).unlink(missing_ok=True)
            shutil.rmtree(path / "index", ignore_errors=True)
        old_path.write_text(json.dumps(meta, indent=1))
        return cls(path)

    @property
    def is_built(self):
        return (self.path / "elev.npy").exists() and "n_points" in self.meta

    def build(self, devices=None, batch=512, force=False):
        """Calcule l'horizon de chaque point de vue (sur tous les GPU disponibles, un morceau de zone chacun)."""
        if self.is_built and not force:
            print(f"carte déjà construite ({self.meta['n_points']} points de vue) : réutilisée (--force pour refaire)")
            return
        devices = devices or gpu_devices()
        x0, y0, x1, y1 = self.meta["bounds"]
        step = self.meta["step"]
        xs = np.arange(x0 + step / 2, x1, step)
        ys = np.arange(y0 + step / 2, y1, step)
        bands = np.array_split(ys, len(devices))
        results = [None] * len(devices)
        t0 = time.time()

        def work(k, dev):
            yb = bands[k]
            if len(yb) == 0:
                results[k] = (np.zeros((0, 3)), np.zeros((0, N_AZ), np.float16))
                return
            dem = Dem(self.meta["dem"], (x0, yb[0] - step, x1, yb[-1] + step), self.meta["dem_coarse"], dev)
            gx, gy = np.meshgrid(xs, yb)
            gx, gy = gx.ravel(), gy.ravel()
            ground = dem.ground(gx, gy).cpu().numpy()
            keep = np.isfinite(ground)
            gx, gy, ground = gx[keep], gy[keep], ground[keep]
            elev = np.zeros((len(gx), N_AZ), np.float16)
            for i in range(0, len(gx), batch):
                sl = slice(i, i + batch)
                t = lambda a: torch.tensor(a[sl], dtype=torch.float32, device=dev)  # noqa: E731
                e, _ = horizon(dem, t(gx), t(gy), t(ground) + self.camera_height, N_AZ)
                elev[sl] = e.cpu().numpy()
                if k == 0 and (i // batch) % 50 == 0:
                    done = (i + batch) * len(devices)
                    total = len(gx) * len(devices)
                    rate = done / (time.time() - t0)
                    print(f"  horizons : ~{min(done, total)}/{total}, reste ~{(total - done) / rate / 60:.1f} min",
                          flush=True)
            results[k] = (np.stack([gx, gy, ground], 1), elev)
            del dem
            torch.cuda.empty_cache() if dev.startswith("cuda") else None

        threads = [threading.Thread(target=work, args=(k, d)) for k, d in enumerate(devices)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        points = np.concatenate([r[0] for r in results])
        np.save(self.path / "points.npy", points)
        np.save(self.path / "elev.npy", np.concatenate([r[1] for r in results]))
        self.meta["n_points"] = int(len(points))
        (self.path / "map.json").write_text(json.dumps(self.meta, indent=1))
        print(f"carte : {len(points)} points de vue en {(time.time() - t0) / 60:.1f} min -> {self.path}")

    def build_index(self, force=False):
        """Index de mots de crête (CPU), en morceaux de INDEX_SHARD points."""
        if self.has_index and not force:
            print("index déjà construit : réutilisé (--force pour refaire)")
            return
        shutil.rmtree(self.path / "index", ignore_errors=True)
        elev = self.elev
        t0 = time.time()
        for k, i in enumerate(range(0, len(elev), INDEX_SHARD)):
            RidgeIndex.build(elev[i:i + INDEX_SHARD]).save(self.path / "index" / f"shard_{k:03d}")
        print(f"index construit en {(time.time() - t0) / 60:.1f} min -> {self.path / 'index'}")

    # ---------- utilisation ----------

    @property
    def camera_height(self):
        return self.meta.get("camera_height", CAM_HEIGHT)

    @property
    def points(self):
        return np.load(self.path / "points.npy")

    @property
    def elev(self):
        return np.load(self.path / "elev.npy", mmap_mode="r")

    @property
    def has_index(self):
        return (self.path / "index").exists()

    def index(self):
        if self._index is None:
            self._index = ShardedIndex(sorted((self.path / "index").glob("shard_*")))
        return self._index

    def dem(self, device):
        """MNT de toute la zone (pour le raffinement et les rendus)."""
        if self._dem is None:
            self._dem = Dem(self.meta["dem"], self.meta["bounds"], self.meta["dem_coarse"], device)
        return self._dem

    def to_lonlat(self, x, y):
        return pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform(x, y)
