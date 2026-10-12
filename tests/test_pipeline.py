"""Test de bout en bout sur CPU, sans téléchargement : relief synthétique, photo synthétique rendue depuis
un point connu, puis les deux modes doivent retrouver le point et le cap."""
import numpy as np
import pytest
import rasterio
import torch

from crestline.horizon import CAM_HEIGHT, camera_rays, render_depth
from crestline.locate import Locator
from crestline.maps import Map

CELL = 20.0
ORIGIN = (2600000.0, 1200000.0)          # LV95, coin nord-ouest du MNT synthétique
SIZE = 1500                              # 30 km de côté


@pytest.fixture(scope="module")
def synthetic_map(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("carte")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] * CELL
    z = np.full((SIZE, SIZE), 800.0)
    for _ in range(60):                  # massifs gaussiens aléatoires
        cx, cy = rng.uniform(0, SIZE * CELL, 2)
        s = rng.uniform(800, 3000)
        z += rng.uniform(300, 1500) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    dem = tmp / "dem.tif"
    profile = {"driver": "GTiff", "dtype": "float32", "width": SIZE, "height": SIZE, "count": 1, "crs": "EPSG:2056",
               "transform": rasterio.transform.from_origin(*ORIGIN, CELL, CELL)}
    with rasterio.open(dem, "w", **profile) as f:
        f.write(z.astype(np.float32), 1)
    x0, y1 = ORIGIN[0] + 12000, ORIGIN[1] - 12000
    m = Map.create(tmp / "map", dem, bounds=(x0, y1 - 6000, x0 + 6000, y1), step=250.0)
    m.build(devices=["cpu"])
    m.build_index()
    return m


def synthetic_photo(m, x, y, heading, pitch, focal=600.0, w=640, h=480):
    dem = m.dem("cpu")
    ground = float(dem.ground([x], [y])[0])
    depth = render_depth(dem, x, y, ground + CAM_HEIGHT, camera_rays(w, h, focal, heading, pitch))
    terrain = torch.isfinite(depth.view(h, w)).numpy()
    return np.where(terrain.any(0), terrain.argmax(0), h), h, focal


@pytest.mark.parametrize("mode", ["brute", "index"])
def test_finds_position_and_heading(synthetic_map, mode):
    x, y, heading = ORIGIN[0] + 14560.0, ORIGIN[1] - 15140.0, 70.0
    rows, h, focal = synthetic_photo(synthetic_map, x, y, heading, pitch=3.0)
    loc = Locator(synthetic_map.path, devices=["cpu"], load_model=False)
    res = loc.search(rows, h, focal, mode=mode)
    best = res["candidates"][0]
    assert np.hypot(best["x"] - x, best["y"] - y) < 150
    assert abs((best["heading_grid"] - heading + 180) % 360 - 180) < 2
