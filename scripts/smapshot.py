"""Client smapshot (photos suisses recalées à la main) et modèle de caméra.

API : https://smapshot.heig-vd.ch/api/v1/  (images publiques, licences par image, souvent CC BY-SA 4.0)
La pose smapshot est en WGS84 (latitude, longitude, altitude ellipsoïdale), avec azimut / tilt / roll
en degrés et une focale en pixels. Les points de calage (GCP) sont fournis en WGS84 et en ECEF.
"""
import json
import time
import urllib.request

import numpy as np

API = "https://smapshot.heig-vd.ch/api/v1"
A, E2 = 6378137.0, 6.69437999014e-3  # ellipsoïde WGS84


def get(path, retries=3):
    for k in range(retries):
        try:
            with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
                return json.load(r)
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 ** k)


def geodetic_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - E2) + h) * np.sin(lat)], axis=-1)


def enu_basis(lat, lon):
    """Lignes : vecteurs Est, Nord, Haut (en ECEF) au point (lat, lon)."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([[-np.sin(lo), np.cos(lo), 0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


class Camera:
    """Caméra sténopé. azimut : depuis le nord, sens horaire ; tilt : positif vers le haut ;
    roll : rotation autour de l'axe de visée ; point principal au centre de l'image."""

    def __init__(self, lat, lon, alt, azimuth, tilt, roll, focal, width, height, roll_sign=-1.0):
        self.lat, self.lon, self.alt = lat, lon, alt
        self.focal, self.width, self.height = focal, width, height
        self.center = geodetic_to_ecef(lat, lon, alt)
        self.enu = enu_basis(lat, lon)
        az, ti, ro = np.radians([azimuth, tilt, roll_sign * roll])
        fwd = np.array([np.sin(az) * np.cos(ti), np.cos(az) * np.cos(ti), np.sin(ti)])
        right0 = np.array([np.cos(az), -np.sin(az), 0.0])
        up0 = np.cross(right0, fwd)
        right = np.cos(ro) * right0 + np.sin(ro) * up0
        up = np.cross(right, fwd)
        self.R = np.stack([right, up, fwd])          # lignes : axes caméra exprimés en ENU

    @classmethod
    def from_attributes(cls, attrs, **kw):
        p = attrs["pose"]
        return cls(p["latitude"], p["longitude"], p["altitude"], p["azimuth"], p["tilt"], p["roll"],
                   p["focal"], attrs["width"], attrs["height"], **kw)

    def ecef_to_camera(self, xyz):
        enu = (np.asarray(xyz) - self.center) @ self.enu.T
        return enu @ self.R.T                        # (x droite, y haut, z avant)

    def project(self, xyz):
        c = self.ecef_to_camera(xyz)
        u = self.width / 2 + self.focal * c[..., 0] / c[..., 2]
        v = self.height / 2 - self.focal * c[..., 1] / c[..., 2]
        return np.stack([u, v], axis=-1), c[..., 2]

    def ray_directions_enu(self, us, vs):
        """Directions (ENU, normées) des rayons passant par les pixels (us, vs)."""
        x = (us - self.width / 2) / self.focal
        y = -(vs - self.height / 2) / self.focal
        d = x[..., None] * self.R[0] + y[..., None] * self.R[1] + self.R[2]
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def gcps(geolocalisation_id):
    g = get(f"/geolocalisations/{geolocalisation_id}")
    pts = g["gcp_json"] or []
    return {"pixel": np.array([[p["x"], p["y"]] for p in pts]),
            "reproj": np.array([[p.get("xReproj", np.nan), p.get("yReproj", np.nan)] for p in pts]),
            "ecef": np.array([[p["X"], p["Y"], p["Z"]] for p in pts]),
            "geodetic": np.array([[p["latitude"], p["longitude"], p["altitude"]] for p in pts])}


def download_photo(image_url, dst, width=1024):
    """Télécharge une photo smapshot. IIIF (ETH) : redimensionnée côté serveur à `width`.
    Hébergée par smapshot (vignette) : assemblée depuis les tuiles Deep Zoom au niveau dont la
    largeur est la plus proche de `width` sans la dépasser de plus de 2x (24 tuiles environ)."""
    import io
    import math
    import re
    from PIL import Image
    if "/full/200,/" in image_url:
        with urllib.request.urlopen(image_url.replace("/full/200,/", f"/full/{width},/"), timeout=120) as r:
            Image.open(io.BytesIO(r.read())).convert("RGB").save(dst, quality=92)
        return
    base = image_url.replace("/images/thumbnails/", "/images/tiles/").rsplit(".", 1)[0]
    with urllib.request.urlopen(base + ".dzi", timeout=60) as r:
        xml = r.read().decode()
    ts = int(re.search(r'TileSize="(\d+)"', xml).group(1))
    ov = int(re.search(r'Overlap="(\d+)"', xml).group(1))
    fmt = re.search(r'Format="(\w+)"', xml).group(1)
    W, H = int(re.search(r'Width="(\d+)"', xml).group(1)), int(re.search(r'Height="(\d+)"', xml).group(1))
    max_level = math.ceil(math.log2(max(W, H)))
    level = max_level
    while level > 0 and math.ceil(W / 2 ** (max_level - level + 1)) >= width * 0.7:
        level -= 1
    w, h = math.ceil(W / 2 ** (max_level - level)), math.ceil(H / 2 ** (max_level - level))
    out = Image.new("RGB", (w, h))
    for c in range(math.ceil(w / ts)):
        for r_ in range(math.ceil(h / ts)):
            with urllib.request.urlopen(f"{base}_files/{level}/{c}_{r_}.{fmt}", timeout=60) as r:
                tile = Image.open(io.BytesIO(r.read())).convert("RGB")
            ox, oy = (ov if c > 0 else 0), (ov if r_ > 0 else 0)
            out.paste(tile.crop((ox, oy, ox + min(ts, w - c * ts), oy + min(ts, h - r_ * ts))), (c * ts, r_ * ts))
    out.save(dst, quality=92)
