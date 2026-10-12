"""Accès réseau : User-Agent identifiable, géocodage OpenStreetMap (Nominatim) avec cache local.

Chaque installation a un identifiant aléatoire (pas un identifiant matériel : rien sur la machine n'est
révélé), ajouté au User-Agent. Les services voient ainsi des utilisateurs distincts, et non une source
unique qu'ils bloqueraient en entier. Nominatim : 1 requête par seconde au plus, résultats gardés en cache.
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import __version__

REPO = "https://github.com/ExtReMLapin/crestline"
NOMINATIM = "https://nominatim.openstreetmap.org"
_lock = threading.Lock()
_last_nominatim = [0.0]


def config_dir():
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "crestline"


def cache_root():
    """$CRESTLINE_CACHE, sinon $XDG_CACHE_HOME/crestline, sinon ~/.cache/crestline."""
    if os.environ.get("CRESTLINE_CACHE"):
        return Path(os.environ["CRESTLINE_CACHE"])
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "crestline"


def install_id():
    path = config_dir() / "install_id"
    try:
        return path.read_text().strip()
    except OSError:
        iid = uuid.uuid4().hex[:12]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(iid)
        except OSError:
            pass
        return iid


def user_agent():
    return f"crestline/{__version__} (+{REPO}; install {install_id()})"


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _nominatim(path, params):
    """Requête Nominatim avec cache disque et au plus 1 requête par seconde."""
    params = {**params, "format": "jsonv2"}
    key = path + "?" + urllib.parse.urlencode(sorted(params.items()))
    cache_file = cache_root() / "geocode.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return cache[key]
    with _lock:
        wait = 1.0 - (time.time() - _last_nominatim[0])
        if wait > 0:
            time.sleep(wait)
        data = json.loads(get(f"{NOMINATIM}/{key}", timeout=30))
        _last_nominatim[0] = time.time()
    cache[key] = data
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
    except OSError:
        pass
    return data


def geocode(place):
    """Nom de lieu -> (lat, lon, nom complet, code pays)."""
    res = _nominatim("search", {"q": place, "limit": 1, "addressdetails": 1})
    if not res:
        raise ValueError(f"lieu introuvable : {place!r}")
    r = res[0]
    return float(r["lat"]), float(r["lon"]), r.get("display_name", place), r.get("address", {}).get("country_code")


def country_at(lat, lon):
    """Code pays (ISO, minuscules) au point, ou None (mer, erreur réseau)."""
    try:
        r = _nominatim("reverse", {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "zoom": 3, "addressdetails": 1})
        return r.get("address", {}).get("country_code")
    except Exception:
        return None
