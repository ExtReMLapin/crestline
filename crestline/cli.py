"""Ligne de commande.

    crestline demo [--cache DOSSIER]                 démo complète (MNT téléchargé, carte construite, photo localisée)
    crestline map build  CARTE (--source ign|swisstopo|copernicus | --dem MNT.tif) --bbox lon0,lat0,lon1,lat1
                     [--bounds x0,y0,x1,y1] [--crs EPSG:2154] [--cell 5]
                     [--step 125] [--camera-height 1.7] [--dem-coarse ...] [--index]
    crestline map index  CARTE                       index de mots de crête (mode index)
    crestline dem fetch --source ign --bbox ... --out DOSSIER   juste télécharger le relief
    crestline locate PHOTO|DOSSIER ... --map CARTE [--mode auto|brute|index] [--fov 60] [--out DOSSIER] [--top 5]

Options communes : --device cuda:0,cuda:1 | cpu (défaut : tous les GPU) ; --model (dépôt HF ou dossier).
Variables d'environnement : CRESTLINE_CACHE, CRESTLINE_MODEL, HF_HOME (cache des poids), XDG_CACHE_HOME.
"""
import argparse
import sys
from pathlib import Path

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def download(url, folder):
    """Image distante téléchargée telle quelle (EXIF conservé, donc la focale)."""
    import re
    import urllib.parse
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name or "image.jpg"
    name = re.sub(r"[^\w.-]", "_", name)[-80:]
    if Path(name).suffix.lower() not in PHOTO_EXT:
        name += ".jpg"
    from .net import get
    path = Path(folder) / name
    path.write_bytes(get(url))
    return path


def photos_from(args, tmp):
    """Fichiers, dossiers et URL http(s)."""
    out = []
    for arg in args:
        if arg.startswith(("http://", "https://")):
            out.append(download(arg, tmp))
            continue
        p = Path(arg)
        out += sorted(q for q in p.iterdir() if q.suffix.lower() in PHOTO_EXT) if p.is_dir() else [p]
    return out


def floats(text):
    return tuple(float(v) for v in text.split(",")) if text else None


def devices(a):
    return a.device.split(",") if a.device else None


def confirm(question, yes):
    if yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit("travail important et pas de terminal pour confirmer : relancez avec --yes")
    if input(f"{question} [O/n] ").strip().lower() not in ("", "o", "oui", "y", "yes"):
        raise SystemExit("abandon")


def zone(a):
    """--bbox, ou --near LIEU [--radius km] (géocodage OpenStreetMap). Renvoie (bbox, code pays ou None)."""
    if getattr(a, "near", None):
        from .net import geocode
        from .sources import bbox_around
        lat, lon, name, country = geocode(a.near)
        print(f"{a.near} : {name} ({lat:.4f}, {lon:.4f}), rayon {a.radius:g} km")
        return bbox_around(lat, lon, a.radius), country
    return floats(a.bbox), None


def prepare_map(map_dir, a, bbox, country, index=True):
    """Télécharge le relief et construit la carte si besoin (réutilise ce qui existe), après estimation."""
    from .maps import Map
    from .sources import SOURCES, already_fetched, choose_source, estimate, fetch
    dem, coarse = getattr(a, "dem", None), getattr(a, "dem_coarse", None)
    source = getattr(a, "source", None) or ("auto" if not dem else None)
    if source:
        if not bbox:
            raise SystemExit("le téléchargement du relief demande --bbox ou --near")
        if source == "auto":
            source = choose_source(bbox, country)
            print(f"source du relief : {source}")
        map_dir = Path(map_dir)
        cell = float(a.cell or SOURCES[source][1])
        fetched = not a.force and already_fetched(map_dir, source, bbox, cell) is not None
        built = not a.force and (map_dir / "elev.npy").exists()
        est = estimate(source, bbox, a.cell, a.step, download=not fetched, build=not built,
                       index=index and (a.force or not (map_dir / "index").exists()))
        if est["minutes"] > 0.05:
            w, h = est["zone_km"]
            print(f"zone {w:.0f} × {h:.0f} km, {est['points'] / 1000:.0f} k points de vue : téléchargement "
                  f"~{est['download_gb']:.1f} Go, disque ~{est['disk_gb']:.1f} Go, mémoire GPU ~{est['gpu_gb']:.1f} Go, "
                  f"~{est['minutes']:.0f} min")
            check_gpu(est["gpu_gb"], a)
            if est["minutes"] > 10 or est["download_gb"] > 2 or est["disk_gb"] > 10:
                confirm("Continuer ?", a.yes)
        dem, coarse, _ = fetch(source, bbox, map_dir, a.cell, force=a.force)
    elif not dem:
        raise SystemExit("il faut --source, --near ou --dem")
    m = Map.create(map_dir, dem, bounds=floats(getattr(a, "bounds", None)), bbox_lonlat=bbox, step=a.step,
                   dem_coarse=coarse, crs=getattr(a, "crs", None), camera_height=a.camera_height, force=a.force)
    m.build(devices(a), force=a.force)
    if index:
        m.build_index(force=a.force)
    return m


def check_gpu(need_gb, a):
    import torch
    if (a.device or "cuda").startswith("cpu") or not torch.cuda.is_available():
        return
    free = torch.cuda.mem_get_info(0)[0] / 1e9
    if need_gb > 0.9 * free:
        raise SystemExit(f"mémoire GPU insuffisante (~{need_gb:.1f} Go nécessaires, {free:.1f} Go libres) : "
                         "réduisez la zone ou augmentez --cell")


def cmd_map_build(a):
    bbox, country = zone(a)
    prepare_map(a.map, a, bbox, country, index=a.index)


def cmd_dem_fetch(a):
    from .sources import choose_source, fetch
    source = choose_source(floats(a.bbox)) if a.source == "auto" else a.source
    fine, coarse, prov = fetch(source, floats(a.bbox), a.out, a.cell, force=a.force)
    print(f"MNT fin : {fine}\nMNT grossier : {coarse}\nprovenance : {prov['fine_shares']}")


def cmd_map_index(a):
    from .maps import Map
    Map(a.map).build_index(force=a.force)


def cmd_locate(a):
    from .locate import Locator
    from .output import default_dir, print_summary, write
    import re
    import tempfile
    if not a.map and not a.near:
        raise SystemExit("il faut --map CARTE ou --near LIEU")
    if a.near:
        from .net import cache_root
        bbox, country = zone(a)
        slug = re.sub(r"[^a-z0-9]+", "-", a.near.lower()).strip("-")
        a.map = cache_root() / "maps" / f"{slug}-{a.radius:g}km"
        prepare_map(a.map, a, bbox, country, index=True)
    loc = Locator(a.map, devices(a), model=a.model)
    tmp = tempfile.TemporaryDirectory()
    photos = photos_from(a.photos, tmp.name)
    for photo in photos:
        try:
            res = loc.locate(photo, focal_px=a.focal_px, fov=a.fov, mode=a.mode, k=a.top, min_conf=a.min_conf)
        except ValueError as e:
            print(e)
            continue
        out = Path(a.out) / photo.stem if (a.out and len(photos) > 1) else Path(a.out) if a.out else default_dir(photo)
        dem = loc.map.dem(loc.device) if res["candidates"] and not a.no_render else None
        write(res, out, dem, loc.map.camera_height)
        print_summary(res, out)


def cmd_demo(a):
    from . import demo
    from .locate import Locator
    from .output import default_dir, print_summary, write
    m = demo.demo_map(cache=a.cache, devices=devices(a))
    photo, info = demo.expected()
    loc = Locator(m.path, devices(a), model=a.model)
    ok = True
    for mode in ("index", "brute"):
        res = loc.locate(photo, mode=mode)                     # focale lue dans l'EXIF de la photo
        out = Path(a.out) / mode if a.out else default_dir(photo).with_name(default_dir(photo).name + f"_{mode}")
        write(res, out, loc.map.dem(loc.device) if res["candidates"] else None, loc.map.camera_height)
        print(f"\n[mode {mode}]")
        print_summary(res, out)
        if res["candidates"]:
            c = res["candidates"][0]
            d = demo.distance_m(c["lat"], c["lon"], info["expected"]["lat"], info["expected"]["lon"])
            good = d <= info["expected"]["tolerance_m"]
            ok &= good
            print(f"  attendu : lac d'Anterne — écart {d:.0f} m : {'OK' if good else 'RATÉ'}")
        else:
            ok = False
    sys.exit(0 if ok else 1)


def common(parser):
    parser.add_argument("--device", help="cuda:0,cuda:1 ou cpu (défaut : tous les GPU)")
    parser.add_argument("--model", help="Crestline : dépôt Hugging Face ou dossier local (défaut : $CRESTLINE_MODEL ou public)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="crestline", description="Retrouver d'où une photo de montagne a été prise.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="démo complète sur le lac d'Anterne (télécharge ~quelques centaines de Mo)")
    d.add_argument("--out", help="dossier de sortie")
    d.add_argument("--cache", help="dossier du MNT et de la carte de démo (défaut : $CRESTLINE_CACHE ou ~/.cache/crestline)")
    common(d)
    d.set_defaults(func=cmd_demo)

    mp = sub.add_parser("map", help="construire une carte").add_subparsers(dest="map_cmd", required=True)
    b = mp.add_parser("build", help="carte = MNT + horizons (mode brute)")
    b.add_argument("map", help="dossier de la carte")
    b.add_argument("--source", choices=("auto", "ign", "swisstopo", "copernicus"),
                   help="télécharger le relief : auto (choisit selon le lieu), ign (France, 5 m), "
                        "swisstopo (Suisse et voisins, 10 m), copernicus (monde, 30 m, imprécis en montagne)")
    b.add_argument("--near", help="ou : zone autour d'un lieu (ex. \"Zermatt\"), au lieu de --bbox")
    b.add_argument("--radius", type=float, default=20.0, help="rayon autour de --near (km)")
    b.add_argument("--yes", "-y", action="store_true", help="ne pas demander de confirmation")
    b.add_argument("--cell", type=float, help="maille du MNT fin téléchargé (m ; défaut : 5 ign, 10 swisstopo, 30 copernicus)")
    b.add_argument("--dem", help="ou : MNT GeoTIFF local (mètres ; en degrés, il est reprojeté en UTM)")
    b.add_argument("--dem-coarse", help="MNT grossier optionnel couvrant zone + 120 km (horizon lointain)")
    b.add_argument("--bbox", help="zone des points de vue : lon_min,lat_min,lon_max,lat_max (défaut : tout le MNT) ; longitude négative : --bbox=-0.3,42.6,...")
    b.add_argument("--bounds", help="zone dans le CRS de la carte : xmin,ymin,xmax,ymax")
    b.add_argument("--crs", help="CRS de la carte, ex. EPSG:2154 (défaut : celui du MNT ; UTM si MNT en degrés)")
    b.add_argument("--step", type=float, default=125.0, help="pas de la grille de points de vue (m)")
    b.add_argument("--camera-height", type=float, default=1.7, help="hauteur de la caméra au-dessus du sol (m)")
    b.add_argument("--device", help="cuda:0,cuda:1 ou cpu (défaut : tous les GPU)")
    b.add_argument("--index", action="store_true", help="construire aussi l'index (mode index)")
    b.add_argument("--force", action="store_true", help="tout refaire (sinon relief, horizons et index existants sont réutilisés)")
    b.set_defaults(func=cmd_map_build)
    dm = sub.add_parser("dem", help="télécharger un MNT").add_subparsers(dest="dem_cmd", required=True)
    f = dm.add_parser("fetch", help="MNT fin (zone + 30 km) et grossier (zone + 120 km), sources publiques")
    f.add_argument("--source", choices=("auto", "ign", "swisstopo", "copernicus"), default="auto")
    f.add_argument("--bbox", required=True, help="lon_min,lat_min,lon_max,lat_max")
    f.add_argument("--cell", type=float)
    f.add_argument("--out", required=True, help="dossier de sortie")
    f.add_argument("--force", action="store_true", help="retélécharger même si déjà présent")
    f.set_defaults(func=cmd_dem_fetch)

    i = mp.add_parser("index", help="index de mots de crête (mode index, recherche rapide)")
    i.add_argument("map")
    i.add_argument("--force", action="store_true", help="reconstruire même si déjà présent")
    i.set_defaults(func=cmd_map_index)

    lo = sub.add_parser("locate", help="localiser des photos : fichiers, dossiers ou URL")
    lo.add_argument("photos", nargs="+", help="fichier, dossier ou URL http(s) d'une image")
    lo.add_argument("--map", help="dossier de la carte")
    lo.add_argument("--near", help="ou : lieu approximatif (ex. \"Chamonix\") ; la carte est construite au besoin")
    lo.add_argument("--radius", type=float, default=20.0, help="rayon de recherche autour de --near (km)")
    lo.add_argument("--source", choices=("auto", "ign", "swisstopo", "copernicus"), default="auto",
                    help="relief pour --near (auto : IGN en France, swisstopo en Suisse, Copernicus ailleurs)")
    lo.add_argument("--cell", type=float, help="maille du relief téléchargé pour --near (m)")
    lo.add_argument("--step", type=float, default=125.0, help="pas de grille pour --near (m)")
    lo.add_argument("--camera-height", type=float, default=1.7)
    lo.add_argument("--force", action="store_true", help="avec --near : tout refaire")
    lo.add_argument("--yes", "-y", action="store_true", help="ne pas demander de confirmation")
    lo.add_argument("--mode", choices=("auto", "brute", "index"), default="auto",
                    help="auto : index s'il existe, sinon brute")
    lo.add_argument("--fov", type=float, help="champ horizontal (degrés), si la focale n'est pas dans l'EXIF")
    lo.add_argument("--focal-px", type=float, help="focale en pixels (largeur d'origine de la photo)")
    lo.add_argument("--out", help="dossier de sortie (défaut : crestline_out/<horodatage>_<photo>)")
    lo.add_argument("--no-render", action="store_true", help="ne pas faire le rendu MNT des candidats")
    lo.add_argument("--top", type=int, default=5, help="nombre de candidats rendus")
    lo.add_argument("--min-conf", type=float, default=0.6,
                    help="confiance Crestline minimale d'une colonne de skyline (0 : tout garder)")
    common(lo)
    lo.set_defaults(func=cmd_locate)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
