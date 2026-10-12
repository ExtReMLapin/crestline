"""Dossier de sortie d'une localisation.

    <sortie>/
    ├── result.json        top 5 : lat/lon, cap, tangage, roulis, score, confiance, liens OSM / Google Maps
    ├── result.geojson     points + direction de visée (geojson.io, QGIS...)
    ├── summary.md         résumé lisible
    ├── skyline.jpg        skyline détectée : vert = utilisée, rouge = ignorée (peu fiable)
    ├── candidates.jpg     planche des 5 candidats
    └── candidates/1.jpg   rendu du MNT depuis le candidat, superposé à la photo :
                           cyan = skyline du MNT, jaune = crêtes du MNT, vert = skyline de la photo
"""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .horizon import camera_rays, render_depth

RENDER_WIDTH = 768


def default_dir(photo, root="crestline_out"):
    return Path(root) / f"{datetime.now():%Y%m%d-%H%M%S}_{Path(photo).stem}"


def draw_line(draw, rows, color, width, keep=None):
    pts = [(u, r) for u, r in enumerate(rows) if np.isfinite(r) and (keep is None or keep[u])]
    run = []
    for p in pts:                                  # segments continus seulement
        if run and p[0] != run[-1][0] + 1:
            if len(run) > 1:
                draw.line(run, fill=color, width=width)
            run = []
        run.append(p)
    if len(run) > 1:
        draw.line(run, fill=color, width=width)


def skyline_image(res):
    img = res["image"].copy()
    d = ImageDraw.Draw(img)
    rows = res["rows"].astype(float)
    rows[(res["rows"] <= 0) | (res["rows"] >= img.height)] = np.nan
    w = max(2, img.width // 400)
    draw_line(d, rows, (255, 40, 40), w)
    draw_line(d, rows, (40, 255, 40), w + 1, keep=res["reliable"])
    return img


def render_candidate(dem, res, c, cam_h):
    """Rendu du MNT vu depuis le candidat (même focale, cap, tangage, roulis), superposé à la photo."""
    img = res["image"]
    scale = RENDER_WIDTH / img.width
    W, H = RENDER_WIDTH, int(round(img.height * scale))
    rays = camera_rays(W, H, res["focal_px"] * scale, c["heading_grid"], c["pitch"], c["roll"])
    ground = float(dem.ground([c["x"]], [c["y"]])[0])
    depth = render_depth(dem, c["x"], c["y"], ground + cam_h, rays).view(H, W).cpu().numpy()
    terrain = np.isfinite(depth)
    sky_rows = np.where(terrain.any(0), terrain.argmax(0), np.nan)
    # crête : relief proche devant un relief nettement plus lointain (pixel du dessus)
    above = np.vstack([np.full((1, W), np.nan), depth[:-1]])
    ridge = terrain & np.isfinite(above) & (above > 1.25 * depth) & (above - depth > 50)
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, img.width // 400)
    for r, u in zip(*np.nonzero(ridge)):
        d.ellipse([u / scale - lw, r / scale - lw, u / scale + lw, r / scale + lw], fill=(255, 220, 0))
    draw_line(d, np.interp(np.arange(img.width) * scale, np.arange(W), sky_rows) / scale, (0, 220, 255), lw + 1)
    rows = res["rows"].astype(float)
    rows[(res["rows"] <= 0) | (res["rows"] >= img.height)] = np.nan
    draw_line(d, rows, (40, 255, 40), max(1, lw - 1), keep=res["reliable"])
    return out


def write(res, out_dir, dem=None, cam_h=1.7):
    out_dir = Path(out_dir)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    skyline_image(res).save(out_dir / "skyline.jpg", quality=90)
    cands = res["candidates"]
    keys = ("photo", "mode", "focal_px", "focal_source", "hfov_deg", "confidence", "confidence_ratio", "error",
            "exif_gps", "gps_error_m")
    summary = {k: res.get(k) for k in keys if k in res}
    summary["candidates"] = [{k: v for k, v in c.items() if k not in ("pitch_hyp",)} for c in cands]
    (out_dir / "result.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    features = []
    for c in cands:
        props = {"rank": c["rank"], "score_deg": round(c["score_deg"], 4), "heading": round(c["heading"], 1)}
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]}})
        # direction de visée sur 2 km (approximation locale suffisante pour l'affichage)
        h = math.radians(c["heading"])
        dlat = 2000 * math.cos(h) / 111_320
        dlon = 2000 * math.sin(h) / (111_320 * math.cos(math.radians(c["lat"])))
        features.append({"type": "Feature", "properties": {**props, "kind": "view"},
                         "geometry": {"type": "LineString",
                                      "coordinates": [[c["lon"], c["lat"]], [c["lon"] + dlon, c["lat"] + dlat]]}})
    (out_dir / "result.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=1))
    lines = [f"# {Path(res['photo']).name}", ""]
    if "error" in res:
        lines += [f"**Échec : {res['error']}**", ""]
    else:
        lines += [f"Mode : {res['mode']} — focale {res['focal_px']:.0f} px ({res['focal_source']}), "
                  f"champ horizontal {res['hfov_deg']:.1f}° — confiance : **{res['confidence']}**", "",
                  "| # | lat, lon | cap | tangage | score | carte |", "|---|---|---|---|---|---|"]
        for c in cands:
            lines.append(f"| {c['rank']} | {c['lat']:.5f}, {c['lon']:.5f} | {c['heading']:.0f}° | {c['pitch']:.1f}° | "
                         f"{c['score_deg']:.3f}° | [OSM]({c['links']['openstreetmap']}) · "
                         f"[Google]({c['links']['google_maps']}) |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    if dem is not None and cands:
        tiles = []
        for c in cands:
            im = render_candidate(dem, res, c, cam_h)
            im.save(out_dir / "candidates" / f"{c['rank']}.jpg", quality=88)
            t = im.copy()
            t.thumbnail((640, 640))
            ImageDraw.Draw(t).text((8, 8), f"#{c['rank']}  {c['lat']:.4f}, {c['lon']:.4f}  score {c['score_deg']:.3f}°",
                                   fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            tiles.append(t)
        sheet = Image.new("RGB", (sum(t.width for t in tiles[:3]), tiles[0].height * ((len(tiles) + 2) // 3)), "black")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 3) * tiles[0].width, (i // 3) * tiles[0].height))
        sheet.save(out_dir / "candidates.jpg", quality=88)
    return out_dir


def print_summary(res, out_dir):
    name = Path(res["photo"]).name
    if "error" in res:
        print(f"{name} : échec — {res['error']}  ({out_dir})")
        return
    c = res["candidates"][0]
    print(f"{name} : {c['lat']:.5f}, {c['lon']:.5f}  cap {c['heading']:.0f}°  score {c['score_deg']:.3f}°  "
          f"confiance {res['confidence']}")
    print(f"  OSM    : {c['links']['openstreetmap']}")
    print(f"  Google : {c['links']['google_maps']}")
    if "gps_error_m" in res:
        print(f"  écart avec le GPS de l'EXIF : {res['gps_error_m']:.0f} m")
    print(f"  détails : {out_dir}")
