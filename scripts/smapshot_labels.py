"""Étiquettes de crêtes pour des photos smapshot : rendu MNT, affinage d'orientation, masques.

Pour chaque photo :
1. rendu de la carte de distance (MNT suisse) sur un cadre élargi (marge) ;
2. affinage cap / tangage / roulis / focale : la position est fiable, donc une rotation de la caméra
   = homographie exacte de l'image. On cherche celle qui pose le plus de lignes MNT (skyline,
   crêtes) sur des contours forts de la photo (grille grossière puis fine, sur GPU) ;
3. étiquettes à la résolution de la photo : ciel, skyline, crêtes (vrais sauts de profondeur) ;
4. zones ignorées : occultants (Mask2Former), nuages (MNT relief / photo ciel), relief absent du
   MNT (MNT ciel / photo relief), lignes sans contour visible sur la photo.

Usage : python scripts/smapshot_labels.py <liste_ids.txt> <sortie> [largeur]
"""
import itertools
import json
import os
import threading
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

sys.path.insert(0, str(Path(__file__).parent))
import dem_render as D  # noqa: E402
import smapshot as S  # noqa: E402
from prepare_geopose import OCCLUDERS  # noqa: E402

MARGIN = 0.08                 # marge de rendu de chaque côté (fraction de la largeur)
SEG_REPO = "facebook/mask2former-swin-large-ade-semantic"
SKY_ID = 2
RIDGE_LOG_JUMP = np.log(1.25)  # le relief masqué est au moins 25 % plus loin...
RIDGE_MIN_JUMP_M = 50.0        # ...et au moins 50 m plus loin...
RIDGE_SHARPNESS = 4.0          # ...et le saut est >= 4x la variation entre pixels voisins côté proche
EDGE_SUPPORT_PX = 2            # une ligne est « vue » si un contour fort est à moins de 2 px
EDGE_SUPPORT_MIN = 0.25        # seuil de contour (0..1, normalisé par photo)
LIMITS = (0.6, 0.6, 0.6, 0.015)  # correction max : lacet, tangage, roulis (°), log-échelle
PRIOR = 0.02                   # pénalité de score par 0,6° (ou 1,5 %) de correction
MIN_GAIN = 0.03                # gain minimal de score de contour pour accepter la correction


# ---------- contours de la photo ----------

def edge_strength(rgb):
    """Force de contour (0..1) : gradient du Lab lissé, normalisé par le 98e percentile."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 1.2)
    g = sum(np.hypot(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0), cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1))
            for c in range(3))
    return np.clip(g / (np.percentile(g, 98) + 1e-6), 0, 1)


# ---------- lignes depuis la carte de distance ----------

def depth_lines(dist):
    """Skyline et crêtes d'occultation. Une crête = pixel proche dont un voisin est nettement plus
    loin, avec un saut franc (bien plus grand que la variation côté proche) : on écarte ainsi les
    pentes rasantes où la distance croît vite mais continûment."""
    sky = dist <= 0
    skyline = ~sky & ndimage.binary_dilation(sky, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    d = np.where(sky, np.nan, dist)
    logd = np.log(np.maximum(d, 1.0))
    ridge = np.zeros(d.shape, bool)
    p = np.pad(logd, 2, constant_values=np.nan)
    pd_ = np.pad(d, 2, constant_values=np.nan)
    h, w = d.shape
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        far = p[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]                 # voisin (côté lointain)
        near_prev = p[2 - dy:2 - dy + h, 2 - dx:2 - dx + w]           # pixel suivant côté proche
        far_d = pd_[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]
        with np.errstate(invalid="ignore"):
            jump = far - logd
            prev = np.abs(logd - near_prev)
            ridge |= (jump > RIDGE_LOG_JUMP) & (far_d - d > RIDGE_MIN_JUMP_M) & \
                     (jump > RIDGE_SHARPNESS * np.nan_to_num(prev, nan=0.0))
    return sky, skyline, ridge & ~sky & ~skyline


# ---------- affinage d'orientation (homographie de rotation) ----------

def rotation(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def homography(params, f, canvas_c, photo_c):
    """Pixels du cadre de rendu -> pixels de la photo, pour (lacet, tangage, roulis en °, log-échelle)."""
    yaw, pitch, roll, log_s = params
    K0 = np.array([[f, 0, canvas_c[0]], [0, f, canvas_c[1]], [0, 0, 1]])
    K1 = np.array([[f * np.exp(log_s), 0, photo_c[0]], [0, f * np.exp(log_s), photo_c[1]], [0, 0, 1]])
    return K1 @ rotation(yaw, pitch, roll) @ np.linalg.inv(K0)


@torch.no_grad()
def score_homographies(Hs, pts, weights, edge, device):
    """Score moyen de contour sous les points de ligne transformés, pour un lot d'homographies."""
    Ht = torch.as_tensor(np.stack(Hs), dtype=torch.float32, device=device)            # (C, 3, 3)
    P = torch.as_tensor(np.c_[pts, np.ones(len(pts))], dtype=torch.float32, device=device)  # (N, 3)
    q = torch.einsum("cij,nj->cni", Ht, P)
    uv = q[..., :2] / q[..., 2:3]
    h, w = edge.shape[-2:]
    grid = torch.stack([(2 * uv[..., 0] + 1) / w - 1, (2 * uv[..., 1] + 1) / h - 1], -1)[:, None]
    val = F.grid_sample(edge.expand(len(Hs), -1, -1, -1), grid, align_corners=False)[:, 0, 0]  # (C, N)
    inside = (grid.abs() <= 1).all(-1)[:, 0]
    wt = torch.as_tensor(weights, dtype=torch.float32, device=device)[None] * inside
    frac = wt.sum(1) / wt.new_tensor(weights).sum()
    s = (val * wt).sum(1) / wt.sum(1).clamp(min=1e-6)
    return torch.where(frac < 0.5, torch.zeros_like(s), s)


def refine_orientation(dist_canvas, edge, f, canvas_c, photo_c, device):
    _, skyline, ridge = depth_lines(dist_canvas)
    ys, xs = np.nonzero(skyline | ridge)
    if len(xs) < 50:
        return np.zeros(4), 0.0, 0.0
    weights = np.where(skyline[ys, xs], 1.0, 0.3)
    sel = np.random.default_rng(0).choice(len(xs), min(len(xs), 6000), replace=False)
    pts, weights = np.c_[xs[sel], ys[sel]].astype(float), weights[sel]
    best = np.zeros(4)
    edge_t = torch.as_tensor(edge, device=device)[None, None]
    base = float(score_homographies([homography(best, f, canvas_c, photo_c)], pts, weights, edge_t, device)[0])
    # grossier (contours élargis pour un bassin d'attraction plus large), puis fin
    # Les poses smapshot sont cliquées à la main (résidus ~0,1-0,3°) : la correction est bornée à
    # ±0,6° / ±1,5 % et pénalisée, pour ne pas s'accrocher à des textures (neige, roche).
    for blur, span, n in ((3.0, (0.6, 0.6, 0.6, 0.015), 7), (1.0, (0.15, 0.15, 0.2, 0.004), 5), (0.0, (0.04, 0.04, 0.05, 0.001), 5)):
        e = torch.as_tensor(cv2.GaussianBlur(edge, (0, 0), blur) if blur else edge, device=device)[None, None]
        grids = [np.clip(best[i] + np.linspace(-span[i], span[i], n), -LIMITS[i], LIMITS[i]) for i in range(4)]
        cands = [np.array(c) for c in itertools.product(*grids)]
        scores = torch.cat([score_homographies([homography(c, f, canvas_c, photo_c) for c in cands[i:i + 2048]],
                                               pts, weights, e, device) for i in range(0, len(cands), 2048)])
        penalty = torch.as_tensor([PRIOR * (np.abs(c[:3]).sum() / 0.6 + abs(c[3]) / 0.015) for c in cands],
                                  dtype=scores.dtype, device=device)
        best = cands[int((scores - penalty).argmax())]
    final = float(score_homographies([homography(best, f, canvas_c, photo_c)], pts, weights, edge_t, device)[0])
    if final < base + MIN_GAIN:            # gain trop faible : on garde la pose cliquée
        return np.zeros(4), base, base
    return best, base, final


def skyline_agreement(sky_dem, sky_photo, occluder):
    """Écart (px) par colonne entre la skyline MNT et celle de la photo, là où les deux existent et
    où le bord n'est pas occulté. Sert à écarter les photos mal posées."""
    def rows(m):
        return np.argmin(np.vstack([m, np.zeros((1, m.shape[1]), bool)]), axis=0)
    r_dem, r_photo = rows(sky_dem), rows(sky_photo)
    h = sky_dem.shape[0]
    cols = np.arange(sky_dem.shape[1])
    ok = (r_dem > 0) & (r_dem < h) & (r_photo > 0) & (r_photo < h)
    ok &= ~occluder[np.clip(r_photo, 0, h - 1), cols]
    e = np.abs(r_dem - r_photo)[ok]
    return {"skyline_cols": float(ok.mean()),
            "skyline_err_med": float(np.median(e)) if e.size > 20 else None,
            "skyline_err_p90": float(np.percentile(e, 90)) if e.size > 20 else None}


# ---------- pipeline par photo ----------

class Labeler:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(SEG_REPO)
        self.seg_model = AutoModelForUniversalSegmentation.from_pretrained(SEG_REPO).to(device).eval()
        id2label = self.seg_model.config.id2label
        self.occ_ids = [k for k, v in id2label.items() if v in OCCLUDERS]
        self.scene, self.scene_center = None, None

    @torch.no_grad()
    def segment(self, rgb):
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        return self.processor.post_process_semantic_segmentation(
            self.seg_model(**inputs), target_sizes=[rgb.shape[:2]])[0].cpu().numpy()

    def prepare(self, image_id, out_dir, width, load_dem):
        """Partie CPU / réseau (exécutée en avance dans un thread) : pose, photo, contours, MNT."""
        attrs = S.get(f"/images/{image_id}/attributes")
        cam = S.Camera.from_attributes(attrs)
        photo_path = out_dir / "photos" / f"{image_id}.jpg"
        if not photo_path.exists():
            tmp = photo_path.with_suffix(f".{threading.get_ident()}.part.jpg")
            S.download_photo(attrs["media"]["image_url"], tmp, width)
            os.replace(tmp, photo_path)
        rgb = np.asarray(Image.open(photo_path).convert("RGB"))
        e0, n0 = D.TO_LV95.transform(cam.lon, cam.lat)
        arrays = D.load_arrays(e0, n0) if load_dem else None
        return {"id": image_id, "attrs": attrs, "cam": cam, "rgb": rgb, "edge": edge_strength(rgb),
                "e0": e0, "n0": n0, "arrays": arrays}

    def process(self, prep, out_dir):
        """Partie GPU : rendu, affinage, masques, écriture des étiquettes."""
        image_id, attrs, cam, rgb, edge = prep["id"], prep["attrs"], prep["cam"], prep["rgb"], prep["edge"]
        if prep["arrays"] is not None:
            self.scene = D.Scene(prep["e0"], prep["n0"], self.device, arrays=prep["arrays"])
        elif self.scene is None or np.hypot(prep["e0"] - self.scene.e0, prep["n0"] - self.scene.n0) > 500:
            self.scene = D.Scene(prep["e0"], prep["n0"], self.device)   # garde-fou : MNT d'une autre zone
        h, w = rgb.shape[:2]
        scale = cam.width / w                          # pixels pleine résolution par pixel de travail
        f = cam.focal / scale
        # rendu sur un cadre élargi : même centre optique, marge de MARGIN de chaque côté
        mw, mh = int(round(w * (1 + 2 * MARGIN))), int(round(h + 2 * MARGIN * w))
        big = S.Camera(cam.lat, cam.lon, cam.alt, 0, 0, 0, cam.focal, mw * scale, mh * scale)
        big.R, big.enu, big.center = cam.R, cam.enu, cam.center
        dist_canvas, _ = D.render_camera(big, mw, mh, device=self.device, scene=self.scene)
        canvas_c, photo_c = (mw / 2, mh / 2), (w / 2, h / 2)
        best, s0, s1 = refine_orientation(dist_canvas, edge, f, canvas_c, photo_c, self.device)
        H = homography(best, f, canvas_c, photo_c)
        dist = cv2.warpPerspective(dist_canvas, H, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        sky, skyline, ridge = depth_lines(np.nan_to_num(dist, nan=1e9))
        dist_before = cv2.warpPerspective(dist_canvas, homography(np.zeros(4), f, canvas_c, photo_c), (w, h),
                                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        # masques à ignorer
        seg = self.segment(rgb)
        photo_sky = seg == SKY_ID
        occluder = np.isin(seg, self.occ_ids)
        # désaccords ciel MNT / ciel photo : seulement les zones épaisses (>= ~7 px), pas la bande de
        # 1-2 px le long de la skyline due à l'imprécision normale de la ligne
        cloud = ndimage.binary_opening(~sky & photo_sky, iterations=3)               # relief MNT, ciel / nuage sur la photo
        missing = ndimage.binary_opening(sky & ~photo_sky & ~occluder, iterations=3)  # relief photo absent du MNT
        support = ndimage.maximum_filter(edge, size=2 * EDGE_SUPPORT_PX + 1) >= EDGE_SUPPORT_MIN
        # crête interne MNT sans contour visible sur la photo (la skyline, elle, est contrôlée par
        # l'accord avec le ciel de la photo : masques cloud / missing ci-dessus)
        unseen = ridge & ~support
        ignore = ndimage.binary_dilation(occluder, iterations=3) | ndimage.binary_dilation(cloud | missing, iterations=2) \
            | ndimage.binary_dilation(unseen, iterations=4) | np.isnan(dist)
        meta = {"id": image_id, "title": attrs["title"], "date": attrs.get("date_shot_min"),
                "license": attrs.get("license"), "collection": attrs.get("collection", {}).get("id")
                if isinstance(attrs.get("collection"), dict) else attrs.get("collection"),
                "yaw": best[0], "pitch": best[1], "roll": best[2], "scale": float(np.exp(best[3])),
                "edge_score_before": s0, "edge_score_after": s1,
                "cloud_frac": float(cloud.mean()), "missing_frac": float(missing.mean()),
                "sky_frac": float(sky.mean()), **skyline_agreement(sky, photo_sky, occluder), "skyline_px": int(skyline.sum()), "ridge_px": int(ridge.sum()),
                "skyline_kept": float((skyline & ~ignore).sum() / max(skyline.sum(), 1)),
                "ridge_kept": float((ridge & ~ignore).sum() / max(ridge.sum(), 1)), "ignore_frac": float(ignore.mean())}
        np.savez_compressed(out_dir / "labels" / f"{image_id}.npz", image=rgb, sky=sky, skyline=skyline,
                            ridge=ridge, ignore=ignore, distance=np.nan_to_num(dist, nan=-2).astype(np.float32),
                            distance_before=np.clip(np.nan_to_num(dist_before, nan=-2), -2, 60000).astype(np.float16), seg=seg.astype(np.uint8))
        return meta


def scene_groups(ids, catalog, radius=400.0):
    """Regroupe les photos consécutives prises à moins de `radius` m d'un même centre : elles
    partagent la même pyramide de MNT. Renvoie, pour chaque photo, s'il faut charger un MNT."""
    pos = catalog.set_index("id")[["longitude", "latitude"]]
    load, center = [], None
    for i in ids:
        lon, lat = pos.loc[int(i)]
        e, n = D.TO_LV95.transform(lon, lat)
        new = center is None or np.hypot(e - center[0], n - center[1]) > radius
        if new:
            center = (e, n)
        load.append(new)
    return load


def main(id_list, out_dir, width=1024, catalog="data/smapshot/catalog.csv", lookahead=4):
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    out_dir = Path(out_dir)
    (out_dir / "photos").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    labeler = Labeler()
    ids = [l.strip() for l in open(id_list) if l.strip()]
    ids = [i for i in ids if not (out_dir / "labels" / f"{i}.npz").exists()]
    load = scene_groups(ids, pd.read_csv(catalog))
    pool = ThreadPoolExecutor(max_workers=lookahead)
    futures = {}

    def submit(k):
        if k < len(ids):
            futures[k] = pool.submit(labeler.prepare, ids[k], out_dir, int(width), load[k])

    for k in range(lookahead):
        submit(k)
    with open(out_dir / f"meta_{Path(id_list).stem}.jsonl", "a") as f:
        for k, image_id in enumerate(ids):
            if (out_dir / "labels" / f"{image_id}.npz").exists():   # fait entre-temps par un autre processus
                futures.pop(k, None)
                submit(k + lookahead)
                continue
            t = time.time()
            try:
                prep = futures.pop(k).result()
                submit(k + lookahead)
                meta = labeler.process(prep, out_dir)
            except Exception as e:  # une photo défectueuse ne bloque pas le lot
                submit(k + lookahead)
                meta = {"id": image_id, "error": repr(e)[:300]}
            meta["seconds"] = round(time.time() - t, 1)
            f.write(json.dumps(meta) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"{k}/{len(ids)} {json.dumps({kk: v for kk, v in meta.items() if kk in ('id', 'error', 'seconds')})}",
                      flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
