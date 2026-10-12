"""Localiser une photo sur une carte : les deux modes partagent tout sauf la présélection.

    photo ──▶ Crestline ──▶ silhouette ─┬─ mode brute : comparaison avec TOUS les points de la carte
                                        └─ mode index : l'index propose ~1 000 lieux, comparés exactement
                                                  │
                                                  ▼
                                   100 meilleurs ──▶ raffinement 25 m ──▶ top 5
"""
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import model as skyline_model
from .camera import focal_px_from_exif, focal_px_from_fov, gps_from_exif
from .dem import true_north_offset
from .maps import Map, gpu_devices
from .match import (MIN_SKYLINE_CONF, PITCHES, MIN_BINS, Query, fit_pose, photo_profiles, profile_ffts, refine, reliable_columns,
                    score_all, skyline_rows, top_candidates)


def confidence(rms):
    """Indice de confiance (heuristique) : rapport entre l'écart du 2e lieu et celui du 1er.
    Sur nos essais, les bonnes réponses ont un rapport ≥ 1,3 ; les fausses restent sous 1,1."""
    if len(rms) < 2 or not np.isfinite(rms[0]):
        return None, "inconnue"
    ratio = float(rms[1] / max(rms[0], 1e-6))
    return ratio, "élevée" if ratio >= 1.25 else "moyenne" if ratio >= 1.12 else "faible"


def links(lat, lon):
    return {"openstreetmap": f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=15/{lat:.6f}/{lon:.6f}",
            "google_maps": f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"}


class Locator:
    """Garde en mémoire le modèle, la carte et son MNT entre plusieurs photos."""

    def __init__(self, map_dir, devices=None, load_model=True, model=None):
        """model : dépôt Hugging Face ou dossier local de Crestline (défaut : $CRESTLINE_MODEL, sinon le dépôt public)."""
        self.map = Map(map_dir)
        self.devices = devices or gpu_devices()
        self.device = self.devices[0]
        self.model = skyline_model.load(model, device=self.device)[0] if load_model else None
        self.points = self.map.points
        self._gpu_elev = None

    def gpu_elev(self):
        """Mode brute : horizons de la carte répartis sur les GPU, gardés entre deux photos."""
        if self._gpu_elev is None:
            elev = self.map.elev
            b = np.linspace(0, len(elev), len(self.devices) + 1).astype(int)
            self._gpu_elev = [torch.from_numpy(np.array(elev[b[i]:b[i + 1]])).to(d)
                              for i, d in enumerate(self.devices)]
        return self._gpu_elev

    def locate(self, photo, focal_px=None, fov=None, mode="auto", k=5, n_places=1000, n_refine=100,
               min_conf=MIN_SKYLINE_CONF):
        """photo : chemin. Renvoie un dict (candidats, image, silhouette...) ; voir output.write."""
        photo = Path(photo)
        gps = gps_from_exif(photo)
        if gps:
            print(f"{photo.name} : position GPS trouvée dans l'EXIF ({gps[0]:.5f}, {gps[1]:.5f}). Elle n'est pas "
                  "utilisée pour chercher, on lance quand même ; l'écart sera affiché à la fin.")
        raw = Image.open(photo)
        image = skyline_model.prepare(raw)
        if focal_px is not None:
            focal, how = float(focal_px) * image.width / raw.width, "--focal-px"
        elif fov is not None:
            focal, how = focal_px_from_fov(fov, image.width), "--fov"
        else:
            f, _, how = focal_px_from_exif(photo)
            if f is None:
                raise ValueError(f"{photo.name} : {how}")
            # la focale EXIF se rapporte à l'image enregistrée ; après rotation EXIF, le grand côté reste le même
            focal = f * max(image.size) / max(raw.size)
        prob = skyline_model.predict(self.model, image)
        rows = skyline_rows(prob[0] > 0.5)
        ok = reliable_columns(prob, rows, min_conf)
        result = {"photo": str(photo), "image": image, "prob": prob, "rows": rows, "reliable": ok, "focal_px": focal,
                  "focal_source": how, "hfov_deg": 2 * math.degrees(math.atan(image.width / 2 / focal))}
        result.update(self.search(rows, image.height, focal, ok, mode, k, n_places, n_refine))
        if gps:
            result["exif_gps"] = list(gps)
            if result["candidates"]:
                c = result["candidates"][0]
                result["gps_error_m"] = math.hypot((c["lat"] - gps[0]) * 111_320,
                                                   (c["lon"] - gps[1]) * 111_320 * math.cos(math.radians(gps[0])))
        return result

    def search(self, rows, height, focal, col_ok=None, mode="auto", k=5, n_places=1000, n_refine=100):
        """Skyline (ligne par colonne) -> candidats. Indépendant de Crestline (testable sans modèle)."""
        P, M, k_center = photo_profiles(rows, height, focal, col_ok=col_ok)
        j0 = int(np.argmin(np.abs(PITCHES)))
        if M[j0].sum() < MIN_BINS:
            return {"candidates": [], "error": "skyline fiable trop courte (nuages, pas de ciel visible ?)"}
        if mode == "auto":
            mode = "index" if self.map.has_index else "brute"
        q = Query(P, M, self.device)
        pts = self.points[:, :2]
        if mode == "brute":
            rms, shift, pitch = score_all((P, M), self.gpu_elev())
            pos = pts
        else:
            idx = np.sort(self.map.index().query(P[j0], M[j0], k_center, n_places, device=self.device))
            B, B2 = profile_ffts(torch.as_tensor(np.asarray(self.map.elev[idx]), device=self.device))
            rms, shift, pitch = (t.cpu().numpy() for t in q.score(B, B2))
            pos = pts[idx]
        dem = self.map.dem(self.device)
        cand = top_candidates(pos, rms, n_refine, radius=500.0)
        pos, rms, shift, pitch = refine(dem, q, pos[cand], self.map.camera_height)
        best = top_candidates(pos, rms, k)
        delta = 360.0 / q.n_az
        ratio, level = confidence(rms[best])
        out = {"mode": mode, "confidence": level, "confidence_ratio": ratio, "candidates": []}
        for rank, i in enumerate(best, 1):
            x, y = float(pos[i, 0]), float(pos[i, 1])
            lon, lat = self.map.to_lonlat(x, y)
            heading_grid = float(((shift[i] + k_center) * delta) % 360)
            c = {"rank": rank, "lat": lat, "lon": lon, "x": x, "y": y, "score_deg": float(rms[i]),
                 "heading_grid": heading_grid, "pitch_hyp": float(PITCHES[pitch[i]])}
            c["pitch"], c["roll"] = fit_pose(dem, P, M, k_center, c, self.map.camera_height)
            c["heading"] = (heading_grid - true_north_offset(self.map.crs, x, y)) % 360
            c["links"] = links(lat, lon)
            out["candidates"].append(c)
        return out
