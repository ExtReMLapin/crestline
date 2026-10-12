"""Focale en pixels, lue dans l'EXIF.

L'EXIF donne la focale réelle (zoom compris), en mm. Pour la convertir en pixels, il faut la taille du capteur :
1. focale équivalente 35 mm (tag 41989), si l'appareil l'écrit ;
2. sinon, densité de pixels du plan focal (FocalPlaneXResolution, tags 41486/41488) ;
3. sinon : à fournir (--fov ou --focal-px).
"""
import math

from PIL import Image

EXIF_IFD = 0x8769
FOCAL_MM, FOCAL_35MM = 37386, 41989
FOCAL_PLANE_XRES, FOCAL_PLANE_UNIT = 41486, 41488
PIXEL_X_DIM = 40962
UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}     # pouce, cm, mm, µm


def focal_px_from_exif(path):
    """Focale en pixels, rapportée à la largeur réelle du fichier (avant rotation EXIF).
    Robuste aux photos redimensionnées : la taille du capteur est calculée sur les dimensions d'origine
    notées dans l'EXIF (PixelXDimension), pas sur celles du fichier.
    Renvoie (focale, largeur du fichier, méthode) ou (None, None, raison)."""
    img = Image.open(path)
    exif = img.getexif().get_ifd(EXIF_IFD)
    w, h = img.size
    f_mm = exif.get(FOCAL_MM)
    f35 = exif.get(FOCAL_35MM)
    if f35:
        # équivalent 35 mm : la diagonale du capteur vaut 43,27 mm
        return float(f35) / 43.27 * math.hypot(w, h), w, "EXIF focale équivalente 35 mm"
    xres = exif.get(FOCAL_PLANE_XRES)
    unit = UNIT_MM.get(exif.get(FOCAL_PLANE_UNIT, 2))
    if f_mm and xres and unit:
        orig_w = exif.get(PIXEL_X_DIM) or w
        sensor_w_mm = float(orig_w) / float(xres) * unit
        return float(f_mm) / sensor_w_mm * w, w, f"EXIF focale {float(f_mm):g} mm, capteur {sensor_w_mm:.2f} mm"
    return None, None, "focale absente de l'EXIF : utiliser --fov (champ horizontal, degrés) ou --focal-px"


def gps_from_exif(path):
    """Position GPS de l'EXIF (lat, lon), ou None."""
    try:
        gps = Image.open(path).getexif().get_ifd(0x8825)
        def deg(v, ref):
            d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            return -d if ref in ("S", "W") else d
        return deg(gps[2], gps[1]), deg(gps[4], gps[3])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, OSError):
        return None


def focal_px_from_fov(fov_deg, width):
    return width / 2 / math.tan(math.radians(fov_deg) / 2)
