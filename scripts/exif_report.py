"""Lit les EXIF des photos (local uniquement) et écrit outputs/exif.csv + des vignettes."""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS


def dms_to_deg(dms, ref):
    deg = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
    return -deg if ref in ("S", "W") else deg


def read_exif(path):
    img = Image.open(path)
    raw = img.getexif()
    exif = {TAGS.get(k, k): v for k, v in raw.items()}
    exif.update({TAGS.get(k, k): v for k, v in raw.get_ifd(0x8769).items()})
    gps = {GPSTAGS.get(k, k): v for k, v in raw.get_ifd(0x8825).items()}
    row = {
        "file": path.name,
        "width": img.width,
        "height": img.height,
        "make": exif.get("Make"),
        "model": exif.get("Model"),
        "orientation": exif.get("Orientation"),
        "datetime": exif.get("DateTimeOriginal"),
        "focal_mm": float(exif["FocalLength"]) if "FocalLength" in exif else None,
        "focal_35mm": exif.get("FocalLengthIn35mmFilm"),
        "lat": dms_to_deg(gps["GPSLatitude"], gps["GPSLatitudeRef"]) if "GPSLatitude" in gps else None,
        "lon": dms_to_deg(gps["GPSLongitude"], gps["GPSLongitudeRef"]) if "GPSLongitude" in gps else None,
        "alt_m": float(gps["GPSAltitude"]) if "GPSAltitude" in gps else None,
        "img_direction": float(gps["GPSImgDirection"]) if "GPSImgDirection" in gps else None,
        "direction_ref": gps.get("GPSImgDirectionRef"),
        "h_accuracy_m": float(gps["GPSHPositioningError"]) if "GPSHPositioningError" in gps else None,
    }
    return row, ImageOps.exif_transpose(img)


def main(photo_dir, out_dir):
    out_dir = Path(out_dir)
    rows = []
    for path in sorted(Path(photo_dir).glob("*.[jJ][pP][gG]")):
        row, img = read_exif(path)
        rows.append(row)
        img.thumbnail((1024, 1024))
        img.save(out_dir / "thumbs" / f"{path.stem}.jpg", quality=85)
    with open(out_dir / "exif.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
