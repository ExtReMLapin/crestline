"""Catalogue des photos smapshot validées des collections montagne (liste seulement, pas de pose).

Usage : python scripts/smapshot_catalog.py <sortie.csv>
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import smapshot as S  # noqa: E402

COLLECTIONS = {12: "Terrestrial images", 15: "Glaciers", 42: "Glaciers II", 45: "Glaciers III",
               46: "Glaciers IV", 57: "Mountains and glaciers"}
FIELDS = ["id", "collection_id", "title", "view_type", "date_shot_min", "date_shot_max",
          "latitude", "longitude", "width", "height", "image_url"]


def main(out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for cid, name in COLLECTIONS.items():
            offset, n = 0, 0
            while True:
                page = S.get(f"/images?collection_id={cid}&state[]=validated&limit=500&offset={offset}")
                rows = page["rows"]
                for r in rows:
                    writer.writerow({k: r.get(k) for k in FIELDS[:-1]} | {"image_url": r["media"]["image_url"]})
                n += len(rows)
                offset += len(rows)
                if not rows or offset >= page["count"]:
                    break
            print(f"{name}: {n} photos", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
