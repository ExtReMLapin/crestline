"""Listes train / val / test des photos smapshot étiquetées, avec filtre de qualité.

Usage : python scripts/smapshot_splits.py <dossier_out> <selection.csv> <dossier_listes>

Une photo est gardée si ses étiquettes existent, sans erreur, et si sa skyline MNT colle à celle
de la photo (écart médian <= 4 px et p90 <= 15 px), ou, faute de skyline comparable, si elle a des
crêtes internes conservées. Le découpage (par maille de 10 km) vient de selection.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd

MAX_MED, MAX_P90 = 4.0, 15.0


def main(out_dir, selection_csv, list_dir):
    out_dir, list_dir = Path(out_dir), Path(list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for f in out_dir.glob("meta_*.jsonl"):
        metas += [json.loads(l) for l in open(f)]
    m = pd.DataFrame(metas).drop_duplicates("id", keep="last")
    m["id"] = m["id"].astype(str)
    m = m[m.get("error").isna()] if "error" in m else m
    m = m[[(out_dir / "labels" / f"{i}.npz").exists() for i in m.id]]
    sel = pd.read_csv(selection_csv, dtype={"id": str})[["id", "split", "year"]]
    m = m.merge(sel, on="id", how="inner")
    has_sky = m.skyline_err_med.notna()
    good_sky = has_sky & (m.skyline_err_med <= MAX_MED) & (m.skyline_err_p90 <= MAX_P90)
    # « crêtes seules » : vraies vues sans ciel (paroi, vue plongeante), donc presque pas de ciel MNT.
    # Exclut les scans de pages d'album (photo collée sur une page : le MNT met du ciel sur le papier).
    ridges_only = ~has_sky & (m.sky_frac < 0.05) & (m.ridge_px * m.ridge_kept > 200)
    m["keep"] = good_sky | ridges_only
    for split, g in m.groupby("split"):
        ids = g[g.keep].id
        ids.to_csv(list_dir / f"{split}.txt", index=False, header=False)
        print(f"{split}: {len(ids)} gardées / {len(g)} étiquetées "
              f"(skyline ok {int((good_sky & (m.split == split)).sum())}, crêtes seules {int((ridges_only & (m.split == split)).sum())})")
    m.to_csv(list_dir / "quality.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
