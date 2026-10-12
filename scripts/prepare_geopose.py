"""Prépare le jeu d'entraînement du détecteur de crêtes à partir de GeoPose3K.

Usage : python scripts/prepare_geopose.py <dossier_extrait> <dossier_sortie> <shard> <n_shards>

Pour chaque photo, écrit <sortie>/<id>.npz contenant, à la résolution du rendu :
  image    (H, W, 3) uint8   photo redimensionnée sur le rendu « pinhole »
  sky      (H, W)    bool    ciel selon le MNT (distance <= 0)
  skyline  (H, W)    bool    bord du ciel selon le MNT
  ridge    (H, W)    bool    crêtes internes (sauts de profondeur) selon le MNT
  ignore   (H, W)    bool    occultants vus sur la photo (arbres, bâtiments...) : hors perte
  sky_iou  float             accord ciel MNT / ciel photo, hors occultants (qualité d'alignement)
et une ligne de métadonnées dans <sortie>/meta_<shard>.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

sys.path.insert(0, str(Path(__file__).parent))
from geopose_labels import load_sample, ridge_labels  # noqa: E402

SEG_REPO = "facebook/mask2former-swin-large-ade-semantic"
OCCLUDERS = {
    "tree", "plant", "flower", "person", "apparel", "bag", "pole", "fence", "signboard", "column",
    "car", "chair", "tent", "building", "house", "skyscraper", "tower", "palm", "railing",
    "bridge", "streetlight", "awning", "windowpane", "door", "roof", "bench", "boat", "truck", "bus",
    "van", "bicycle", "minibike", "animal", "flag", "umbrella", "stairs", "step", "booth", "hovel",
    "ship", "lamp", "car", "canopy", "grandstand",
}
IGNORE_DILATE_PX = 3


@torch.no_grad()
def photo_segmentation(processor, model, photo, device):
    inputs = processor(images=photo, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(photo.height, photo.width)])[0].cpu().numpy()


def main(extracted, out_dir, shard, n_shards):
    shard, n_shards = int(shard), int(n_shards)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    processor = AutoImageProcessor.from_pretrained(SEG_REPO)
    model = AutoModelForUniversalSegmentation.from_pretrained(SEG_REPO).to(device).eval()
    name2id = {v: k for k, v in model.config.id2label.items()}
    sky_id = name2id["sky"]
    occ_ids = [name2id[n] for n in OCCLUDERS if n in name2id]

    samples = sorted(p for p in Path(extracted).iterdir() if (p / "pinhole/distance_crop.pfm.gz").exists())
    samples = [s for s in samples[shard::n_shards] if not (out_dir / f"{s.name}.npz").exists()]
    meta_path = out_dir / f"meta_{shard}.csv"
    if meta_path.exists():  # reprise : on garde les lignes réussies, on refait les autres
        import pandas as pd
        ok = pd.read_csv(meta_path)
        ok[ok["error"].isna()].to_csv(meta_path, index=False)
    with open(meta_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["id", "mode", "lat", "lon", "alt", "fov", "yaw", "pitch", "roll",
                             "height", "width", "sky_iou", "skyline_px", "ridge_px", "ignore_frac", "error"])
        for i, sample in enumerate(samples):
            try:
                photo, distance, info = load_sample(sample)
                sky = distance <= 0
                skyline, ridge = ridge_labels(distance)
                seg = photo_segmentation(processor, model, photo, device)
                ignore = ndimage.binary_dilation(np.isin(seg, occ_ids), iterations=IGNORE_DILATE_PX)
                photo_sky = seg == sky_id
                keep = ~ignore
                inter = (sky & photo_sky & keep).sum()
                union = ((sky | photo_sky) & keep).sum()
                sky_iou = float(inter / union) if union else float("nan")
                np.savez_compressed(out_dir / f"{sample.name}.npz", image=np.asarray(photo), sky=sky,
                                    skyline=skyline, ridge=ridge, ignore=ignore, sky_iou=sky_iou)
                writer.writerow([sample.name, info["mode"], info["lat"], info["lon"], info["alt"],
                                 info["fov"], info["yaw"], info["pitch"], info["roll"],
                                 sky.shape[0], sky.shape[1], round(sky_iou, 4), int(skyline.sum()),
                                 int(ridge.sum()), round(float(ignore.mean()), 4), ""])
            except Exception as e:  # une photo défectueuse ne doit pas arrêter le lot
                writer.writerow([sample.name] + [""] * 14 + [repr(e)[:200]])
            if i % 100 == 0:
                print(f"shard {shard}: {i}/{len(samples)}", flush=True)
                f.flush()


if __name__ == "__main__":
    main(*sys.argv[1:5])
