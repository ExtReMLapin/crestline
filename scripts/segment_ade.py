"""Segmentation sémantique ADE20K des photos (inférence locale, GPU).

Usage : python scripts/segment_ade.py <modele> <dossier_photos> <dossier_sortie>
  modele : mask2former | oneformer | segformer

Sortie par photo : <stem>.npz (carte de labels ADE20K, résolution de travail)
et <stem>.png (superposition colorée ciel / relief / occultants).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation, AutoModelForSemanticSegmentation

MODELS = {
    "mask2former": "facebook/mask2former-swin-large-ade-semantic",
    "oneformer": "shi-labs/oneformer_ade20k_swin_large",
    "segformer": "nvidia/segformer-b5-finetuned-ade-640-640",
}
WORK_LONG_SIDE = 1536

# Regroupement des 150 classes ADE20K pour la skyline.
TERRAIN = {"mountain", "hill", "rock", "earth", "grass", "field", "sand", "land", "path", "dirt track"}
WATER = {"water", "sea", "lake", "river"}


def load(name, device):
    repo = MODELS[name]
    processor = AutoImageProcessor.from_pretrained(repo)
    if name == "segformer":
        model = AutoModelForSemanticSegmentation.from_pretrained(repo)
    else:
        model = AutoModelForUniversalSegmentation.from_pretrained(repo)
    return processor, model.to(device).eval()


@torch.no_grad()
def segment(name, processor, model, image, device):
    if name == "oneformer":
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    else:
        inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    outputs = model(**inputs)
    size = [(image.height, image.width)]
    if name == "segformer":
        logits = torch.nn.functional.interpolate(outputs.logits, size=size[0], mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].cpu().numpy()
    return processor.post_process_semantic_segmentation(outputs, target_sizes=size)[0].cpu().numpy()


def overlay(image, labels, id2label):
    names = np.array([id2label[i] for i in range(len(id2label))])
    cls = names[labels]
    color = np.zeros((*labels.shape, 3), np.uint8)
    color[cls == "sky"] = (80, 160, 255)
    color[np.isin(cls, list(TERRAIN))] = (200, 120, 40)
    color[np.isin(cls, list(WATER))] = (0, 200, 200)
    other = (cls != "sky") & ~np.isin(cls, list(TERRAIN | WATER))
    color[other] = (230, 40, 160)  # occultants : arbres, personnes, objets...
    return Image.blend(image, Image.fromarray(color), 0.45)


def main(name, photo_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load(name, device)
    id2label = model.config.id2label
    for path in sorted(Path(photo_dir).glob("*.[jJ][pP][gG]")):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image.thumbnail((WORK_LONG_SIDE, WORK_LONG_SIDE))
        labels = segment(name, processor, model, image, device).astype(np.uint8)
        np.savez_compressed(out_dir / f"{path.stem}.npz", labels=labels)
        overlay(image, labels, id2label).save(out_dir / f"{path.stem}.png")
        present = {id2label[int(i)]: round(float((labels == i).mean()), 3) for i in np.unique(labels)}
        print(path.stem, dict(sorted(present.items(), key=lambda kv: -kv[1])[:6]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
