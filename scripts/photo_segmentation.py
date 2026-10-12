"""Segmentation ADE20K (Mask2Former) des photos GeoPose3K, à la résolution du rendu MNT.

Usage : python scripts/photo_segmentation.py <prep> <sortie> <shard> <n_shards>
Écrit <sortie>/<id>.npz avec `labels` (H, W) uint8 (ids ADE20K du modèle).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

SEG_REPO = "facebook/mask2former-swin-large-ade-semantic"


@torch.no_grad()
def main(prep, out, shard, n_shards):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(SEG_REPO)
    model = AutoModelForUniversalSegmentation.from_pretrained(SEG_REPO).to("cuda").eval()
    files = sorted(Path(prep).glob("*.npz"))[int(shard)::int(n_shards)]
    for i, f in enumerate(files):
        if (out / f.name).exists():
            continue
        image = np.load(f)["image"]
        inputs = processor(images=Image.fromarray(image), return_tensors="pt").to("cuda")
        labels = processor.post_process_semantic_segmentation(
            model(**inputs), target_sizes=[image.shape[:2]])[0].cpu().numpy().astype(np.uint8)
        np.savez_compressed(out / f.name, labels=labels)
        if i % 500 == 0:
            print(f"{i}/{len(files)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
