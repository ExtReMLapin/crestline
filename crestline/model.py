"""Modèle Crestline : skyline et crêtes internes sur une photo de montagne (poids sur Hugging Face).

Le modèle sort 3 cartes de probabilité par pixel : ciel, skyline, crêtes internes.
https://huggingface.co/LPN64/crestline-convnext-base
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = "LPN64/crestline-convnext-base"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(repo_or_dir=None, device="cuda"):
    """Charge le modèle : dossier local ou dépôt Hugging Face (téléchargé une fois, cache HF standard, $HF_HOME).
    Défaut : $CRESTLINE_MODEL, sinon le dépôt public."""
    repo_or_dir = repo_or_dir or os.environ.get("CRESTLINE_MODEL") or REPO
    import segmentation_models_pytorch as smp
    from safetensors.torch import load_file
    path = Path(repo_or_dir)
    if not (path / "config.json").exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_or_dir, allow_patterns=["config.json", "model.safetensors"]))
    config = json.loads((path / "config.json").read_text())
    model = smp.Unet(config["encoder"], encoder_weights=None, classes=3)
    model.load_state_dict({k: v.float() for k, v in load_file(path / "model.safetensors").items()})  # stocké en bf16
    return model.to(device).eval(), config


@torch.no_grad()
def predict(model, image):
    """image : PIL.Image RGB. Renvoie les probabilités (3, H, W) : ciel, skyline, crêtes internes."""
    a = (np.asarray(image, np.float32) / 255 - MEAN) / STD
    device = next(model.parameters()).device
    x = torch.from_numpy(a).permute(2, 0, 1)[None].to(device)
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
    else:
        logits = model(x)
    return torch.sigmoid(logits.float())[0, :, :h, :w].cpu().numpy()


def prepare(image, long_side=1536):
    """Rotation EXIF appliquée, RGB, grand côté ramené à `long_side`."""
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((long_side, long_side), Image.LANCZOS)
    return image
