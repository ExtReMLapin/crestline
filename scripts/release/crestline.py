"""Crestline — détection de la skyline et des crêtes internes sur des photos de montagne.

    pip install torch segmentation-models-pytorch timm safetensors pillow huggingface_hub
    python crestline.py photo.jpg sortie.jpg

Le modèle prédit 3 cartes par pixel : ciel, skyline, crêtes internes (contours d'occultation).
"""
import json
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import load_file

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(repo_or_dir=".", device=None):
    """Charge le modèle depuis un dossier local ou un dépôt Hugging Face."""
    path = Path(repo_or_dir)
    if not (path / "config.json").exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_or_dir))
    config = json.load(open(path / "config.json"))
    model = smp.Unet(config["encoder"], encoder_weights=None, classes=3)
    model.load_state_dict({k: v.float() for k, v in load_file(path / "model.safetensors").items()})  # stocké en bf16
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), config


@torch.no_grad()
def predict(model, image, long_side=1536):
    """image : PIL.Image. Renvoie l'image redimensionnée et les probabilités (3, H, W) : ciel, skyline, crêtes."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((long_side, long_side), Image.LANCZOS)
    a = (np.asarray(image, np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(a).permute(2, 0, 1)[None].to(next(model.parameters()).device)
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
    prob = torch.sigmoid(model(x))[0, :, :h, :w].float().cpu().numpy()
    return image, prob


def lines(prob, ridge_threshold=0.4):
    """Skyline (ligne par colonne) et masque des crêtes internes (maxima verticaux au-dessus du seuil)."""
    sky = prob[0] > 0.5
    skyline_rows = np.argmin(np.vstack([sky, np.zeros((1, sky.shape[1]), bool)]), axis=0)
    r = prob[2]
    up = np.pad(r, ((1, 0), (0, 0)), mode="edge")[:-1]
    down = np.pad(r, ((0, 1), (0, 0)), mode="edge")[1:]
    ridges = (r > ridge_threshold) & (r >= up) & (r >= down)
    return skyline_rows, ridges


def overlay(image, skyline_rows, ridges, width=3):
    a = np.asarray(image).copy()
    h, w = a.shape[:2]
    from scipy import ndimage
    a[ndimage.binary_dilation(ridges, iterations=width // 2)] = (0, 230, 255)
    cols = np.arange(w)
    for dy in range(-(width // 2), width // 2 + 1):
        a[np.clip(skyline_rows + dy, 0, h - 1), cols] = (40, 255, 40)
    return Image.fromarray(a)


if __name__ == "__main__":
    model, config = load(Path(__file__).parent)
    img, prob = predict(model, Image.open(sys.argv[1]))
    rows, ridges = lines(prob, config.get("ridge_threshold", 0.4))
    overlay(img, rows, ridges).save(sys.argv[2], quality=92)
