"""Jeu de données GeoPose3K préparé (sorties de prepare_geopose.py) pour le détecteur de crêtes."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset

LINE_SIGMA = 1.5          # adoucissement des lignes cibles (px), tolère le décalage du recalage MNT
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_split(split_dir, name):
    return [l.strip() for l in open(Path(split_dir) / f"geoPose3K_final_{name}.txt") if l.strip()]


def load_meta(prep_dir):
    meta = pd.concat([pd.read_csv(p) for p in sorted(Path(prep_dir).glob("meta_*.csv"))])
    return meta[meta["error"].isna()].set_index("id")


def soft_line(mask):
    """Ligne binaire -> carte douce (pic à 1) par flou gaussien de la distance à la ligne."""
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    dist = ndimage.distance_transform_edt(~mask)
    return np.exp(-0.5 * (dist / LINE_SIGMA) ** 2).astype(np.float32)


def top_sky_rows(sky):
    """Ligne de ciel par colonne = nombre de pixels de ciel consécutifs depuis le haut."""
    return np.argmin(np.vstack([sky, np.zeros((1, sky.shape[1]), bool)]), axis=0)


def normalize(image):
    return (image.astype(np.float32) / 255.0 - MEAN) / STD


def worker_init(_):
    """Un seul thread par worker : sinon OpenCV/OMP lancent un pool par processus (sur-souscription)."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class RidgeDataset(Dataset):
    """Lit le cache brut (build_cache.py) en mmap : une seule copie en RAM pour tous les processus.

    Renvoie des tenseurs uint8 (image HWC->CHW, cibles 0..255) ; la normalisation se fait sur GPU
    (voir to_device), ce qui divise par 4 les transferts vers le GPU.
    """

    def __init__(self, cache_dir, ids, train, crop=512):
        self.cache_dir = Path(cache_dir)
        self.ids = list(ids)
        self.train = train
        self.crop = crop

    def __len__(self):
        return len(self.ids)

    def augment(self, image, targets):
        """Échelle, roulis, symétrie, recadrage, puis photométrie (brume, exposition, saturation)."""
        h, w = image.shape[:2]
        scale = np.random.uniform(0.7, 1.4)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-4, 4), scale)
        M[:, 2] += np.array([self.crop, self.crop]) / 2 - np.array([w, h]) / 2   # centre dans le recadrage
        M[:, 2] += (np.random.uniform(-1, 1) * max(0, w * scale - self.crop) / 2,
                    np.random.uniform(-1, 1) * max(0, h * scale - self.crop) / 2)
        size = (self.crop, self.crop)
        image = cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # hors cadre : ciel/lignes = 0, « à ignorer » = 255
        targets = cv2.warpAffine(targets, M, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 255))
        if np.random.rand() < 0.5:
            image, targets = image[:, ::-1], targets[:, ::-1]
        image = image.astype(np.float32) * np.random.uniform(0.6, 1.4) + np.random.uniform(-30, 30)
        if np.random.rand() < 0.4:                                                    # brume
            haze = np.random.uniform(0.1, 0.5)
            image = (1 - haze) * image + haze * np.random.uniform(180, 255)
        if np.random.rand() < 0.3:
            image = cv2.GaussianBlur(image, (0, 0), np.random.uniform(0.5, 1.5))
        gray = image.mean(axis=2, keepdims=True)
        image = gray + (image - gray) * np.random.uniform(0.5, 1.5)                   # saturation
        return np.clip(image, 0, 255).astype(np.uint8), np.ascontiguousarray(targets)

    def __getitem__(self, i):
        a = np.load(self.cache_dir / f"{self.ids[i]}.npy", mmap_mode="r")
        image, targets = np.ascontiguousarray(a[..., :3]), np.ascontiguousarray(a[..., 3:])
        if self.train:
            image, targets = self.augment(image, targets)
        return (torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1),
                torch.from_numpy(targets).permute(2, 0, 1))


MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1) * 255
STD_T = torch.tensor(STD).view(1, 3, 1, 1) * 255


def to_device(image, targets, device):
    """uint8 -> float sur GPU : image normalisée, cibles (ciel, skyline, crêtes) dans [0,1], masque valide."""
    image = image.to(device, non_blocking=True).float()
    image = ((image - MEAN_T.to(device)) / STD_T.to(device)).contiguous(memory_format=torch.channels_last)
    targets = targets.to(device, non_blocking=True).float() / 255.0
    return image, targets[:, :3], targets[:, 3] < 0.5
