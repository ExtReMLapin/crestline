"""Version GPU du recalage des étiquettes (même méthode et même score que realign_labels.py).

La recherche de la similitude (dx, dy, angle, échelle) est faite sur GPU par lots :
1) grille exhaustive grossière à demi-résolution, 2) deux grilles locales à pleine résolution.
La génération finale des étiquettes réutilise le code CPU (realign_labels.finalize).

Usage : python scripts/realign_labels_gpu.py <extrait> <prep> <photo_seg> <sortie> <shard> <n_shards>
Écrit <sortie>/<id>.npz et <sortie>/meta_realign_<shard>.csv
"""
import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import realign_labels as R  # noqa: E402
from geopose_labels import read_pfm_gz  # noqa: E402

BATCH = 1024


def inverse_grids(params, h, w, device):
    """Grilles d'échantillonnage (N, h, w, 2) réalisant cv2.warpAffine(M) pour chaque jeu de paramètres.

    M est la similitude de R.similarity (convention cv2.getRotationMatrix2D, centre de l'image).
    """
    p = torch.as_tensor(params, dtype=torch.float32, device=device)
    dx, dy, angle, log_s = p.unbind(1)
    a = torch.deg2rad(angle)
    s = torch.exp(log_s)
    alpha, beta = s * torch.cos(a), s * torch.sin(a)
    cx, cy = w / 2, h / 2
    tx = (1 - alpha) * cx - beta * cy + dx * w
    ty = beta * cx + (1 - alpha) * cy + dy * w
    # inverse de [[alpha, beta, tx], [-beta, alpha, ty]] : rotation inverse / s²
    det = alpha * alpha + beta * beta
    ia, ib = alpha / det, -beta / det           # [[ia, ib], [-ib, ia]]
    itx = -(ia * tx + ib * ty)
    ity = -(-ib * tx + ia * ty)
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    sx = ia[:, None, None] * xs + ib[:, None, None] * ys + itx[:, None, None]
    sy = -ib[:, None, None] * xs + ia[:, None, None] * ys + ity[:, None, None]
    # pixels -> coordonnées normalisées (align_corners=False)
    return torch.stack([(2 * sx + 1) / w - 1, (2 * sy + 1) / h - 1], dim=-1)


@torch.no_grad()
def scores(params, terrain, photo_rows, usable, clip):
    """Score (même formule que R.objective) pour un lot de paramètres. terrain : (h, w) float 0/1."""
    h, w = terrain.shape
    device = terrain.device
    src = torch.stack([terrain, torch.ones_like(terrain)])[None]          # canaux : terrain, dans le cadre
    out = []
    cols = torch.arange(w, device=device)
    rows_c = photo_rows.clamp(0, h - 1)
    batch = max(16, min(BATCH, int(2e8 // (h * w))))  # ~4 Go de mémoire GPU par lot, quelle que soit l'image
    for i in range(0, len(params), batch):
        chunk = params[i:i + batch]
        grid = inverse_grids(chunk, h, w, device)
        warped = F.grid_sample(src.expand(len(chunk), -1, -1, -1), grid, mode="nearest",
                               padding_mode="zeros", align_corners=False)
        is_terrain = warped[:, 0] > 0.5                                    # (N, h, w)
        in_frame = warped[:, 1] > 0.5
        has = is_terrain.any(dim=1)
        dem_rows = torch.where(has, is_terrain.to(torch.uint8).argmax(dim=1), torch.full_like(cols, h)[None])
        unknown = ~in_frame[:, rows_c, cols]                               # (N, w)
        ok = usable[None] & ~unknown & (dem_rows < h)
        err = (dem_rows - photo_rows[None]).abs().float().clamp(max=clip)
        frac = ok.float().mean(dim=1)
        score = (err * ok).sum(dim=1) / ok.sum(dim=1).clamp(min=1)
        c = torch.as_tensor(chunk, dtype=torch.float32, device=device)
        score = score + 0.5 * (c[:, 2].abs() / 3 + c[:, 3].abs() / 0.05)
        score = torch.where(frac < R.MIN_COLS * 0.5, torch.full_like(score, clip * 10), score)
        out.append(score)
    return torch.cat(out)


def local_grid(center, steps, n=5):
    offsets = np.arange(n) - n // 2
    return np.array([center + np.array(o) * steps for o in itertools.product(offsets, repeat=4)])


def search(distance, photo_rows, usable, device):
    h, w = distance.shape
    clip = R.CLIP_PX * h
    terrain = torch.as_tensor(np.nan_to_num(distance, nan=-1.0) > 0, dtype=torch.float32, device=device)
    rows_t = torch.as_tensor(photo_rows, device=device)
    usable_t = torch.as_tensor(usable, device=device)
    # 1) grille grossière à demi-résolution (même grille que la version CPU)
    coarse = np.array(list(itertools.product(np.linspace(-0.08, 0.08, 17), np.linspace(-0.08, 0.08, 17),
                                             np.linspace(-4, 4, 9), np.log(np.linspace(0.94, 1.06, 7)))))
    s_half = scores(coarse, terrain[::2, ::2].contiguous(), rows_t[::2] // 2, usable_t[::2], clip / 2)
    best = coarse[int(s_half.argmin())]
    # 2) affinage local à pleine résolution, pas divisé par 2 à chaque tour
    steps = np.array([0.005, 0.005, 0.5, 0.005])
    for _ in range(2):
        cand = np.vstack([local_grid(best, steps), np.zeros((1, 4))])     # l'identité reste candidate
        cand[:, 0:2] = cand[:, 0:2].clip(-0.1, 0.1)
        cand[:, 2] = cand[:, 2].clip(-5, 5)
        cand[:, 3] = cand[:, 3].clip(-0.08, 0.08)
        s_full = scores(cand, terrain, rows_t, usable_t, clip)
        best = cand[int(s_full.argmin())]
        steps = steps / 2
    return best


def main(extracted, prep, photo_seg, out, shard, n_shards):
    from transformers import AutoConfig
    from prepare_geopose import OCCLUDERS
    id2label = AutoConfig.from_pretrained("facebook/mask2former-swin-large-ade-semantic").id2label
    occ_ids = [k for k, v in id2label.items() if v in OCCLUDERS]
    device = "cuda"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in Path(prep).glob("*.npz") if (Path(photo_seg) / f.name).exists())
    files = files[int(shard)::int(n_shards)]
    with open(out / f"meta_realign_{shard}.csv", "w", newline="") as fcsv:
        writer = None
        for i, f in enumerate(files):
            sample_dir = Path(extracted) / f.stem
            distance = read_pfm_gz(sample_dir / "pinhole/distance_crop.pfm.gz")
            seg = np.load(Path(photo_seg) / f.name)["labels"]
            photo_rows, usable = R.photo_skyline(seg, np.isin(seg, occ_ids))
            best = search(distance, photo_rows, usable, device) if usable.mean() >= R.MIN_COLS else np.zeros(4)
            row = R.finalize(f, distance, seg, occ_ids, best, out / f.name)
            if writer is None:
                writer = csv.DictWriter(fcsv, fieldnames=sorted(row))
                writer.writeheader()
            writer.writerow(row)
            if i % 200 == 0:
                print(f"shard {shard}: {i}/{len(files)}", flush=True)
                fcsv.flush()


if __name__ == "__main__":
    main(*sys.argv[1:7])
