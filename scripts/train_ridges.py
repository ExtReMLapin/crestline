"""Entraîne le détecteur de crêtes (ciel + skyline + crêtes internes) sur GeoPose3K.

Lancement multi-GPU :
  uv run torchrun --nproc_per_node=3 scripts/train_ridges.py --out outputs/ridge_model
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent))
from ridge_data import RidgeDataset, load_meta, read_split, to_device, worker_init  # noqa: E402

LINE_POS_WEIGHT = 10.0    # les pixels de ligne sont rares : on les sur-pondère


def build_model(encoder="tu-convnext_base.fb_in22k", pretrained=True):
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, classes=3)


def load_trained(ckpt, device):
    """Recharge un modèle entraîné ; l'encodeur est lu dans config.json à côté du checkpoint."""
    config = json.load(open(Path(ckpt).parent / "config.json"))
    model = build_model(config.get("encoder", "tu-convnext_base.fb_in22k"), pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def loss_fn(logits, y, valid):
    """BCE masquée : ciel (cible binaire) + deux cartes de lignes (cibles douces sur-pondérées)."""
    v = valid.unsqueeze(1).float()
    sky = F.binary_cross_entropy_with_logits(logits[:, :1], y[:, :1], reduction="none")
    lines = F.binary_cross_entropy_with_logits(logits[:, 1:], y[:, 1:], reduction="none")
    lines = lines * (1 + (LINE_POS_WEIGHT - 1) * (y[:, 1:] > 0.1))
    denom = v.sum().clamp(min=1)
    return (sky * v).sum() / denom, (lines * v).sum() / (2 * denom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prep", default="data/geopose_prep", help="métadonnées (meta_*.csv)")
    p.add_argument("--cache", default="data/geopose_cache")
    p.add_argument("--workers", type=int, default=6, help="par GPU")
    p.add_argument("--encoder", default="tu-convnext_base.fb_in22k")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--splits", default="data/geopose3k/splits")
    p.add_argument("--out", default="outputs/ridge_model")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8, help="par GPU")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-sky-iou", type=float, default=0.9)
    p.add_argument("--compile", action="store_true", help="torch.compile (fusion des opérations)")
    p.add_argument("--grad-bf16", action="store_true", help="DDP : gradients échangés en bf16 (trafic / 2)")
    p.add_argument("--bench", type=int, default=0, help="mesure le débit sur N pas puis s'arrête")
    p.add_argument("--train-list", help="liste d'ids (remplace le découpage GeoPose3K)")
    p.add_argument("--val-list", help="liste d'ids de validation")
    p.add_argument("--init", help="poids de départ (checkpoint .pt)")
    args = p.parse_args()

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True           # recadrages de taille fixe : choix d'algorithmes mis en cache
    torch.set_float32_matmul_precision("high")      # TF32 pour ce qui reste en fp32
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(rank)
    np.random.seed(rank)

    if args.train_list:
        train_ids = [l.strip() for l in open(args.train_list) if l.strip()]
        val_ids = [l.strip() for l in open(args.val_list) if l.strip()]
    else:
        meta = load_meta(args.prep)
        good = set(meta.index[meta["sky_iou"] >= args.min_sky_iou])
        train_ids = [i for i in read_split(args.splits, "train") if i in good]
        val_ids = [i for i in read_split(args.splits, "val") if i in good]
    if args.bench:  # assez de pas dans une « époque » pour mesurer, quelle que soit la taille de batch
        train_ids = train_ids * 20
    if rank == 0:
        print(f"train {len(train_ids)} / val {len(val_ids)} photos", flush=True)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        json.dump(vars(args) | {"n_train": len(train_ids), "n_val": len(val_ids)},
                  open(Path(args.out) / "config.json", "w"), indent=2)

    train_ds = RidgeDataset(args.cache, train_ids, train=True, crop=args.crop)
    val_ds = RidgeDataset(args.cache, val_ids, train=True, crop=args.crop)  # recadrages aléatoires aussi en val (taille fixe)
    sampler = DistributedSampler(train_ds, shuffle=True)
    loader = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True, persistent_workers=True,
                  prefetch_factor=4, worker_init_fn=worker_init)
    train_dl = DataLoader(train_ds, sampler=sampler, drop_last=True, **loader)
    val_dl = DataLoader(val_ds, sampler=DistributedSampler(val_ds, shuffle=False, seed=0), **loader)

    raw = build_model(args.encoder)
    if args.init:
        raw.load_state_dict(torch.load(args.init, map_location="cpu"))
    raw = raw.to(device).to(memory_format=torch.channels_last)
    model = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[device.index])
    if args.grad_bf16:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=0.01, fused=True)
    total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)

    best = float("inf")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0, run = time.time(), torch.zeros(2, device=device)
        for step, (image, targets) in enumerate(train_dl):
            if args.bench and step == 20:            # après la compilation et le préchauffage
                torch.cuda.synchronize()
                t_bench = time.time()
            if args.bench and step == 20 + args.bench:
                torch.cuda.synchronize()
                rate = args.bench * args.batch * world / (time.time() - t_bench)
                if rank == 0:
                    print(f"BENCH {rate:.1f} images/s  ({vars(args)})", flush=True)
                dist.destroy_process_group()
                return
            x, y, valid = to_device(image, targets, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            l_sky, l_lines = loss_fn(logits.float(), y, valid)
            loss = l_sky + l_lines
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += torch.stack([l_sky.detach(), l_lines.detach()])  # pas de .item() : évite une synchro GPU par pas

        model.eval()
        val = torch.zeros(3, device=device)
        with torch.no_grad():
            for image, targets in val_dl:
                x, y, valid = to_device(image, targets, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                l_sky, l_lines = loss_fn(logits.float(), y, valid)
                val += torch.stack([l_sky, l_lines, torch.tensor(1.0, device=device)])
        dist.all_reduce(val)
        v_sky, v_lines = (val[:2] / val[2]).tolist()
        if rank == 0:
            n = len(train_dl)
            run = run.tolist()
            print(f"epoch {epoch:02d}  train sky {run[0]/n:.4f} lines {run[1]/n:.4f}  "
                  f"val sky {v_sky:.4f} lines {v_lines:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            state = raw.state_dict()
            if epoch % 10 == 9 or epoch == args.epochs - 1:
                torch.save(state, Path(args.out) / "last.pt")
            if v_sky + v_lines < best:
                best = v_sky + v_lines
                torch.save(state, Path(args.out) / "best.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
