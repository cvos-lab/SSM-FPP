"""Train one network with distributed data parallel on all visible GPUs.

    python train.py --config configs/ssm_fpp.yaml --seed 42

Objective: 0.75 * MSE + 0.25 * (1 - SSIM) on normalised depth, full frame.
Adam (lr 1e-4), cosine annealing to 1e-6 over all epochs, one sample per GPU.
The checkpoint with the lowest validation loss is kept.

--seed changes weight initialisation, dropout and batch order only; the
training/validation split is always made with seed 42.
"""
import argparse
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12358")

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image import StructuralSimilarityIndexMeasure

from data.dataset import PrecomputedFringeDepthDataset
from models import build_model

SPLIT_SEED = 42


class L2SSIMLoss(nn.Module):
    """0.75 * MSE + 0.25 * (1 - SSIM); SSIM: Gaussian window 11, sigma 1.5, range 1."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, pred, target):
        return 0.75 * self.mse(pred, target) + 0.25 * (1 - self.ssim(pred, target))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def worker(rank, world_size, args, cfg):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    set_seed(args.seed)
    device = f"cuda:{rank}"

    dataset = PrecomputedFringeDepthDataset(
        fringe_path=os.path.join(args.data, "fringe_train.pt"),
        Z_path=os.path.join(args.data, "Z_train.pt"),
        ref_fringe_path=os.path.join(args.data, "ref_fringe.pt"),
        ref_bg_path=os.path.join(args.data, "ref_bg.pt"),
        input_mode=cfg.get("input_mode", "6ch"))
    n_val = int(cfg["validation_split"] * len(dataset))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(SPLIT_SEED))

    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    bs = cfg["per_gpu_batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, sampler=train_sampler, num_workers=4,
                              worker_init_fn=seed_worker, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=bs, sampler=val_sampler, num_workers=4,
                            worker_init_fn=seed_worker, pin_memory=True)

    model = DDP(build_model(cfg).to(device), device_ids=[rank], output_device=rank)
    loss_fn = L2SSIMLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    epochs = args.epochs or cfg["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["eta_min"])

    ckpt = os.path.join(args.out, f"{args.name}.pth")
    best_val, best_epoch, t_start = float("inf"), -1, time.time()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        tr = torch.zeros(2, device=device)
        for batch in train_loader:
            x = batch["fringe"].to(device, non_blocking=True)
            y = batch["Z"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            tr += torch.tensor([loss.item() * x.size(0), x.size(0)], device=device)

        model.eval()
        va = torch.zeros(2, device=device)
        with torch.no_grad():
            for batch in val_loader:
                x = batch["fringe"].to(device, non_blocking=True)
                y = batch["Z"].to(device, non_blocking=True)
                va += torch.tensor([loss_fn(model(x), y).item() * x.size(0), x.size(0)],
                                   device=device)
        dist.all_reduce(tr)
        dist.all_reduce(va)
        train_loss, val_loss = (tr[0] / tr[1]).item(), (va[0] / va[1]).item()

        if rank == 0:
            print(f"epoch {epoch + 1:3d}/{epochs}  train {train_loss:.6f}  val {val_loss:.6f}",
                  flush=True)
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch + 1
                torch.save(model.module.state_dict(), ckpt)
        scheduler.step()

    if rank == 0:
        with open(os.path.join(args.out, f"{args.name}.txt"), "w") as f:
            f.write(f"config: {args.config}\nseed: {args.seed}\n"
                    f"best validation loss: {best_val:.6f} at epoch {best_epoch}\n"
                    f"epochs: {epochs}\nGPUs: {world_size}\n"
                    f"training time: {(time.time() - t_start) / 60:.1f} min\n")
        print(f"best validation loss {best_val:.6f} at epoch {best_epoch} -> {ckpt}")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data/data_pt")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--name", default=None,
                    help="checkpoint name (default: <config name>_seed<seed>)")
    ap.add_argument("--epochs", type=int, default=None, help="override the config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    args.name = args.name or f"{os.path.splitext(os.path.basename(args.config))[0]}_seed{args.seed}"
    os.makedirs(args.out, exist_ok=True)

    world_size = torch.cuda.device_count()
    assert world_size > 0, "CUDA GPU required"
    mp.spawn(worker, args=(world_size, args, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
