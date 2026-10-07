"""FLOPs per 768 x 512 input (batch 1) for the four networks.

    python tools/count_flops.py

Standard operations are counted with torch.utils.flop_counter.FlopCounterMode
(one multiply-add = 2 FLOPs). The Mamba layers run fused CUDA kernels that the
counter cannot see, so for SSM-FPP each Mamba layer is replaced during counting
by an analytic count over L tokens of width C (inner width D = expand * C,
state size N, dt rank R, conv kernel K):

    in-projection C -> 2D        2 L C (2D)
    depthwise causal conv1d      2 L D K
    x-projection D -> R + 2N     2 L D (R + 2N)
    dt-projection R -> D         2 L R D
    selective scan               9 L D N   (estimate: exp, state update, C.h, D-skip)
    out-projection D -> C        2 L D C
"""
import os
import sys

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import build_model  # noqa: E402

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

H, W = 512, 768
CONFIGS = ["hidnet", "nafnet", "restormer", "ssm_fpp"]
counted = {"mamba": 0, "scan": 0}


def mamba_flops(self, x):
    B, L, C = x.shape
    D, N, R, K = self.d_inner, self.d_state, self.dt_rank, self.d_conv
    scan = 9 * L * D * N
    f = (2 * L * C * 2 * D + 2 * L * D * K + 2 * L * D * (R + 2 * N)
         + 2 * L * R * D + scan + 2 * L * D * C)
    counted["mamba"] += B * f
    counted["scan"] += B * scan
    return torch.zeros_like(x)


def main():
    dev = torch.device("cuda")
    for name in CONFIGS:
        with open(os.path.join(ROOT, "configs", f"{name}.yaml")) as f:
            cfg = yaml.safe_load(f)
        m = build_model(cfg).to(dev).eval()
        counted["mamba"] = counted["scan"] = 0
        original = Mamba.forward if Mamba is not None else None
        if Mamba is not None:
            Mamba.forward = mamba_flops
        try:
            with torch.no_grad(), FlopCounterMode(display=False) as fc:
                m(torch.randn(1, cfg["in_channels"], H, W, device=dev))
        finally:
            if Mamba is not None:
                Mamba.forward = original
        total = fc.get_total_flops() + counted["mamba"]
        line = f"{name:10s} {total / 1e9:8.1f} GFLOPs"
        if counted["mamba"]:
            line += (f"  (Mamba layers {counted['mamba'] / 1e9:.1f} G, "
                     f"of which selective scan {counted['scan'] / 1e9:.1f} G)")
        print(line)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
