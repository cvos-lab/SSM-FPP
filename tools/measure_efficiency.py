"""Parameters, inference latency and peak GPU memory for the four networks.

    python tools/measure_efficiency.py [--json efficiency.json]

Run on an otherwise idle GPU. One 768 x 512 input, batch 1, FP32, under
torch.no_grad(); the input is created on the GPU beforehand, so data transfer
is excluded. Latency is the mean of --iters forward passes after --warmup
passes, with torch.cuda.synchronize() before and after the timed loop. Peak
memory is torch.cuda.max_memory_allocated() after reset_peak_memory_stats(),
covering warm-up and timed passes. Results depend on the GPU and software.
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import build_model  # noqa: E402

H, W = 512, 768
CONFIGS = ["hidnet", "nafnet", "restormer", "ssm_fpp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  input {H}x{W}, batch 1, fp32")
    rows = []
    for name in CONFIGS:
        with open(os.path.join(ROOT, "configs", f"{name}.yaml")) as f:
            cfg = yaml.safe_load(f)
        m = build_model(cfg).to(dev).eval()
        x = torch.randn(1, cfg["in_channels"], H, W, device=dev)
        n_par = sum(p.numel() for p in m.parameters())

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(args.warmup):
                m(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                m(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.iters
        peak_mib = torch.cuda.max_memory_allocated() / 2 ** 20

        rows.append({"model": name, "params_M": n_par / 1e6, "latency_ms": dt * 1e3,
                     "frame_rate_Hz": 1.0 / dt, "peak_mem_GiB": peak_mib / 1024})
        print(f"{name:10s} {n_par / 1e6:6.2f} M  {dt * 1e3:7.1f} ms  "
              f"{1.0 / dt:5.1f} Hz  {peak_mib / 1024:5.2f} GiB")
        del m, x
        torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
