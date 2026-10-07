"""Evaluate a trained network on the 69 held-out measurements.

    python evaluate.py --config configs/ssm_fpp.yaml --ckpt checkpoints/ssm_fpp_seed42.pth

Depth is scored in millimetres, Z_mm = 350 * Z_norm - 200.

Foreground mask (from the reference depth only, identical for every model):
    valid    reference depth > -200 mm (the no-return value)
    plane    the dominant depth, i.e. the mode of a 1-mm histogram of valid depths
    mask     valid and |Z - plane| > 5 mm, eroded twice with a 4-connected element

Metrics, computed per measurement and then averaged over measurements:
    MAE, RMSE (mm); AbsRel; GE = mean |grad(Z_pred) - grad(Z_ref)| (mm/pixel,
    central differences); Rough = RMS of the 5-point Laplacian of the error
    (mm/pixel^2); <0.5 / <1 / <2 mm = percentage of mask pixels within tolerance.
"""
import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

from data.dataset import PrecomputedFringeDepthDataset
from models import build_model

MM_SCALE, MM_OFFSET = 350.0, -200.0
FLOOR_MM = -200.0
PLANE_MARGIN_MM = 5.0
ERODE = 2


def to_mm(norm):
    return norm * MM_SCALE + MM_OFFSET


def erode(mask, k):
    """Binary erosion, k iterations of a 4-connected (cross-shaped) element."""
    m = mask.copy()
    for _ in range(k):
        p = np.pad(m, 1, mode="constant", constant_values=False)
        m = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:] & p[1:-1, 1:-1]
    return m


def foreground_mask(ref_mm):
    valid = ref_mm > FLOOR_MM + 1e-6
    if valid.sum() == 0:
        return valid
    v = ref_mm[valid]
    hist, edges = np.histogram(v, bins=np.arange(v.min(), v.max() + 1.0, 1.0))
    if len(hist) == 0:
        return valid
    plane = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    return erode(valid & (np.abs(ref_mm - plane) > PLANE_MARGIN_MM), ERODE)


def metrics(pred_mm, ref_mm, mask):
    e = pred_mm - ref_mm
    sel = np.abs(e)[mask]
    if sel.size == 0:
        return None
    out = {"MAE": sel.mean(), "RMSE": np.sqrt((sel ** 2).mean())}

    g, p = ref_mm[mask], pred_mm[mask]
    ok = (np.abs(g) > 1.0) & (np.sign(g) == np.sign(p)) & (np.abs(p) > 1e-6)
    out["AbsRel"] = (np.abs(p[ok] - g[ok]) / np.abs(g[ok])).mean() if ok.sum() else np.nan

    for t in (0.5, 1.0, 2.0):
        out[f"<{t:g}mm"] = 100.0 * (sel < t).mean()

    gp_y, gp_x = np.gradient(pred_mm)
    gg_y, gg_x = np.gradient(ref_mm)
    out["GE"] = np.sqrt((gp_y - gg_y) ** 2 + (gp_x - gg_x) ** 2)[mask].mean()

    lap = (-4.0 * e + np.roll(e, 1, 0) + np.roll(e, -1, 0)
           + np.roll(e, 1, 1) + np.roll(e, -1, 1))
    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = True
    m2 = mask & inner
    out["Rough"] = np.sqrt((lap[m2] ** 2).mean()) if m2.sum() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/data_pt")
    ap.add_argument("--csv", default=None, help="optional per-measurement output")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PrecomputedFringeDepthDataset(
        fringe_path=os.path.join(args.data, "fringe_test.pt"),
        Z_path=os.path.join(args.data, "Z_test.pt"),
        ref_fringe_path=os.path.join(args.data, "ref_fringe.pt"),
        ref_bg_path=os.path.join(args.data, "ref_bg.pt"),
        input_mode=cfg.get("input_mode", "6ch"))

    model = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    sd = OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
    model.load_state_dict(sd)
    model.eval()

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            ref = to_mm(item["Z"].squeeze().numpy().astype(np.float64))
            pred = to_mm(model(item["fringe"].unsqueeze(0).to(device))
                         .squeeze().cpu().numpy().astype(np.float64))
            r = metrics(pred, ref, foreground_mask(ref))
            r["measurement"] = ds.get_original_index(i)
            rows.append(r)

    cols = ["MAE", "RMSE", "AbsRel", "GE", "Rough", "<0.5mm", "<1mm", "<2mm"]
    print(f"{os.path.basename(args.ckpt)}: foreground, mean over {len(rows)} measurements")
    for c in cols:
        print(f"  {c:7s} {np.nanmean([r[c] for r in rows]):.4f}")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("measurement," + ",".join(cols) + "\n")
            for r in rows:
                f.write(f"{r['measurement']}," + ",".join(f"{r[c]:.6f}" for c in cols) + "\n")
        print(f"per-measurement values -> {args.csv}")


if __name__ == "__main__":
    main()
