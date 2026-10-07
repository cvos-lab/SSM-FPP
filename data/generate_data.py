"""Build the tensors used for training and evaluation.

    python data/generate_data.py --root /path/to/depth_dataset --out data/data_pt

Expected raw layout (one measurement = 30 camera frames + one depth label):

    <root>/left_cam/images/*.png        NUM_SAMPLES x 30 frames, sorted by name
    <root>/left_cam/depth_labels/*.txt  NUM_SAMPLES files, 512 x 768 values
                                        (label units = mm x 3.1954; -1000 = no return)

Outputs in --out:
    fringe_train.pt, Z_train.pt   3000 measurements (training + validation)
    fringe_test.pt,  Z_test.pt    69 held-out measurements
    ref_fringe.pt, ref_bg.pt      reference fringe and background (sample 2306)

The captured fringe image I_f is stored as recorded (8-bit -> [-1, 1]); it is
NOT modified using the depth labels.
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMG_H, IMG_W = 512, 768
NUM_SAMPLES = 3070
FRAMES_PER_SAMPLE = 30
FRAME_INDEX = 8           # frame used as the fringe input I_f
BG_FRAME_INDEX = 28       # uniformly illuminated frame of the reference sample
REF_SAMPLE_INDEX = 2306   # reference sample: supplies I_ref and B_ref, used in no split

NUM_TEST = 69
NUM_TRAIN = 3000
RANDOM_SEED = 42

Z_MIN = -200.0 * 3.1954   # label units
Z_MAX = 150.0 * 3.1954


def read_z(path):
    with open(path, "r") as f:
        data = np.array(f.read().split())
    return data.reshape(IMG_H, IMG_W).astype(float)


def read_frame(path):
    """8-bit image -> float in [-1, 1]."""
    return np.array(Image.open(path)).astype(np.float32) / 127.5 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="raw dataset directory (contains left_cam/)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "data_pt"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z_paths = sorted(glob.glob(os.path.join(args.root, "left_cam", "depth_labels", "*.txt")))
    fringe_paths = sorted(glob.glob(os.path.join(args.root, "left_cam", "images", "*.png")))
    assert len(z_paths) == NUM_SAMPLES, f"expected {NUM_SAMPLES} depth files, found {len(z_paths)}"
    assert len(fringe_paths) >= NUM_SAMPLES * FRAMES_PER_SAMPLE, \
        f"expected {NUM_SAMPLES * FRAMES_PER_SAMPLE} images, found {len(fringe_paths)}"

    # reference fringe and background, captured once (sample 2306)
    base = REF_SAMPLE_INDEX * FRAMES_PER_SAMPLE
    ref_f = torch.from_numpy(read_frame(fringe_paths[base + FRAME_INDEX]))[None, None]
    ref_bg = torch.from_numpy(read_frame(fringe_paths[base + BG_FRAME_INDEX]))[None, None]
    torch.save(ref_f, os.path.join(args.out, "ref_fringe.pt"))
    torch.save(ref_bg, os.path.join(args.out, "ref_bg.pt"))

    fringe = torch.zeros((NUM_SAMPLES, 1, IMG_H, IMG_W), dtype=torch.float32)
    depth = torch.zeros((NUM_SAMPLES, 1, IMG_H, IMG_W), dtype=torch.float32)
    for i in tqdm(range(NUM_SAMPLES), desc="loading"):
        z = read_z(z_paths[i])
        z[z == -1000.0] = Z_MIN                                  # no return -> -200 mm
        depth[i, 0] = torch.from_numpy((z - Z_MIN) / (Z_MAX - Z_MIN)).float()
        fringe[i, 0] = torch.from_numpy(
            read_frame(fringe_paths[i * FRAMES_PER_SAMPLE + FRAME_INDEX]))

    # seeded split at measurement level; the reference sample is excluded
    available = [i for i in range(NUM_SAMPLES) if i != REF_SAMPLE_INDEX]
    assert len(available) == NUM_TRAIN + NUM_TEST
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = np.array(available)
    rng.shuffle(shuffled)
    test_idx = sorted(shuffled[:NUM_TEST].tolist())
    train_idx = sorted(shuffled[NUM_TEST:].tolist())

    for name, idx in (("train", train_idx), ("test", test_idx)):
        torch.save({"data": fringe[idx], "original_indices": idx},
                   os.path.join(args.out, f"fringe_{name}.pt"))
        torch.save({"data": depth[idx], "original_indices": idx},
                   os.path.join(args.out, f"Z_{name}.pt"))
        print(f"{name}: {len(idx)} measurements")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
