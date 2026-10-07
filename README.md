# SSM-FPP

Code for single-shot fringe projection profilometry (FPP): a network maps one
captured fringe image to a dense depth map. The repository contains SSM-FPP,
a multiscale encoder-decoder whose blocks use a selective state-space model
(Mamba) as the context mixer, and the three baselines it is compared with
(HiDNet, NAFNet, Restormer), all trained and evaluated under one protocol.

## Requirements

Python 3.11, PyTorch 2.6 with CUDA 12.4, and an NVIDIA GPU (mamba-ssm needs CUDA).

```bash
pip install -r requirements.txt
```

## Data

Each measurement consists of 30 camera frames and one depth map from multi-shot
phase-shifting FPP. Expected layout:

```
<root>/left_cam/images/*.png          30 frames per measurement, sorted by name
<root>/left_cam/depth_labels/*.txt    one 512 x 768 depth map per measurement
```

Build the tensors used for training and evaluation:

```bash
python data/generate_data.py --root /path/to/depth_dataset --out data/data_pt
```

This writes 3000 measurements for training and validation, 69 held-out
measurements for testing, and the reference images. The split is fixed (seed 42).

### Network input

Six channels, each 512 x 768, assembled in `data/dataset.py`:

| Channel | Content |
|---|---|
| I_f | captured fringe image (frame 8 of the measurement) |
| I_ref | reference fringe image on a flat plane (frame 8 of sample 2306) |
| (I_f - I_ref)/2 | difference channel, recomputed for every measurement |
| B_ref | reference background, uniform illumination (frame 28 of sample 2306) |
| u, v | normalised camera coordinates, (x - c_x)/f_x and (y - c_y)/f_y |

Only I_f is acquired per measurement; I_ref, B_ref, u and v are fixed for the
measurement system. Images are scaled to [-1, 1]. Configurations with
`input_mode: "1ch"` use I_f alone.

## Training

```bash
python train.py --config configs/ssm_fpp.yaml --seed 42
```

All visible GPUs are used (one sample per GPU). Every configuration uses the
same objective (0.75 MSE + 0.25 (1 - SSIM) on normalised depth), Adam with
learning rate 1e-4, cosine annealing to 1e-6 over 400 epochs, and keeps the
checkpoint with the lowest validation loss. `--seed` changes initialisation,
dropout and batch order; the training/validation split does not change.

| Config | Network |
|---|---|
| `configs/ssm_fpp.yaml` | SSM-FPP |
| `configs/hidnet.yaml` | HiDNet |
| `configs/nafnet.yaml` | NAFNet (SIDD width-32 configuration) |
| `configs/restormer.yaml` | Restormer |
| `configs/ablation/mixer_*.yaml` | SSM-FPP with the context mixer replaced by identity, expanded depthwise convolution or MDTA |
| `configs/ablation/input_1ch_*.yaml` | SSM-FPP and HiDNet with I_f as the only input |

## Evaluation

```bash
python evaluate.py --config configs/ssm_fpp.yaml --ckpt checkpoints/ssm_fpp_seed42.pth
```

Metrics are computed on a foreground mask derived from the reference depth
(no-return pixels and the dominant background plane removed, then a 2-pixel
erosion), per measurement, and averaged over the 69 held-out measurements:
MAE and RMSE (mm), AbsRel, gradient error GE (mm/pixel), Laplacian roughness
(mm/pixel^2), and the percentage of pixels within 0.5, 1 and 2 mm.
`--csv` writes per-measurement values.

## Computational cost

```bash
python tools/measure_efficiency.py   # parameters, latency, peak GPU memory
python tools/count_flops.py          # FLOPs per 768 x 512 input
```

Latency and memory depend on the GPU and software versions. For the Mamba
layers, whose fused kernels standard FLOP counters do not cover, FLOPs are
counted analytically (see `tools/count_flops.py`).

## Repository layout

```
models/        ssm_fpp.py, hidnet.py, nafnet.py, restormer.py, build_model()
data/          generate_data.py, dataset.py
configs/       network and ablation configurations
train.py       distributed training
evaluate.py    held-out evaluation
tools/         efficiency and FLOP measurement
```

## Citation

If you use this code, please cite the accompanying paper; the reference will be
added here on publication.

## License

MIT, see [LICENSE](LICENSE).
