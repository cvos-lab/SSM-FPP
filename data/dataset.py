import torch
import numpy as np
from torch.utils.data import Dataset


# =========================================================================
# Hardcoded camera intrinsics for the left camera (calibrated values).
# =========================================================================
FX = 2.7674498064097711e+03
FY = 2.7675590254576309e+03
CX = 3.8258601201085355e+02
CY = 2.0311869177718791e+02
# =========================================================================


class PrecomputedFringeDepthDataset(Dataset):
    """
    Fringe-to-depth dataset supporting two input modes:

    input_mode = "6ch":   [I, REF_F, Diff, REF_BG, u_norm, v_norm]
        Six-channel geometric input. Channels:
            I        : single-frame fringe input              (range ~[-1, 1])
            REF_F    : reference fringe from sample 2306      (range ~[-1, 1])
            Diff     : (I - REF_F) / 2                        (range ~[-1, 1])
            REF_BG   : reference background from sample 2306  (range ~[-1, 1])
            u_norm   : (u_pixel - cx) / fx, broadcast (H, W)  (range ~[-0.14, +0.14])
            v_norm   : (v_pixel - cy) / fy, broadcast (H, W)  (range ~[-0.07, +0.11])

    input_mode = "1ch":   [I]
        Single-channel fringe input (raw fringe only).

    Args:
        fringe_path:      path to fringe .pt (dict with "data"/"original_indices" or raw tensor)
        Z_path:           path to depth  .pt (same format)
        ref_fringe_path:  path to ref_fringe.pt (1, 1, H, W). Required for "6ch", ignored for "1ch".
        ref_bg_path:      path to ref_bg.pt     (1, 1, H, W). Required for "6ch", ignored for "1ch".
        input_mode:       "6ch" or "1ch"
    """

    def __init__(self, fringe_path, Z_path,
                 ref_fringe_path=None, ref_bg_path=None,
                 input_mode="6ch"):
        assert input_mode in ("6ch", "1ch"), f"Unknown input_mode: {input_mode}"
        self.input_mode = input_mode

        # --- fringe (always loaded) ---
        raw_fringe = torch.load(fringe_path, weights_only=False)
        if isinstance(raw_fringe, dict):
            self.fringe = raw_fringe["data"]
            self.original_indices = raw_fringe.get("original_indices", None)
        else:
            self.fringe = raw_fringe
            self.original_indices = None

        # --- depth (always loaded) ---
        raw_Z = torch.load(Z_path, weights_only=False)
        self.Z = raw_Z["data"] if isinstance(raw_Z, dict) else raw_Z

        # --- shape sanity ---
        assert self.fringe.ndim == 4, f"fringe expected 4D (N,1,H,W), got {self.fringe.shape}"
        _, _, H, W = self.fringe.shape

        # --- references and u/v: ONLY needed for 6-channel mode ---
        if self.input_mode == "6ch":
            assert ref_fringe_path is not None and ref_bg_path is not None, (
                "input_mode='6ch' requires both ref_fringe_path and ref_bg_path."
            )

            raw_ref_f = torch.load(ref_fringe_path, weights_only=False)
            ref_f = raw_ref_f["data"] if isinstance(raw_ref_f, dict) else raw_ref_f
            # Saved shape (1, 1, H, W) -> stored as (1, H, W)
            self.ref_fringe = ref_f.squeeze(0) if ref_f.ndim == 4 else ref_f

            raw_ref_bg = torch.load(ref_bg_path, weights_only=False)
            ref_bg = raw_ref_bg["data"] if isinstance(raw_ref_bg, dict) else raw_ref_bg
            self.ref_bg = ref_bg.squeeze(0) if ref_bg.ndim == 4 else ref_bg

            # u, v normalized camera coordinate maps (hardcoded intrinsics)
            us = np.arange(W, dtype=np.float32)
            vs = np.arange(H, dtype=np.float32)
            u_grid = (us - CX) / FX
            v_grid = (vs - CY) / FY
            u_map = np.tile(u_grid[None, :], (H, 1))
            v_map = np.tile(v_grid[:, None], (1, W))
            self.u_tensor = torch.from_numpy(u_map).unsqueeze(0)  # (1, H, W)
            self.v_tensor = torch.from_numpy(v_map).unsqueeze(0)  # (1, H, W)
        else:
            self.ref_fringe = None
            self.ref_bg = None
            self.u_tensor = None
            self.v_tensor = None

    def __len__(self):
        return self.fringe.shape[0]

    def __getitem__(self, idx):
        I = self.fringe[idx]              # (1, H, W)
        target_Z = self.Z[idx]            # (1, H, W)

        if self.input_mode == "1ch":
            return {"fringe": I, "Z": target_Z}

        # input_mode == "6ch"
        REF_F = self.ref_fringe           # (1, H, W)
        REF_BG = self.ref_bg              # (1, H, W)
        Diff = (I - REF_F) / 2.0          # (1, H, W)
        combined = torch.cat([I, REF_F, Diff, REF_BG, self.u_tensor, self.v_tensor], dim=0)
        return {"fringe": combined, "Z": target_Z}

    def get_original_index(self, idx):
        return self.original_indices[idx] if self.original_indices is not None else idx