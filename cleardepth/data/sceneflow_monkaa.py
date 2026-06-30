"""
Scene Flow — Monkaa Dataset
============================
PyTorch Dataset for the Monkaa subset of Scene Flow, used to pretrain
ClearDepthNet before fine-tuning on Booster.

Directory layout (<root_dir>/):
    frames_cleanpass/<scene>/left/0000.png   0001.png   ...
    frames_cleanpass/<scene>/right/0000.png  0001.png   ...
    disparity/<scene>/left/0000.pfm          0001.pfm   ...
    disparity/<scene>/right/...              (unused — left-referenced only)

Frame <scene>/left/000N.png pairs with <scene>/right/000N.png and
disparity/<scene>/left/000N.pfm. Only the left (reference-camera) disparity
is used, matching Booster's disp_00.npy convention.

No official train/val split — held out at the *scene* level (not frame
level) to avoid leakage between near-duplicate consecutive frames.

Resizing: 540×960 (native Monkaa resolution) → 360×720, matching Booster's
target resolution so a pretrained checkpoint transfers directly via
--resume in train_booster.py.

Disparity rescaling: disp_out = disp_raw * (width / ORIGINAL_W), same
convention as BoosterDataset.

PFM format reference: http://www.pauldebevec.com/Research/HDR/PFM/
"""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

# ── Dataset constants ──────────────────────────────────────────────────────
ORIGINAL_H = 540
ORIGINAL_W = 960


# ── PFM reader ──────────────────────────────────────────────────────────────

def read_pfm(path) -> np.ndarray:
    """
    Read a Portable Float Map (.pfm) file.

    Header format:
        line 1 : 'PF' (color, 3 channels) or 'Pf' (grayscale, 1 channel)
        line 2 : '<width> <height>'
        line 3 : '<scale>'   negative = little-endian, positive = big-endian
        then   : raw float32 data, stored bottom-to-top, left-to-right

    Returns:
        (H, W) float32 array for grayscale, or (H, W, 3) for color.
        Always returned top-to-bottom (row 0 = top of image), matching
        standard image array conventions.
    """
    with open(path, 'rb') as f:
        header = f.readline().decode('latin-1').rstrip()
        if header == 'PF':
            channels = 3
        elif header == 'Pf':
            channels = 1
        else:
            raise ValueError(f"Not a valid PFM file: {path} (header={header!r})")

        # Dimension line — skip blank/comment lines defensively
        dim_line = f.readline().decode('latin-1').rstrip()
        while dim_line == '':
            dim_line = f.readline().decode('latin-1').rstrip()
        dims = re.match(r'^(\d+)\s+(\d+)\s*$', dim_line)
        if not dims:
            raise ValueError(f"Malformed PFM dimension line in {path}: {dim_line!r}")
        width, height = int(dims.group(1)), int(dims.group(2))

        scale = float(f.readline().decode('latin-1').rstrip())
        endian = '<' if scale < 0 else '>'   # negative scale = little-endian

        data = np.fromfile(f, dtype=endian + 'f4')
        expected = width * height * channels
        if data.size != expected:
            raise ValueError(
                f"PFM data size mismatch in {path}: "
                f"got {data.size}, expected {expected}"
            )

        if channels == 3:
            data = data.reshape(height, width, 3)
        else:
            data = data.reshape(height, width)

        # PFM stores rows bottom-to-top — flip to standard top-to-bottom
        data = np.flipud(data)

        return np.ascontiguousarray(data)


# ── Dataset ──────────────────────────────────────────────────────────────────

class SceneFlowMonkaaDataset(Dataset):
    """
    Monkaa subset of Scene Flow, resized to match Booster's training
    resolution for compatible pretrain → fine-tune transfer.

    Args:
        root_dir     : Path to Monkaa root, e.g. '/data/monkaa'.
                       Expects 'frames_cleanpass/' and 'disparity/' subdirs.
        split        : 'train' or 'val'.
        height       : Output image height (default 360).
        width        : Output image width  (default 720).
        augment      : If True, apply asymmetric colour jitter (train only).
        max_samples  : Optionally cap total samples (debugging).
        val_fraction : Fraction of scenes held out for validation (default 0.15).
        seed         : RNG seed for reproducible scene-level split.

    Returns (per sample, dict):
        left      : (3, H, W) float32 tensor normalised to [-1, 1]
        right     : (3, H, W) float32 tensor normalised to [-1, 1]
        disparity : (1, H, W) float32 disparity in output-pixel units
        scene     : scene name string
        frame     : frame stem string (e.g. '0000')
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        height: int = 360,
        width: int = 720,
        augment: bool = True,
        max_samples: Optional[int] = None,
        val_fraction: float = 0.15,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.split    = split
        self.height   = height
        self.width    = width
        self.augment  = augment and (split == 'train')

        frames_dir = self.root_dir / 'frames_cleanpass'
        disp_dir   = self.root_dir / 'disparity'
        if not frames_dir.exists():
            raise FileNotFoundError(
                f"Monkaa frames_cleanpass not found: {frames_dir}\n"
                f"Expected structure: <root_dir>/frames_cleanpass/<scene>/left/*.png"
            )
        if not disp_dir.exists():
            raise FileNotFoundError(
                f"Monkaa disparity not found: {disp_dir}\n"
                f"Expected structure: <root_dir>/disparity/<scene>/left/*.pfm"
            )

        # ── Scene-level train/val split ────────────────────────────────────
        all_scenes = sorted(d.name for d in frames_dir.iterdir() if d.is_dir())
        if not all_scenes:
            raise FileNotFoundError(f"No scenes found under {frames_dir}")

        rng = random.Random(seed)
        scenes_shuffled = all_scenes.copy()
        rng.shuffle(scenes_shuffled)
        n_val = max(1, int(len(scenes_shuffled) * val_fraction))

        if split == 'val':
            active_scenes = scenes_shuffled[:n_val]
        else:
            active_scenes = scenes_shuffled[n_val:]

        # ── Build flat sample list ─────────────────────────────────────────
        self.samples: List[Dict] = []
        for scene in sorted(active_scenes):
            left_dir  = frames_dir / scene / 'left'
            right_dir = frames_dir / scene / 'right'
            disp_left_dir = disp_dir / scene / 'left'

            if not (left_dir.exists() and right_dir.exists() and disp_left_dir.exists()):
                continue

            left_stems  = {p.stem for p in left_dir.glob('*.png')}
            right_stems = {p.stem for p in right_dir.glob('*.png')}
            disp_stems  = {p.stem for p in disp_left_dir.glob('*.pfm')}
            common = sorted(left_stems & right_stems & disp_stems)

            for stem in common:
                self.samples.append({
                    'left' : left_dir / f'{stem}.png',
                    'right': right_dir / f'{stem}.png',
                    'disp' : disp_left_dir / f'{stem}.pfm',
                    'scene': scene,
                    'frame': stem,
                })

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}' in {frames_dir}"
            )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_disp(self, disp_path) -> torch.Tensor:
        """Read PFM disparity, sanitize, resize, rescale → (1, H, W)."""
        raw = read_pfm(disp_path).astype(np.float32)   # (ORIGINAL_H, ORIGINAL_W)

        # Sanitize: clip negative / inf / nan to 0 (invalid/occluded pixels)
        raw = np.where(np.isfinite(raw), raw, 0.0)
        raw = np.clip(raw, 0.0, None)

        t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        resized = F.interpolate(
            t, size=(self.height, self.width), mode='nearest'
        )
        # Scale disparity values to output-pixel units
        resized = resized * (self.width / ORIGINAL_W)
        return resized.squeeze(0)   # (1, H, W)

    def _load_image(self, path) -> torch.Tensor:
        """Load PNG as float [0,1] tensor (3, H, W) at target resolution."""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.width, self.height), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _color_jitter(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Asymmetric colour jitter for stereo training.
        Same logic as BoosterDataset — left and right each get independently
        sampled jitter parameters 50% of the time; otherwise shared.
        """
        def sample_params() -> tuple:
            return (
                random.uniform(0.6, 1.4),   # brightness
                random.uniform(0.6, 1.4),   # contrast
                random.uniform(0.6, 1.4),   # saturation
                random.uniform(-0.1, 0.1),  # hue
            )

        def apply(img: torch.Tensor, p: tuple) -> torch.Tensor:
            b, c, s, h = p
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            img = TF.adjust_hue(img, h)
            return img.clamp(0.0, 1.0)

        p_left = sample_params()
        p_right = sample_params() if random.random() < 0.5 else p_left
        return apply(left, p_left), apply(right, p_right)

    # ── __getitem__ ────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        left  = self._load_image(s['left'])
        right = self._load_image(s['right'])
        disp  = self._load_disp(s['disp'])

        if self.augment:
            left, right = self._color_jitter(left, right)

        # Normalise images from [0, 1] → [-1, 1]
        left  = left  * 2.0 - 1.0
        right = right * 2.0 - 1.0

        return {
            'left'     : left,
            'right'    : right,
            'disparity': disp,
            'scene'    : s['scene'],
            'frame'    : s['frame'],
        }

    # ── Utility ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_scenes = len({s['scene'] for s in self.samples})
        return (
            f"SceneFlowMonkaaDataset(split={self.split!r}, scenes={n_scenes}, "
            f"samples={len(self.samples)}, size={self.height}×{self.width})"
        )


# ── Quick self-test ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='Path to Monkaa root, e.g. /data/monkaa')
    args = parser.parse_args()

    for split in ('train', 'val'):
        ds = SceneFlowMonkaaDataset(args.root, split=split, max_samples=5)
        print(ds)
        sample = ds[0]
        print(f"  left:      {list(sample['left'].shape)}")
        print(f"  right:     {list(sample['right'].shape)}")
        print(f"  disparity: {list(sample['disparity'].shape)}  "
              f"range=[{sample['disparity'].min():.2f}, "
              f"{sample['disparity'].max():.2f}]")
        print(f"  scene={sample['scene']}  frame={sample['frame']}")
