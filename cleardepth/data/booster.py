"""
Booster Stereo Dataset
======================
PyTorch Dataset for the Booster transparent-object stereo benchmark.

Directory layout (<root_dir>/train/balanced/):
    <SceneName>/camera_00/im0.png  im1.png  ...   (left images)
    <SceneName>/camera_02/im0.png  im1.png  ...   (right images, same filenames)
    <SceneName>/disp_00.npy                        (GT disparity, float32, H×W)
    <SceneName>/mask_00.png                        (valid-pixel mask, white=valid)

One sample = (scene, illumination_index).
All illuminations within a scene share a single disparity map.

Train / val split is done at scene level (scene_fraction held out for val),
so there is zero data leakage between splits.

Disparity rescaling:
    disp_00.npy is given for the original camera resolution (3008×4112).
    After spatial resize to (height, width) the disparity values must be
    scaled by width/original_width so they remain in units of output pixels:
        disp_out = disp_raw * (width / ORIGINAL_W)

Paper:  "Booster: a Benchmark for Depth from Images of Specular and
        Transparent Surfaces" (TPAMI 2024)
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

# ── Dataset constants ──────────────────────────────────────────────────────
ORIGINAL_H = 3008
ORIGINAL_W = 4112


class BoosterDataset(Dataset):
    """
    Booster stereo dataset.

    Args:
        root_dir     : Path to the Booster root, e.g. '/data/booster_gt'.
        split        : 'train' or 'val'.
        height       : Output image height (default 360).
        width        : Output image width  (default 720).
        augment      : If True, apply asymmetric colour jitter (train only).
        max_samples  : Optionally cap total samples (useful for debugging).
        val_fraction : Fraction of scenes held out for validation (default 0.15).
        seed         : RNG seed for reproducible scene-level split.

    Returns (per sample, dict):
        left      : (3, H, W) float32 tensor normalised to [-1, 1]
        right     : (3, H, W) float32 tensor normalised to [-1, 1]
        disparity : (1, H, W) float32 disparity in output-pixel units
        mask      : (1, H, W) float32 binary mask (1.0 = valid pixel)
        scene     : scene name string
        illum     : illumination image stem string (e.g. 'im0')
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

        balanced_dir = self.root_dir / 'train' / 'balanced'
        if not balanced_dir.exists():
            raise FileNotFoundError(
                f"Booster balanced split not found: {balanced_dir}\n"
                f"Expected structure: <root_dir>/train/balanced/<SceneName>/..."
            )

        # ── Scene-level train/val split ────────────────────────────────────
        all_scenes = sorted(d.name for d in balanced_dir.iterdir() if d.is_dir())
        if not all_scenes:
            raise FileNotFoundError(f"No scenes found under {balanced_dir}")

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
            scene_dir  = balanced_dir / scene
            left_dir   = scene_dir / 'camera_00'
            right_dir  = scene_dir / 'camera_02'
            disp_path  = scene_dir / 'disp_00.npy'
            mask_path  = scene_dir / 'mask_00.png'

            if not (left_dir.exists() and right_dir.exists()):
                continue
            if not disp_path.exists():
                continue

            # Intersect left and right filenames to avoid mismatched pairs
            left_names  = {p.name for p in left_dir.glob('*.png')}
            right_names = {p.name for p in right_dir.glob('*.png')}
            common = sorted(left_names & right_names)

            for fname in common:
                self.samples.append({
                    'left' : left_dir  / fname,
                    'right': right_dir / fname,
                    'disp' : disp_path,
                    'mask' : mask_path if mask_path.exists() else None,
                    'scene': scene,
                    'illum': Path(fname).stem,
                })

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}' in {balanced_dir}"
            )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # ── Per-scene caches (avoid repeated resize of large .npy files) ──
        # Memory cost: 38 scenes × 360×720 × 4 bytes ≈ 40 MB total.
        # With num_workers > 0 each worker has its own copy; still cheap.
        self._disp_cache: Dict[str, torch.Tensor] = {}
        self._mask_cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_disp(self, disp_path) -> torch.Tensor:
        """Return resized disparity (1, H, W), caching by file path."""
        key = str(disp_path)
        if key not in self._disp_cache:
            raw = np.load(disp_path).astype(np.float32)
            # Replace non-finite values (inf / nan from occlusion) with 0
            raw = np.where(np.isfinite(raw), raw, 0.0)
            t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            resized = F.interpolate(
                t, size=(self.height, self.width), mode='nearest'
            )
            # Scale disparity values to output-pixel units
            resized = resized * (self.width / ORIGINAL_W)
            self._disp_cache[key] = resized.squeeze(0)  # (1, H, W)
        return self._disp_cache[key]

    def _load_mask(self, mask_path) -> torch.Tensor:
        """Return resized binary mask (1, H, W), caching by file path."""
        key = str(mask_path)
        if key not in self._mask_cache:
            img = Image.open(mask_path).convert('L')
            arr = np.array(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            resized = F.interpolate(
                t, size=(self.height, self.width), mode='nearest'
            )
            self._mask_cache[key] = (resized.squeeze(0) > 0.5).float()  # (1,H,W)
        return self._mask_cache[key]

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

        Left and right each get independently sampled jitter parameters
        50% of the time; otherwise they share the same parameters (symmetric).
        This teaches the network to handle photometric variation across views,
        which is common in real stereo rigs with imperfect radiometric sync.
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
        # 50%: each view gets independent jitter; 50%: both share same jitter
        p_right = sample_params() if random.random() < 0.5 else p_left
        return apply(left, p_left), apply(right, p_right)

    # ── __getitem__ ────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        left  = self._load_image(s['left'])
        right = self._load_image(s['right'])
        disp  = self._load_disp(s['disp'])
        mask  = self._load_mask(s['mask']) if s['mask'] is not None \
                else torch.ones(1, self.height, self.width)

        if self.augment:
            left, right = self._color_jitter(left, right)

        # Normalise images from [0, 1] → [-1, 1]
        left  = left  * 2.0 - 1.0
        right = right * 2.0 - 1.0

        return {
            'left'     : left,
            'right'    : right,
            'disparity': disp,
            'mask'     : mask,
            'scene'    : s['scene'],
            'illum'    : s['illum'],
        }

    # ── Utility ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_scenes = len({s['scene'] for s in self.samples})
        return (
            f"BoosterDataset(split={self.split!r}, scenes={n_scenes}, "
            f"samples={len(self.samples)}, size={self.height}×{self.width})"
        )


# ── Quick self-test ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='Path to Booster root, e.g. /data/booster_gt')
    args = parser.parse_args()

    for split in ('train', 'val'):
        ds = BoosterDataset(args.root, split=split, max_samples=5)
        print(ds)
        sample = ds[0]
        print(f"  left:      {list(sample['left'].shape)}")
        print(f"  right:     {list(sample['right'].shape)}")
        print(f"  disparity: {list(sample['disparity'].shape)}  "
              f"range=[{sample['disparity'].min():.2f}, "
              f"{sample['disparity'].max():.2f}]")
        print(f"  mask:      {list(sample['mask'].shape)}  "
              f"valid={sample['mask'].mean():.2%}")
        print(f"  scene={sample['scene']}  illum={sample['illum']}")
