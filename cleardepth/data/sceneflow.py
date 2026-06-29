"""
Scene Flow Dataset
==================
Loads stereo image pairs and disparity maps from the Scene Flow dataset.

Scene Flow has three subsets:
  - FlyingThings3D : Random objects flying through a 3D scene
  - Driving        : Simulated driving sequences
  - Monkaa         : Furry creature animations

All three share the same file structure and PFM disparity format,
so one Dataset class handles all of them via a root path argument.

Expected directory layout:
  <root>/
    frames_cleanpass/TRAIN/A/0000/left/0006.png
    frames_cleanpass/TRAIN/A/0000/right/0006.png
    disparity/TRAIN/A/0000/left/0006.pfm

Paper reference: Section IV-A
  "We first pre-train on Scene Flow and CREStereo for 300K steps"
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple

from .transforms import StereoTransform


# ---------------------------------------------------------------------------
# PFM file reader
# ---------------------------------------------------------------------------

def read_pfm(path: str) -> np.ndarray:
    """
    Read a .pfm (Portable Float Map) file into a numpy array.

    PFM stores single-channel floating-point data — used for disparity
    and depth ground truth in many stereo datasets (Scene Flow, Middlebury).

    Args:
        path : Path to the .pfm file.

    Returns:
        data : 2D numpy array of shape (H, W), dtype float32.
               Disparity values are in pixels.
    """
    with open(path, 'rb') as f:
        # Header line 1: "PF" (colour) or "Pf" (grayscale)
        header = f.readline().rstrip()
        if header == b'PF':
            channels = 3
        elif header == b'Pf':
            channels = 1
        else:
            raise ValueError(f"Not a PFM file: {path}")

        # Header line 2: width height
        dims = f.readline().rstrip()
        W, H = map(int, dims.split())

        # Header line 3: scale (negative = little-endian)
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        scale  = abs(scale)

        # Read raw float data
        data = np.frombuffer(f.read(), dtype=np.dtype(f'{endian}f4'))
        data = data.reshape((H, W, channels) if channels == 3 else (H, W))

        # PFM is stored bottom-to-top — flip vertically
        data = np.flipud(data)

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SceneFlowDataset(Dataset):
    """
    Scene Flow stereo dataset loader.

    Args:
        root      : Path to the Scene Flow root directory.
        split     : 'train' or 'test'.
        height    : Crop height for training (ignored for test).
        width     : Crop width  for training (ignored for test).
        augment   : Whether to apply random augmentations.
        subsets   : Which Scene Flow subsets to include.
                    Default: all three ['A', 'B', 'C'] (FlyingThings3D pass dirs)
                    For Driving/Monkaa, pass their specific subset names.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        height: int = 360,
        width: int  = 720,
        augment: bool = True,
        subsets: List[str] = None,
    ):
        super().__init__()
        self.root    = root
        self.split   = split.upper()   # Scene Flow uses uppercase: TRAIN/TEST
        self.transform = StereoTransform(height=height, width=width,
                                         augment=augment)

        # Collect all (left_path, right_path, disp_path) triplets
        self.samples = self._collect_samples(subsets)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {root}. "
                f"Check that the Scene Flow dataset is correctly placed at "
                f"{root}/frames_cleanpass and {root}/disparity"
            )

    def _collect_samples(self, subsets) -> List[Tuple[str, str, str]]:
        """
        Walk the dataset directory structure and collect file triplets.
        Returns list of (left_img_path, right_img_path, left_disp_path).
        """
        samples = []

        img_root  = os.path.join(self.root, 'frames_cleanpass', self.split)
        disp_root = os.path.join(self.root, 'disparity', self.split)

        if not os.path.exists(img_root):
            return samples   # Dataset not present — return empty list

        # Scene Flow FlyingThings3D uses subdirectories A, B, C
        # Driving and Monkaa have different structures but same file format
        top_dirs = sorted(os.listdir(img_root)) if subsets is None else subsets

        for subset in top_dirs:
            subset_img  = os.path.join(img_root,  subset)
            subset_disp = os.path.join(disp_root, subset)

            if not os.path.isdir(subset_img):
                continue

            for seq in sorted(os.listdir(subset_img)):
                left_dir  = os.path.join(subset_img,  seq, 'left')
                right_dir = os.path.join(subset_img,  seq, 'right')
                disp_dir  = os.path.join(subset_disp, seq, 'left')

                if not (os.path.isdir(left_dir) and
                        os.path.isdir(right_dir) and
                        os.path.isdir(disp_dir)):
                    continue

                for fname in sorted(os.listdir(left_dir)):
                    if not fname.endswith('.png'):
                        continue
                    stem = os.path.splitext(fname)[0]

                    left_path  = os.path.join(left_dir,  fname)
                    right_path = os.path.join(right_dir, fname)
                    disp_path  = os.path.join(disp_dir,  stem + '.pfm')

                    if (os.path.exists(left_path) and
                        os.path.exists(right_path) and
                        os.path.exists(disp_path)):
                        samples.append((left_path, right_path, disp_path))

        return samples

    def _load_image(self, path: str) -> torch.Tensor:
        """Load PNG image → float32 tensor (3, H, W) in [0, 1]."""
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)

    def _load_disp(self, path: str) -> torch.Tensor:
        """Load PFM disparity → float32 tensor (1, H, W) in pixels."""
        arr = read_pfm(path)                             # (H, W)
        return torch.from_numpy(arr).unsqueeze(0)        # (1, H, W)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        left_path, right_path, disp_path = self.samples[idx]

        left  = self._load_image(left_path)
        right = self._load_image(right_path)
        disp  = self._load_disp(disp_path)

        # Apply augmentations and normalisation
        left, right, disp = self.transform(left, right, disp)

        return {
            'left'      : left,    # (3, H, W) in [-1, 1]
            'right'     : right,   # (3, H, W) in [-1, 1]
            'disparity' : disp,    # (1, H, W) in pixels
            'left_path' : left_path,   # For debugging
        }