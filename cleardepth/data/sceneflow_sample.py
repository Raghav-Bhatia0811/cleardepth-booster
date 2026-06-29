"""
Scene Flow Sample Pack Dataset
================================
Loads from the small sample pack downloaded from:
https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

Sample pack structure (different from full Scene Flow):
  <root>/
    Monkaa/
      RGB_cleanpass/
        left/   0048.png, 0049.png, 0050.png
        right/  0048.png, 0049.png, 0050.png
      disparity/
        0048.pfm, 0049.pfm, 0050.pfm
    FlyingThings3D/
      RGB_cleanpass/
        left/   0006.png, 0007.png, 0008.png
        right/  0006.png, 0007.png, 0008.png
      disparity/
        0006.pfm, 0007.pfm, 0008.pfm

Note: disparity files sit directly in disparity/ (no left/right split).
      File names match between RGB and disparity folders.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List


# ---------------------------------------------------------------------------
# PFM reader (same format as full Scene Flow)
# ---------------------------------------------------------------------------

def read_pfm(filepath: str) -> np.ndarray:
    """
    Read a .pfm (Portable Float Map) file into a float32 numpy array.

    Args:
        filepath: path to the .pfm file

    Returns:
        2D numpy array of shape (H, W), dtype float32
        Values are disparity in pixels.
    """
    with open(filepath, 'rb') as f:
        # Line 1: tag — must be 'Pf' for grayscale
        tag = f.readline().decode('utf-8').strip()
        assert tag == 'Pf', f"Expected grayscale PFM ('Pf'), got '{tag}'"

        # Line 2: width height
        dims = f.readline().decode('utf-8').strip().split()
        width, height = int(dims[0]), int(dims[1])

        # Line 3: scale (negative = little-endian)
        scale = float(f.readline().decode('utf-8').strip())
        endian = '<' if scale < 0 else '>'

        # Raw float data
        data = np.frombuffer(f.read(), dtype=np.dtype(f'{endian}f4'))
        data = data.reshape((height, width))

        # PFM stored bottom-to-top — flip vertically
        data = np.flipud(data)

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SceneFlowSampleDataset(Dataset):
    """
    Dataset loader for the Scene Flow sample pack.

    This handles the flat structure of the sample pack where
    disparity files sit directly in disparity/ alongside the
    RGB_cleanpass/left and RGB_cleanpass/right folders.

    Args:
        root      : Path to the Sampler folder containing subset folders
                    e.g. "C:/Users/ragha/Downloads/Sampler/Sampler"
        subsets   : List of subset folder names to include
                    e.g. ["Monkaa", "FlyingThings3D"]
        pass_name : Image subfolder name, default "RGB_cleanpass"
    """

    def __init__(
        self,
        root: str,
        subsets: List[str],
        pass_name: str = 'RGB_cleanpass',
    ):
        self.samples = []  # list of (left_path, right_path, disp_path)

        for subset in subsets:
            subset_dir = os.path.join(root, subset)
            left_dir   = os.path.join(subset_dir, pass_name, 'left')
            right_dir  = os.path.join(subset_dir, pass_name, 'right')
            disp_dir   = os.path.join(subset_dir, 'disparity')

            # Verify all three directories exist
            for d, label in [(left_dir, 'left images'),
                              (right_dir, 'right images'),
                              (disp_dir, 'disparity')]:
                if not os.path.isdir(d):
                    raise RuntimeError(
                        f"[SceneFlowSample] {label} directory not found:\n"
                        f"  {d}\n"
                        f"Check that root='{root}' and subset='{subset}' "
                        f"are correct."
                    )

            # Collect all left PNGs, sorted by frame number
            left_files = sorted(
                [f for f in os.listdir(left_dir) if f.endswith('.png')],
                key=lambda x: int(re.sub(r'\D', '', x))
            )

            for fname in left_files:
                frame_id   = os.path.splitext(fname)[0]  # e.g. "0048"
                left_path  = os.path.join(left_dir,  f"{frame_id}.png")
                right_path = os.path.join(right_dir, f"{frame_id}.png")
                disp_path  = os.path.join(disp_dir,  f"{frame_id}.pfm")

                # Only add triplet if all three files exist
                if (os.path.isfile(right_path) and
                        os.path.isfile(disp_path)):
                    self.samples.append((left_path, right_path, disp_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[SceneFlowSample] No samples found!\n"
                f"  root    = {root}\n"
                f"  subsets = {subsets}\n"
                f"Double-check your paths."
            )

        print(f"[SceneFlowSample] Found {len(self.samples)} samples "
              f"from: {subsets}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        left_path, right_path, disp_path = self.samples[idx]

        left  = self._load_image(left_path)    # (3, H, W)
        right = self._load_image(right_path)   # (3, H, W)
        disp  = self._load_disp(disp_path)     # (1, H, W)

        return {
            'left':      left,
            'right':     right,
            'disparity': disp,
            'left_path': left_path,  # useful for debugging
        }

    def _load_image(self, path: str) -> torch.Tensor:
        """Load PNG → float32 tensor (3, H, W) in [0, 1]."""
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _load_disp(self, path: str) -> torch.Tensor:
        """Load PFM → float32 tensor (1, H, W) in pixels."""
        arr = read_pfm(path)                            # (H, W)
        return torch.from_numpy(arr).unsqueeze(0)       # (1, H, W)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_sample_loader(
    root: str,
    subsets: List[str],
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a DataLoader from the Scene Flow sample pack.

    Args:
        root        : Path to the Sampler folder
        subsets     : List of subset names e.g. ["Monkaa", "FlyingThings3D"]
        batch_size  : Batch size (keep 1 for sample pack)
        num_workers : Keep 0 on Windows to avoid multiprocessing issues
        shuffle     : Whether to shuffle samples

    Returns:
        DataLoader ready to iterate
    """
    dataset = SceneFlowSampleDataset(
        root=root,
        subsets=subsets,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
    )