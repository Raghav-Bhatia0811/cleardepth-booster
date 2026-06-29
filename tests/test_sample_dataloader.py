"""
tests/test_sample_dataloader.py
================================
Verifies the Scene Flow sample pack data loader works correctly.
Run with: pytest tests/test_sample_dataloader.py -v
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from cleardepth.data.sceneflow_sample import (
    SceneFlowSampleDataset,
    build_sample_loader,
    read_pfm,
)

# ── Change this if your sample pack is in a different location ──
SAMPLE_ROOT = "C:/Users/ragha/Downloads/Sampler/Sampler"
SUBSETS     = ["Monkaa", "FlyingThings3D"]


def test_pfm_reader():
    """PFM file is read into a valid float32 numpy array."""
    pfm_path = os.path.join(
        SAMPLE_ROOT, "Monkaa", "disparity", "0048.pfm"
    )
    disp = read_pfm(pfm_path)

    assert disp.ndim == 2,                   "Expected 2D array"
    assert str(disp.dtype) == 'float32',     "Expected float32"
    assert disp.min() >= 0,                  "Disparity should be non-negative"
    assert disp.max() < 10000,               "Suspiciously large max value"


def test_dataset_length():
    """Dataset finds exactly 6 samples (3 frames x 2 subsets)."""
    dataset = SceneFlowSampleDataset(
        root=SAMPLE_ROOT,
        subsets=SUBSETS,
    )
    assert len(dataset) == 6, \
        f"Expected 6 samples, got {len(dataset)}"


def test_sample_shapes():
    """Each sample returns tensors with correct shapes and value ranges."""
    dataset = SceneFlowSampleDataset(
        root=SAMPLE_ROOT,
        subsets=SUBSETS,
    )
    sample = dataset[0]

    left  = sample['left']
    right = sample['right']
    disp  = sample['disparity']

    # Must be tensors
    assert isinstance(left,  torch.Tensor), "left must be a tensor"
    assert isinstance(right, torch.Tensor), "right must be a tensor"
    assert isinstance(disp,  torch.Tensor), "disp must be a tensor"

    # Must be 3D: (C, H, W)
    assert left.ndim  == 3, "left must be 3D (C, H, W)"
    assert right.ndim == 3, "right must be 3D (C, H, W)"
    assert disp.ndim  == 3, "disp must be 3D (1, H, W)"

    # Correct number of channels
    assert left.shape[0]  == 3, "left must have 3 channels"
    assert right.shape[0] == 3, "right must have 3 channels"
    assert disp.shape[0]  == 1, "disp must have 1 channel"

    # Spatial sizes must all match
    assert left.shape[1:] == right.shape[1:], \
        f"left/right spatial mismatch: {left.shape} vs {right.shape}"
    assert left.shape[1:] == disp.shape[1:], \
        f"image/disp spatial mismatch: {left.shape} vs {disp.shape}"

    # Value ranges
    assert left.min()  >= 0.0 and left.max()  <= 1.0, "left not in [0, 1]"
    assert right.min() >= 0.0 and right.max() <= 1.0, "right not in [0, 1]"
    assert disp.min()  >= 0.0,                         "negative disparity found"


def test_all_samples_readable():
    """Every sample in the dataset loads without errors."""
    dataset = SceneFlowSampleDataset(
        root=SAMPLE_ROOT,
        subsets=SUBSETS,
    )
    for i in range(len(dataset)):
        sample = dataset[i]
        assert 'left'      in sample
        assert 'right'     in sample
        assert 'disparity' in sample
        assert 'left_path' in sample


def test_dataloader_batch():
    """DataLoader iterates all batches with correct shapes."""
    loader = build_sample_loader(
        root=SAMPLE_ROOT,
        subsets=SUBSETS,
        batch_size=1,
        shuffle=False,
    )

    assert len(loader) == 6, f"Expected 6 batches, got {len(loader)}"

    for batch in loader:
        left  = batch['left']       # (1, 3, H, W)
        right = batch['right']      # (1, 3, H, W)
        disp  = batch['disparity']  # (1, 1, H, W)

        assert left.ndim  == 4, "batched left should be 4D"
        assert right.ndim == 4, "batched right should be 4D"
        assert disp.ndim  == 4, "batched disp should be 4D"

        assert left.shape[0]  == 1, "batch size should be 1"
        assert left.shape[1]  == 3, "3 image channels"
        assert disp.shape[1]  == 1, "1 disparity channel"

        # Spatial dims must match across left, right, disp
        assert left.shape[2:] == right.shape[2:]
        assert left.shape[2:] == disp.shape[2:]