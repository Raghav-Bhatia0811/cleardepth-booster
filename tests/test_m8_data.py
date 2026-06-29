"""
Milestone 8 Smoke Test — Data Pipeline
=======================================
Tests transforms and dataset utilities WITHOUT requiring the actual
Scene Flow dataset on disk. Uses synthetic tensors to verify the pipeline.

Run with: pytest tests/test_m8_data.py -v
"""

import torch
import numpy as np
import pytest
import os
import tempfile

from cleardepth.data.transforms import (
    RandomCrop, RandomHorizontalFlip,
    AsymmetricColorJitter, Normalise, StereoTransform
)
from cleardepth.data.sceneflow import read_pfm, SceneFlowDataset


def make_stereo(H=128, W=256):
    """Helper: make random stereo pair + disparity."""
    left  = torch.rand(3, H, W)
    right = torch.rand(3, H, W)
    disp  = torch.rand(1, H, W) * 100 + 1.0
    return left, right, disp


# ===========================================================================
# Test Group 1: Transforms
# ===========================================================================

class TestRandomCrop:

    def test_output_size(self):
        left, right, disp = make_stereo(H=128, W=256)
        crop = RandomCrop(height=64, width=128)
        l, r, d = crop(left, right, disp)
        assert l.shape == (3, 64, 128)
        assert r.shape == (3, 64, 128)
        assert d.shape == (1, 64, 128)

    def test_all_three_same_crop(self):
        """Left, right and disparity must receive the same crop region."""
        # Fill each image with its index value so we can verify alignment
        left  = torch.zeros(3, 64, 128)
        right = torch.ones(3, 64, 128)
        disp  = torch.full((1, 64, 128), 2.0)

        crop = RandomCrop(height=32, width=64)
        l, r, d = crop(left, right, disp)
        # Values must still match their originals (just a crop, no mixing)
        assert l.max() == 0.0
        assert r.min() == 1.0
        assert d.mean() == pytest.approx(2.0)


class TestRandomHorizontalFlip:

    def test_flip_swaps_images(self):
        """After a flip, left should become the flipped right and vice versa."""
        left  = torch.rand(3, 32, 64)
        right = torch.rand(3, 32, 64)
        disp  = torch.rand(1, 32, 64) * 10

        flip = RandomHorizontalFlip(p=1.0)   # Always flip
        l_out, r_out, d_out = flip(left, right, disp)

        # New left = flipped old right
        torch.testing.assert_close(l_out, torch.flip(right, dims=[2]))
        # New right = flipped old left
        torch.testing.assert_close(r_out, torch.flip(left,  dims=[2]))
        # Disparity negated and flipped
        torch.testing.assert_close(d_out, -torch.flip(disp, dims=[2]))

    def test_no_flip_at_p0(self):
        """p=0 must never flip."""
        left, right, disp = make_stereo()
        flip = RandomHorizontalFlip(p=0.0)
        l, r, d = flip(left, right, disp)
        torch.testing.assert_close(l, left)
        torch.testing.assert_close(r, right)
        torch.testing.assert_close(d, disp)


class TestAsymmetricColorJitter:

    def test_output_shape_unchanged(self):
        left, right, disp = make_stereo()
        jitter = AsymmetricColorJitter()
        l, r, d = jitter(left, right, disp)
        assert l.shape == left.shape
        assert r.shape == right.shape
        assert d.shape == disp.shape

    def test_disp_unchanged(self):
        """Disparity must not be altered by colour jitter."""
        left, right, disp = make_stereo()
        jitter = AsymmetricColorJitter()
        _, _, d = jitter(left, right, disp)
        torch.testing.assert_close(d, disp)

    def test_output_in_valid_range(self):
        """Jittered images must stay in [0, 1]."""
        left, right, disp = make_stereo()
        jitter = AsymmetricColorJitter()
        l, r, _ = jitter(left, right, disp)
        assert l.min() >= 0.0 and l.max() <= 1.0
        assert r.min() >= 0.0 and r.max() <= 1.0


class TestNormalise:

    def test_maps_to_minus1_plus1(self):
        left  = torch.zeros(3, 32, 64)   # all zeros → should become -1
        right = torch.ones(3,  32, 64)   # all ones  → should become +1
        disp  = torch.rand(1,  32, 64) * 50
        disp_orig = disp.clone()

        norm = Normalise()
        l, r, d = norm(left, right, disp)

        assert l.min() == pytest.approx(-1.0) and l.max() == pytest.approx(-1.0)
        assert r.min() == pytest.approx( 1.0) and r.max() == pytest.approx( 1.0)
        torch.testing.assert_close(d, disp_orig)   # disp unchanged


class TestStereoTransform:

    def test_training_output_size(self):
        left, right, disp = make_stereo(H=128, W=256)
        t = StereoTransform(height=64, width=128, augment=True)
        l, r, d = t(left, right, disp)
        assert l.shape == (3, 64, 128)
        assert d.shape == (1, 64, 128)

    def test_images_in_normalised_range(self):
        left, right, disp = make_stereo(H=128, W=256)
        t = StereoTransform(height=64, width=128, augment=True)
        l, r, _ = t(left, right, disp)
        assert l.min() >= -1.0 and l.max() <= 1.0
        assert r.min() >= -1.0 and r.max() <= 1.0

    def test_no_augment_mode(self):
        """In augment=False mode, only normalisation is applied."""
        left  = torch.zeros(3, 64, 128)
        right = torch.ones(3,  64, 128)
        disp  = torch.rand(1, 64, 128)
        t = StereoTransform(height=64, width=128, augment=False)
        l, r, _ = t(left, right, disp)
        # Zeros → -1, ones → +1
        assert l.min() == pytest.approx(-1.0)
        assert r.max() == pytest.approx( 1.0)


# ===========================================================================
# Test Group 2: PFM reader
# ===========================================================================

class TestPFMReader:

    def _write_pfm(self, path, data):
        """Write a minimal PFM file for testing."""
        H, W = data.shape
        with open(path, 'wb') as f:
            f.write(b'Pf\n')
            f.write(f'{W} {H}\n'.encode())
            f.write(b'-1.0\n')   # little-endian, scale=1
            # PFM is stored bottom-to-top
            flipped = np.flipud(data).astype('<f4')
            f.write(flipped.tobytes())

    def test_read_correct_shape(self):
        data = np.random.rand(32, 64).astype(np.float32) * 100
        with tempfile.NamedTemporaryFile(suffix='.pfm', delete=False) as f:
            path = f.name
        try:
            self._write_pfm(path, data)
            loaded = read_pfm(path)
            assert loaded.shape == (32, 64)
            np.testing.assert_allclose(loaded, data, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_read_returns_float32(self):
        data = np.ones((16, 32), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.pfm', delete=False) as f:
            path = f.name
        try:
            self._write_pfm(path, data)
            loaded = read_pfm(path)
            assert loaded.dtype == np.float32
        finally:
            os.unlink(path)


# ===========================================================================
# Test Group 3: SceneFlowDataset (no data on disk)
# ===========================================================================

class TestSceneFlowDataset:

    def test_empty_root_raises(self):
        """Dataset with non-existent root must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="No samples found"):
            SceneFlowDataset(root='/nonexistent/path/sceneflow')

    def test_dataset_with_synthetic_files(self):
        """
        Create a minimal fake Scene Flow directory tree and verify
        the dataset loads from it correctly.
        """
        import struct
        from PIL import Image

        with tempfile.TemporaryDirectory() as root:
            # Create directory structure
            left_dir  = os.path.join(root, 'frames_cleanpass', 'TRAIN',
                                     'A', '0000', 'left')
            right_dir = os.path.join(root, 'frames_cleanpass', 'TRAIN',
                                     'A', '0000', 'right')
            disp_dir  = os.path.join(root, 'disparity', 'TRAIN',
                                     'A', '0000', 'left')
            for d in [left_dir, right_dir, disp_dir]:
                os.makedirs(d, exist_ok=True)

            # Write a tiny PNG image (8×16 pixels)
            H_img, W_img = 8, 16
            img_arr = np.random.randint(0, 255, (H_img, W_img, 3),
                                        dtype=np.uint8)
            Image.fromarray(img_arr).save(
                os.path.join(left_dir,  '0001.png'))
            Image.fromarray(img_arr).save(
                os.path.join(right_dir, '0001.png'))

            # Write a tiny PFM disparity
            disp_data = np.random.rand(H_img, W_img).astype(np.float32) * 50 + 1
            pfm_path  = os.path.join(disp_dir, '0001.pfm')
            with open(pfm_path, 'wb') as f:
                f.write(b'Pf\n')
                f.write(f'{W_img} {H_img}\n'.encode())
                f.write(b'-1.0\n')
                flipped = np.flipud(disp_data).astype('<f4')
                f.write(flipped.tobytes())

            # Load dataset (no crop since image is tiny — use augment=False)
            dataset = SceneFlowDataset(
                root=root, split='train',
                height=H_img, width=W_img, augment=False
            )

            assert len(dataset) == 1

            sample = dataset[0]
            assert 'left'      in sample
            assert 'right'     in sample
            assert 'disparity' in sample
            assert sample['left'].shape  == (3, H_img, W_img)
            assert sample['right'].shape == (3, H_img, W_img)
            assert sample['disparity'].shape == (1, H_img, W_img)

            # Images should be normalised to [-1, 1]
            assert sample['left'].min() >= -1.0
            assert sample['left'].max() <=  1.0