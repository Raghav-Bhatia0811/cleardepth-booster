"""
Milestone 10 Smoke Test — Evaluation + Visualization
======================================================
Run with: pytest tests/test_m10_eval.py -v
"""

import torch
import numpy as np
import os
import tempfile
import pytest

from cleardepth.evaluation.visualize import (
    disp_to_color, error_to_color, tensor_to_rgb,
    make_comparison_figure, save_disp_image, save_comparison_figure,
)
from cleardepth.evaluation.metrics import compute_metrics


H, W = 32, 64


# ===========================================================================
# Test Group 1: Visualization utilities
# ===========================================================================

class TestDispToColor:

    def test_output_shape(self):
        disp = torch.rand(1, H, W) * 100
        rgb  = disp_to_color(disp)
        assert rgb.shape == (H, W, 3)

    def test_output_dtype_uint8(self):
        disp = torch.rand(1, H, W) * 100
        rgb  = disp_to_color(disp)
        assert rgb.dtype == np.uint8

    def test_output_range(self):
        disp = torch.rand(1, H, W) * 100
        rgb  = disp_to_color(disp)
        assert rgb.min() >= 0 and rgb.max() <= 255

    def test_2d_input_accepted(self):
        """Should accept (H, W) as well as (1, H, W)."""
        disp = torch.rand(H, W) * 100
        rgb  = disp_to_color(disp)
        assert rgb.shape == (H, W, 3)

    def test_vmin_vmax_respected(self):
        """Uniform disparity should produce the same colour at every pixel."""
        disp = torch.ones(1, H, W) * 50
        rgb1 = disp_to_color(disp, vmin=0, vmax=100)
        rgb2 = disp_to_color(disp, vmin=0, vmax=50)
        # Every pixel must be identical (std across spatial dims = 0)
        assert rgb1.reshape(-1, 3).std(axis=0).sum() == 0
        assert rgb2.reshape(-1, 3).std(axis=0).sum() == 0
        # Different vmax should produce different colours
        assert not np.array_equal(rgb1[0, 0], rgb2[0, 0])

    def test_different_colormaps(self):
        disp = torch.rand(1, H, W) * 100
        for cmap in ['plasma', 'jet', 'magma', 'hot']:
            rgb = disp_to_color(disp, colormap=cmap)
            assert rgb.shape == (H, W, 3)


class TestErrorToColor:

    def test_output_shape(self):
        pred = torch.rand(1, H, W) * 80
        gt   = torch.rand(1, H, W) * 80
        err  = error_to_color(pred, gt)
        assert err.shape == (H, W, 3)

    def test_zero_error_is_dark(self):
        """Perfect prediction → zero error → every pixel must be the same colour."""
        pred = torch.ones(1, H, W) * 40.0
        gt   = torch.ones(1, H, W) * 40.0
        err  = error_to_color(pred, gt, max_error=5.0)
        # Every pixel identical (std across spatial dims = 0)
        assert err.reshape(-1, 3).std(axis=0).sum() == 0


class TestTensorToRgb:

    def test_minus1_to_1_range(self):
        img = torch.full((3, H, W), -1.0)
        rgb = tensor_to_rgb(img)
        assert rgb.min() == 0

        img = torch.full((3, H, W), 1.0)
        rgb = tensor_to_rgb(img)
        assert rgb.max() == 255

    def test_output_shape(self):
        img = torch.rand(3, H, W) * 2 - 1
        rgb = tensor_to_rgb(img)
        assert rgb.shape == (H, W, 3)
        assert rgb.dtype == np.uint8


class TestComparisonFigure:

    def test_output_shape(self):
        """Figure must be H × 4W (four panels side by side)."""
        img  = torch.rand(3, H, W) * 2 - 1
        pred = torch.rand(1, H, W) * 80
        gt   = torch.rand(1, H, W) * 80 + 1

        fig = make_comparison_figure(img, pred, gt)
        assert fig.shape == (H, W * 4, 3)
        assert fig.dtype == np.uint8

    def test_save_comparison_figure(self):
        """save_comparison_figure must write a valid PNG file."""
        img  = torch.rand(3, H, W) * 2 - 1
        pred = torch.rand(1, H, W) * 80
        gt   = torch.rand(1, H, W) * 80 + 1

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            save_comparison_figure(img, pred, gt, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_disp_image(self):
        """save_disp_image must write a valid PNG file."""
        disp = torch.rand(1, H, W) * 100
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            save_disp_image(disp, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ===========================================================================
# Test Group 2: End-to-end evaluation with tiny model
# ===========================================================================

class TestEvaluationPipeline:

    def test_full_eval_smoke(self):
        """
        Run the full evaluation loop with a tiny model and synthetic data.
        Verifies the evaluate() function from evaluate.py works end-to-end.
        """
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        from torch.utils.data import DataLoader
        from cleardepth.models.cleardepth_net import ClearDepthNet
        from cleardepth.training.trainer import SyntheticStereoDataset

        # Tiny model for speed
        model = ClearDepthNet(
            embed_dim=32,
            depths=[1, 1, 1, 1],
            num_heads=[1, 2, 4, 8],
            reduction_ratios=[8, 4, 2, 1],
            hidden_dim=64,
            n_gru_iters=2,
        )
        model.eval()

        dataset = SyntheticStereoDataset(length=4, height=64, width=128)
        loader  = DataLoader(dataset, batch_size=2, shuffle=False,
                             num_workers=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Import and call the evaluate function directly
            from scripts.evaluate import evaluate
            results = evaluate(
                model=model,
                loader=loader,
                device=torch.device('cpu'),
                n_iters=2,
                max_disp=64.0,
                output_dir=tmpdir,
                save_viz=True,
                n_viz=2,
            )

        # Check all expected metric keys are present
        for key in ['avg_err', 'rms', 'bad_0.5', 'bad_1.0', 'bad_2.0', 'bad_3.0']:
            assert key in results, f"Missing metric: {key}"

        # Metrics must be finite
        for k, v in results.items():
            assert np.isfinite(v), f"Non-finite metric {k}={v}"