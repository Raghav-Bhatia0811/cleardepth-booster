"""
Milestone 6 Smoke Test — Full Model Assembly
=============================================
Tests the complete ClearDepthNet end-to-end at smoke-test resolution.

Run with: pytest tests/test_m6_full_model.py -v
"""

import torch
import pytest
from cleardepth.models.cleardepth_net import ClearDepthNet


BATCH   = 1
H_IMG   = 64
W_IMG   = 128
H_FEAT  = H_IMG // 4   # 16
W_FEAT  = W_IMG // 4   # 32
N_ITERS = 4


def make_model(**kwargs):
    defaults = dict(
        embed_dim=64,
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        reduction_ratios=[8, 4, 2, 1],
        hidden_dim=128,
        n_gru_iters=N_ITERS,
        corr_levels=4,
        corr_radius=4,
    )
    defaults.update(kwargs)
    return ClearDepthNet(**defaults)


class TestClearDepthNet:

    def test_training_mode_output_count(self):
        """Training mode must return n_iters predictions."""
        model = make_model()
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG)
        preds = model(left, right, n_iters=N_ITERS, test_mode=False)
        assert isinstance(preds, list), "Training output must be a list"
        assert len(preds) == N_ITERS

    def test_training_mode_shapes(self):
        """Each training prediction must be (B, 1, H/4, W/4)."""
        model = make_model()
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG)
        preds = model(left, right, n_iters=N_ITERS, test_mode=False)
        expected = (BATCH, 1, H_FEAT, W_FEAT)
        for i, d in enumerate(preds):
            assert d.shape == torch.Size(expected), \
                f"Pred {i}: expected {expected}, got {tuple(d.shape)}"

    def test_inference_mode_shape(self):
        """
        Inference mode must return a single full-resolution tensor.
        test_mode=True bilinearly upsamples the GRU's 1/4-scale output by
        upsample_scale (default 4), per the Architecture Report's
        "Inference Path — bilinear upsampled 4x to full HxW".
        """
        model = make_model()
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG)
        final = model(left, right, n_iters=N_ITERS, test_mode=True)
        assert isinstance(final, torch.Tensor), \
            "Inference output must be a tensor, not a list"
        assert final.shape == torch.Size((BATCH, 1, H_IMG, W_IMG))

    def test_inference_matches_bilinear_upsample_of_last_training_pred(self):
        """
        Inference output must exactly equal a bilinear ×upsample_scale
        upsample of the last training-mode (1/4-scale) prediction — the
        upsample is a plain, non-learned F.interpolate call, so this is
        an exact (not approximate) invariant.
        """
        import torch.nn.functional as F
        model = make_model()
        model.eval()
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG)
        with torch.no_grad():
            preds = model(left, right, n_iters=N_ITERS, test_mode=False)
            final = model(left, right, n_iters=N_ITERS, test_mode=True)

        expected = F.interpolate(
            preds[-1], scale_factor=model.upsample_scale,
            mode='bilinear', align_corners=False,
        )
        torch.testing.assert_close(final, expected)

    def test_gradient_flows_end_to_end(self):
        """Gradients must reach the input images from the loss."""
        model = make_model()
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG, requires_grad=True)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG, requires_grad=True)
        preds = model(left, right, n_iters=2, test_mode=False)
        loss = sum(p.mean() for p in preds)
        loss.backward()
        assert left.grad  is not None and not torch.isnan(left.grad).any()
        assert right.grad is not None and not torch.isnan(right.grad).any()

    def test_no_nan_in_output(self):
        """All predictions must be finite."""
        model = make_model()
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG)
        preds = model(left, right, n_iters=N_ITERS, test_mode=False)
        for i, d in enumerate(preds):
            assert not torch.isnan(d).any(), f"NaN in prediction {i}"
            assert not torch.isinf(d).any(), f"Inf in prediction {i}"

    def test_param_count_breakdown(self):
        """Print parameter breakdown and check total is reasonable."""
        model = make_model()
        counts = model.param_count()
        print("\nParameter breakdown:")
        for name, n in counts.items():
            print(f"  {name:20s}: {n:>12,}")
        # ClearDepth paper reports 99.45M total — our smoke config is smaller
        # With embed_dim=64 expect somewhere in the 10-30M range
        assert counts['total'] > 1_000_000, "Model suspiciously small"
        assert counts['total'] < 200_000_000, "Model unexpectedly huge"

    def test_eval_deterministic(self):
        """In eval mode, same inputs must give identical outputs."""
        model = make_model()
        model.eval()
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG)
        with torch.no_grad():
            out1 = model(left, right, n_iters=N_ITERS, test_mode=True)
            out2 = model(left, right, n_iters=N_ITERS, test_mode=True)
        torch.testing.assert_close(out1, out2)

    def test_n_iters_override(self):
        """n_iters argument must override the default from config."""
        model = make_model(n_gru_iters=22)
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG)
        # Override to 3 iters — should get exactly 3 predictions back
        preds = model(left, right, n_iters=3, test_mode=False)
        assert len(preds) == 3