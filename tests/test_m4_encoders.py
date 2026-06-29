"""
Milestone 4 Smoke Test — Encoders + Correlation Pyramid
========================================================
Tests FeatureEncoder, ContextEncoder, and CorrelationPyramid
at smoke-test resolution (64×128, batch=1).

Run with: pytest tests/test_m4_encoders.py -v
"""

import torch
import pytest
from cleardepth.models.encoders.feature_encoder import FeatureEncoder
from cleardepth.models.encoders.context_encoder import ContextEncoder
from cleardepth.models.correlation.correlation_pyramid import CorrelationPyramid


BATCH      = 1
H_IMG      = 64
W_IMG      = 128
EMBED_C    = 64
HIDDEN_DIM = 128
# Feature map dims at 1/4 scale
H_FEAT = H_IMG // 4   # 16
W_FEAT = W_IMG // 4   # 32


# ===========================================================================
# Test Group 1: FeatureEncoder
# ===========================================================================

class TestFeatureEncoder:

    def test_output_shapes(self):
        """Left and right feature maps must both be (B, C, H/4, W/4)."""
        enc = FeatureEncoder(embed_dim=EMBED_C)
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG)
        fl, fr = enc(left, right)
        expected = (BATCH, EMBED_C, H_FEAT, W_FEAT)
        assert fl.shape == torch.Size(expected), f"Left:  {tuple(fl.shape)}"
        assert fr.shape == torch.Size(expected), f"Right: {tuple(fr.shape)}"

    def test_shared_weights(self):
        """Left and right branches must share the exact same parameters."""
        enc = FeatureEncoder(embed_dim=EMBED_C)
        # There is only one backbone — verify it's a single object
        assert hasattr(enc, 'backbone'), "Missing backbone attribute"
        # Parameter count: shared backbone counted once
        total_params = sum(p.numel() for p in enc.parameters())
        backbone_params = sum(p.numel() for p in enc.backbone.parameters())
        assert total_params == backbone_params, (
            "FeatureEncoder has extra parameters beyond the backbone "
            "— weights are NOT shared correctly"
        )

    def test_different_inputs_different_outputs(self):
        """Different images must produce different feature maps."""
        enc = FeatureEncoder(embed_dim=EMBED_C)
        enc.eval()
        with torch.no_grad():
            left  = torch.randn(BATCH, 3, H_IMG, W_IMG)
            right = torch.randn(BATCH, 3, H_IMG, W_IMG)
            fl, fr = enc(left, right)
        assert not torch.allclose(fl, fr), \
            "Left and right features are identical — backbone not running correctly"

    def test_gradient_flow(self):
        """Gradients must flow through both left and right passes."""
        enc = FeatureEncoder(embed_dim=EMBED_C)
        left  = torch.randn(BATCH, 3, H_IMG, W_IMG, requires_grad=True)
        right = torch.randn(BATCH, 3, H_IMG, W_IMG, requires_grad=True)
        fl, fr = enc(left, right)
        (fl.mean() + fr.mean()).backward()
        assert left.grad  is not None and not torch.isnan(left.grad).any()
        assert right.grad is not None and not torch.isnan(right.grad).any()


# ===========================================================================
# Test Group 2: ContextEncoder
# ===========================================================================

class TestContextEncoder:

    def test_output_shapes(self):
        """c_k, c_r, c_h must all be (B, hidden_dim, H/4, W/4)."""
        enc = ContextEncoder(embed_dim=EMBED_C, hidden_dim=HIDDEN_DIM)
        left = torch.randn(BATCH, 3, H_IMG, W_IMG)
        c_k, c_r, c_h = enc(left)
        expected = (BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        assert c_k.shape == torch.Size(expected), f"c_k: {tuple(c_k.shape)}"
        assert c_r.shape == torch.Size(expected), f"c_r: {tuple(c_r.shape)}"
        assert c_h.shape == torch.Size(expected), f"c_h: {tuple(c_h.shape)}"

    def test_three_heads_independent(self):
        """c_k, c_r, c_h must be distinct tensors (not the same projection)."""
        enc = ContextEncoder(embed_dim=EMBED_C, hidden_dim=HIDDEN_DIM)
        enc.eval()
        left = torch.randn(BATCH, 3, H_IMG, W_IMG)
        with torch.no_grad():
            c_k, c_r, c_h = enc(left)
        # All three should differ (probability of equality is essentially zero)
        assert not torch.allclose(c_k, c_r), "c_k == c_r: proj heads not independent"
        assert not torch.allclose(c_r, c_h), "c_r == c_h: proj heads not independent"
        assert not torch.allclose(c_k, c_h), "c_k == c_h: proj heads not independent"

    def test_output_bounded(self):
        """tanh activation must bound outputs to (-1, 1)."""
        enc = ContextEncoder(embed_dim=EMBED_C, hidden_dim=HIDDEN_DIM)
        enc.eval()
        left = torch.randn(BATCH, 3, H_IMG, W_IMG) * 100   # large input
        with torch.no_grad():
            c_k, c_r, c_h = enc(left)
        for name, c in [('c_k', c_k), ('c_r', c_r), ('c_h', c_h)]:
            assert c.min() >= -1.0 - 1e-6, f"{name} below -1"
            assert c.max() <=  1.0 + 1e-6, f"{name} above +1"

    def test_gradient_flow(self):
        """Gradients must flow through all three projection heads."""
        enc = ContextEncoder(embed_dim=EMBED_C, hidden_dim=HIDDEN_DIM)
        left = torch.randn(BATCH, 3, H_IMG, W_IMG, requires_grad=True)
        c_k, c_r, c_h = enc(left)
        (c_k.mean() + c_r.mean() + c_h.mean()).backward()
        assert left.grad is not None
        assert not torch.isnan(left.grad).any()

    def test_proj_heads_have_separate_params(self):
        """proj_k, proj_r, proj_h must have independent weight tensors."""
        enc = ContextEncoder(embed_dim=EMBED_C, hidden_dim=HIDDEN_DIM)
        wk = enc.proj_k[0].weight.data
        wr = enc.proj_r[0].weight.data
        wh = enc.proj_h[0].weight.data
        # Weights are randomly initialised — probability of collision ≈ 0
        assert not torch.allclose(wk, wr), "proj_k and proj_r share weights"
        assert not torch.allclose(wr, wh), "proj_r and proj_h share weights"


# ===========================================================================
# Test Group 3: CorrelationPyramid
# ===========================================================================

class TestCorrelationPyramid:

    def test_output_shape_default(self):
        """Default: 4 levels × 9 offsets = 36 channels."""
        pyramid = CorrelationPyramid(num_levels=4, radius=4)
        feat_l = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT)
        feat_r = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT)
        disp   = torch.zeros(BATCH, 1, H_FEAT, W_FEAT)
        out = pyramid(feat_l, feat_r, disp)
        expected_ch = 4 * (2 * 4 + 1)   # 36
        assert out.shape == (BATCH, expected_ch, H_FEAT, W_FEAT), \
            f"Expected ({BATCH}, {expected_ch}, {H_FEAT}, {W_FEAT}), got {tuple(out.shape)}"

    def test_output_channels_property(self):
        """out_channels property must match actual output."""
        pyramid = CorrelationPyramid(num_levels=4, radius=4)
        feat_l = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT)
        feat_r = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT)
        disp   = torch.zeros(BATCH, 1, H_FEAT, W_FEAT)
        out = pyramid(feat_l, feat_r, disp)
        assert out.shape[1] == pyramid.out_channels

    def test_zero_disparity_center_lookup(self):
        """At zero disparity, correlation should be highest at offset=0
        when left and right features are identical."""
        pyramid = CorrelationPyramid(num_levels=1, radius=4)
        feat = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT)
        disp = torch.zeros(BATCH, 1, H_FEAT, W_FEAT)
        # When left == right and disparity=0, offset=0 should give max correlation
        out = pyramid(feat, feat, disp)   # (B, 9, H, W)
        center_idx = 4   # radius=4, so offset=0 is at index 4
        center_corr = out[:, center_idx, :, :]
        for i in range(9):
            if i != center_idx:
                # Center should be >= other offsets on average
                assert center_corr.mean() >= out[:, i, :, :].mean() - 1e-4

    def test_gradient_flow(self):
        """Gradients must flow from correlation output back to features."""
        pyramid = CorrelationPyramid(num_levels=4, radius=4)
        feat_l = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT, requires_grad=True)
        feat_r = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT, requires_grad=True)
        disp   = torch.zeros(BATCH, 1, H_FEAT, W_FEAT)
        out = pyramid(feat_l, feat_r, disp)
        out.mean().backward()
        assert feat_l.grad is not None and not torch.isnan(feat_l.grad).any()
        assert feat_r.grad is not None and not torch.isnan(feat_r.grad).any()

    def test_nonzero_disparity(self):
        """Pyramid must handle non-zero disparity without crashing."""
        pyramid = CorrelationPyramid(num_levels=4, radius=4)
        feat_l = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT)
        feat_r = torch.randn(BATCH, EMBED_C, H_FEAT, W_FEAT)
        # Disparity of 5 pixels
        disp = torch.ones(BATCH, 1, H_FEAT, W_FEAT) * 5.0
        out = pyramid(feat_l, feat_r, disp)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_no_learnable_parameters(self):
        """CorrelationPyramid is a pure computation — no learned weights."""
        pyramid = CorrelationPyramid()
        total_params = sum(p.numel() for p in pyramid.parameters())
        assert total_params == 0, \
            f"Expected 0 parameters, got {total_params}"