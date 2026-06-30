"""
Milestone 5 Smoke Test — Post-Fusion GRU + Disparity Decoder
=============================================================
Run with: pytest tests/test_m5_gru.py -v
"""

import torch
import pytest
from cleardepth.models.gru.disparity_decoder import DisparityDecoder
from cleardepth.models.gru.post_fusion_gru import PostFusionGRU, PostFusionGRUCell
from cleardepth.models.correlation.correlation_pyramid import CorrelationPyramid


BATCH      = 1
H_FEAT     = 16     # 1/4 scale of 64×128
W_FEAT     = 32
HIDDEN_DIM = 128
CORR_CH    = 36     # 4 levels × 9 offsets


# ===========================================================================
# Test Group 1: DisparityDecoder
# ===========================================================================

class TestDisparityDecoder:

    def test_output_shape(self):
        """Decoder must output (B, 1, H, W)."""
        decoder = DisparityDecoder(hidden_dim=HIDDEN_DIM)
        h = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        delta = decoder(h)
        assert delta.shape == (BATCH, 1, H_FEAT, W_FEAT)

    def test_gradient_flow(self):
        decoder = DisparityDecoder(hidden_dim=HIDDEN_DIM)
        h = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT, requires_grad=True)
        delta = decoder(h)
        delta.mean().backward()
        assert h.grad is not None and not torch.isnan(h.grad).any()

    def test_signed_output(self):
        """Delta should be able to take both positive and negative values."""
        decoder = DisparityDecoder(hidden_dim=HIDDEN_DIM)
        decoder.eval()
        with torch.no_grad():
            h = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
            delta = decoder(h)
        # With random weights, output should span both signs
        assert delta.min() < 0 and delta.max() > 0, \
            "Decoder output is all-positive or all-negative — check activations"


# ===========================================================================
# Test Group 2: PostFusionGRUCell
# ===========================================================================

class TestPostFusionGRUCell:

    def _make_cell(self):
        input_dim = CORR_CH + 1 + 3 * HIDDEN_DIM
        return PostFusionGRUCell(input_dim=input_dim, hidden_dim=HIDDEN_DIM)

    def test_output_shape(self):
        """Cell output hidden state must match input hidden state shape."""
        cell = self._make_cell()
        input_dim = CORR_CH + 1 + 3 * HIDDEN_DIM
        h = torch.zeros(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        x = torch.randn(BATCH, input_dim,  H_FEAT, W_FEAT)
        c_k = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        c_r = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        c_h = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        h_new = cell(h, x, c_k, c_r, c_h)
        assert h_new.shape == h.shape

    def test_structural_bias_has_effect(self):
        """Different structural biases must produce different hidden states."""
        cell = self._make_cell()
        cell.eval()
        input_dim = CORR_CH + 1 + 3 * HIDDEN_DIM
        h = torch.zeros(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        x = torch.randn(BATCH, input_dim, H_FEAT, W_FEAT)

        c_k1 = torch.zeros(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        c_r1 = torch.zeros(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        c_h1 = torch.zeros(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)

        c_k2 = torch.ones(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        c_r2 = torch.ones(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        c_h2 = torch.ones(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)

        with torch.no_grad():
            h1 = cell(h, x, c_k1, c_r1, c_h1)
            h2 = cell(h, x, c_k2, c_r2, c_h2)

        assert not torch.allclose(h1, h2), \
            "Structural biases have no effect on hidden state"

    def test_gradient_flow(self):
        cell = self._make_cell()
        input_dim = CORR_CH + 1 + 3 * HIDDEN_DIM
        h   = torch.zeros(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT, requires_grad=True)
        x   = torch.randn(BATCH, input_dim,  H_FEAT, W_FEAT, requires_grad=True)
        c_k = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT, requires_grad=True)
        c_r = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT, requires_grad=True)
        c_h = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT, requires_grad=True)
        h_new = cell(h, x, c_k, c_r, c_h)
        h_new.mean().backward()
        for name, t in [('h', h), ('x', x), ('c_k', c_k),
                        ('c_r', c_r), ('c_h', c_h)]:
            assert t.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(t.grad).any(), f"NaN gradient for {name}"


# ===========================================================================
# Test Group 3: PostFusionGRU (full multi-scale)
# ===========================================================================

class TestPostFusionGRU:

    def _make_gru(self):
        return PostFusionGRU(
            corr_channels=CORR_CH,
            hidden_dim=HIDDEN_DIM,
            n_gru_layers=3,
        )

    def _make_inputs(self):
        corr_fn = CorrelationPyramid(num_levels=4, radius=4)
        feat_l = torch.randn(BATCH, 64, H_FEAT, W_FEAT)
        feat_r = torch.randn(BATCH, 64, H_FEAT, W_FEAT)
        c_k = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        c_r = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        c_h = torch.randn(BATCH, HIDDEN_DIM, H_FEAT, W_FEAT)
        return corr_fn, feat_l, feat_r, c_k, c_r, c_h

    def test_prediction_count(self):
        """Must return exactly n_iters predictions."""
        gru = self._make_gru()
        corr_fn, fl, fr, c_k, c_r, c_h = self._make_inputs()
        for n in [1, 4, 8]:
            preds = gru(fl, fr, c_k, c_r, c_h, corr_fn, n_iters=n)
            assert len(preds) == n, f"Expected {n} preds, got {len(preds)}"

    def test_prediction_shapes(self):
        """Each prediction must be (B, 1, H_feat, W_feat)."""
        gru = self._make_gru()
        corr_fn, fl, fr, c_k, c_r, c_h = self._make_inputs()
        preds = gru(fl, fr, c_k, c_r, c_h, corr_fn, n_iters=4)
        for i, d in enumerate(preds):
            assert d.shape == (BATCH, 1, H_FEAT, W_FEAT), \
                f"Iter {i}: expected ({BATCH}, 1, {H_FEAT}, {W_FEAT}), got {tuple(d.shape)}"

    def test_disparity_changes_over_iterations(self):
        """Disparity estimates must change across iterations."""
        gru = self._make_gru()
        corr_fn, fl, fr, c_k, c_r, c_h = self._make_inputs()
        preds = gru(fl, fr, c_k, c_r, c_h, corr_fn, n_iters=4)
        assert not torch.allclose(preds[0], preds[-1]), \
            "Disparity unchanged across iterations — GRU not updating"

    def test_no_nan_in_predictions(self):
        """All predictions must be finite."""
        gru = self._make_gru()
        corr_fn, fl, fr, c_k, c_r, c_h = self._make_inputs()
        preds = gru(fl, fr, c_k, c_r, c_h, corr_fn, n_iters=4)
        for i, d in enumerate(preds):
            assert not torch.isnan(d).any(), f"NaN in iteration {i}"
            assert not torch.isinf(d).any(), f"Inf in iteration {i}"

    def test_gradient_flow_through_iterations(self):
        """Gradients must flow from final prediction back to inputs."""
        gru = self._make_gru()
        corr_fn, fl, fr, c_k, c_r, c_h = self._make_inputs()
        fl.requires_grad_(True)
        c_k.requires_grad_(True)
        preds = gru(fl, fr, c_k, c_r, c_h, corr_fn, n_iters=3)
        # Sum all predictions for loss (like sequence loss does)
        loss = sum(p.mean() for p in preds)
        loss.backward()
        assert fl.grad  is not None and not torch.isnan(fl.grad).any()
        assert c_k.grad is not None and not torch.isnan(c_k.grad).any()