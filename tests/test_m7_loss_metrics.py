"""
Milestone 7 Smoke Test — Loss + Metrics
========================================
Run with: pytest tests/test_m7_loss_metrics.py -v
"""

import torch
import pytest
from cleardepth.loss.sequence_loss import SequenceLoss
from cleardepth.evaluation.metrics import (
    compute_metrics, aggregate_metrics, format_metrics
)


BATCH  = 2
H, W   = 16, 32
N_PRED = 6    # Simulating 6 GRU iterations


def make_preds_and_gt(noise_scale=2.0):
    gt    = torch.rand(BATCH, 1, H, W) * 100 + 1.0   # gt in (1, 101)
    preds = [gt + torch.randn_like(gt) * noise_scale for _ in range(N_PRED)]
    return preds, gt


# ===========================================================================
# Test Group 1: SequenceLoss
# ===========================================================================

class TestSequenceLoss:

    def test_returns_scalar(self):
        loss_fn = SequenceLoss()
        preds, gt = make_preds_and_gt()
        loss = loss_fn(preds, gt)
        assert loss.shape == torch.Size([]), \
            f"Loss must be scalar, got shape {loss.shape}"

    def test_loss_is_positive(self):
        loss_fn = SequenceLoss()
        preds, gt = make_preds_and_gt()
        loss = loss_fn(preds, gt)
        assert loss.item() > 0

    def test_zero_loss_for_perfect_predictions(self):
        """Perfect predictions (pred == gt) must give zero loss."""
        loss_fn = SequenceLoss()
        gt    = torch.rand(BATCH, 1, H, W) * 100 + 1.0
        preds = [gt.clone() for _ in range(N_PRED)]
        loss  = loss_fn(preds, gt)
        assert loss.item() < 1e-5, f"Perfect pred loss should be ~0, got {loss.item()}"

    def test_later_iters_weighted_more(self):
        """
        Final prediction has weight 1.0; earlier ones are discounted.
        A model that is only accurate at the last step should have
        lower loss than one accurate only at the first step.
        """
        loss_fn = SequenceLoss(gamma=0.9)
        gt = torch.rand(BATCH, 1, H, W) * 100 + 1.0

        # Case A: only last prediction is accurate
        preds_a = [gt + torch.ones_like(gt) * 10] * (N_PRED - 1) + [gt.clone()]

        # Case B: only first prediction is accurate
        preds_b = [gt.clone()] + [gt + torch.ones_like(gt) * 10] * (N_PRED - 1)

        loss_a = loss_fn(preds_a, gt).item()
        loss_b = loss_fn(preds_b, gt).item()

        assert loss_a < loss_b, (
            f"Final-accurate model (loss={loss_a:.4f}) should have lower loss "
            f"than first-accurate model (loss={loss_b:.4f})"
        )

    def test_invalid_pixels_excluded(self):
        """Pixels with gt <= 0 must not contribute to loss."""
        loss_fn = SequenceLoss()
        gt = torch.rand(BATCH, 1, H, W) * 100 + 1.0

        # Mark half the pixels as invalid
        gt_with_invalid = gt.clone()
        gt_with_invalid[:, :, :H//2, :] = 0.0   # top half invalid

        preds = [gt.clone() for _ in range(N_PRED)]   # perfect predictions

        # Loss should still be ~0 (valid pixels are perfectly predicted)
        loss = loss_fn(preds, gt_with_invalid)
        assert loss.item() < 1e-5, \
            f"Loss with perfect valid-pixel preds should be ~0, got {loss.item()}"

    def test_gradient_flows_through_loss(self):
        """Loss must be differentiable w.r.t. predictions."""
        loss_fn = SequenceLoss()
        gt    = torch.rand(BATCH, 1, H, W) * 100 + 1.0
        preds = [torch.randn(BATCH, 1, H, W, requires_grad=True)
                 for _ in range(N_PRED)]
        loss = loss_fn(preds, gt)
        loss.backward()
        for i, p in enumerate(preds):
            assert p.grad is not None, f"No gradient for pred {i}"

    def test_gamma_effect(self):
        """Higher gamma → earlier iterations get more weight → higher loss
        when early preds are bad."""
        gt    = torch.rand(BATCH, 1, H, W) * 100 + 1.0
        preds = [gt + torch.ones_like(gt) * (N_PRED - i)
                 for i in range(N_PRED)]   # earlier = worse

        loss_high_gamma = SequenceLoss(gamma=0.99)(preds, gt).item()
        loss_low_gamma  = SequenceLoss(gamma=0.5)(preds, gt).item()

        # High gamma weights early (bad) iters more → higher total loss
        assert loss_high_gamma > loss_low_gamma, \
            "Higher gamma should give higher loss when early preds are worse"


# ===========================================================================
# Test Group 2: Metrics
# ===========================================================================

class TestMetrics:

    def test_returns_all_keys(self):
        pred = torch.rand(BATCH, 1, H, W) * 100
        gt   = torch.rand(BATCH, 1, H, W) * 100 + 1.0
        m = compute_metrics(pred, gt)
        for key in ['avg_err', 'rms', 'bad_0.5', 'bad_1.0', 'bad_2.0', 'bad_3.0']:
            assert key in m, f"Missing metric key: {key}"

    def test_perfect_prediction_gives_zero_errors(self):
        gt   = torch.rand(BATCH, 1, H, W) * 50 + 1.0
        pred = gt.clone()
        m = compute_metrics(pred, gt)
        assert m['avg_err'] < 1e-5
        assert m['rms']     < 1e-5
        assert m['bad_0.5'] < 1e-5
        assert m['bad_3.0'] < 1e-5

    def test_rms_geq_avg_err(self):
        """RMS is always >= mean absolute error (by Jensen's inequality)."""
        pred = torch.rand(BATCH, 1, H, W) * 100
        gt   = torch.rand(BATCH, 1, H, W) * 100 + 1.0
        m = compute_metrics(pred, gt)
        assert m['rms'] >= m['avg_err'] - 1e-6, \
            f"RMS ({m['rms']:.4f}) < AvgErr ({m['avg_err']:.4f})"

    def test_bad_thresholds_ordered(self):
        """Bad-0.5 >= Bad-1.0 >= Bad-2.0 >= Bad-3.0 (looser threshold = fewer bad)."""
        pred = torch.rand(BATCH, 1, H, W) * 10
        gt   = torch.rand(BATCH, 1, H, W) * 10 + 1.0
        m = compute_metrics(pred, gt)
        assert m['bad_0.5'] >= m['bad_1.0'] >= m['bad_2.0'] >= m['bad_3.0'], \
            "Bad-N thresholds not monotonically ordered"

    def test_invalid_pixels_excluded(self):
        """Metrics must ignore pixels where gt <= 0."""
        gt   = torch.rand(BATCH, 1, H, W) * 50 + 1.0
        pred = gt.clone()

        gt_invalid = gt.clone()
        gt_invalid[:, :, :H//2, :] = -1.0   # top half invalid

        # Perfect prediction on valid pixels → all metrics should be 0
        m = compute_metrics(pred, gt_invalid)
        assert m['avg_err'] < 1e-5

    def test_aggregate_metrics(self):
        """aggregate_metrics must return mean of all batch metrics."""
        m1 = {'avg_err': 2.0, 'rms': 3.0, 'bad_1.0': 10.0}
        m2 = {'avg_err': 4.0, 'rms': 5.0, 'bad_1.0': 20.0}
        agg = aggregate_metrics([m1, m2])
        assert abs(agg['avg_err'] - 3.0) < 1e-6
        assert abs(agg['rms']     - 4.0) < 1e-6
        assert abs(agg['bad_1.0'] - 15.0) < 1e-6

    def test_format_metrics_is_string(self):
        pred = torch.rand(BATCH, 1, H, W) * 100
        gt   = torch.rand(BATCH, 1, H, W) * 100 + 1.0
        m = compute_metrics(pred, gt)
        s = format_metrics(m)
        assert isinstance(s, str) and len(s) > 0
        assert 'AvgErr' in s and 'RMS' in s