"""
tests/test_real_forward_pass.py
================================
Feeds real images from the Scene Flow sample pack through the
full ClearDepth model and verifies the output is a valid disparity map.

Run with: pytest tests/test_real_forward_pass.py -v
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from omegaconf import OmegaConf

from cleardepth.data.sceneflow_sample import build_sample_loader
from cleardepth.models.cleardepth_net import ClearDepthNet
from cleardepth.loss.sequence_loss import SequenceLoss


# ── Paths ──────────────────────────────────────────────────────────────────
SAMPLE_ROOT = "C:/Users/ragha/Downloads/Sampler/Sampler"
SUBSETS     = ["Monkaa", "FlyingThings3D"]

# ── Use CPU so VRAM is not a concern for this test ─────────────────────────
DEVICE = torch.device('cpu')


@pytest.fixture(scope='module')
def cfg():
    """Load smoke_test.yaml which merges model + training configs."""
    base = OmegaConf.load("configs/model/cleardepth.yaml")
    return base


@pytest.fixture(scope='module')
def model(cfg):
    """Build ClearDepthNet from config and set to eval mode."""
    net = ClearDepthNet(
        in_channels      = cfg.backbone.in_channels,
        embed_dim        = cfg.backbone.embed_dims[0],  # base dim = 64
        depths           = list(cfg.backbone.depths),
        num_heads        = list(cfg.backbone.num_heads),
        reduction_ratios = list(cfg.backbone.reduction_ratios),
        mlp_ratio        = cfg.backbone.mlp_ratio,
        drop_rate        = cfg.backbone.drop_rate,
        drop_path_rate   = cfg.backbone.drop_path_rate,
        hidden_dim       = cfg.gru.hidden_dim,
        n_gru_layers     = 3,
        n_gru_iters      = 3,   # keep small for fast testing
        corr_levels      = cfg.correlation.num_levels,
        corr_radius      = cfg.correlation.radius,
    )
    net.to(DEVICE)
    net.eval()
    return net


@pytest.fixture(scope='module')
def sample_batch():
    """Load the first real batch from the sample pack."""
    loader = build_sample_loader(
        root=SAMPLE_ROOT,
        subsets=SUBSETS,
        batch_size=1,
        shuffle=False,
    )
    return next(iter(loader))


# ── Tests ───────────────────────────────────────────────────────────────────

def test_model_loads(model):
    """Model builds and reports parameter count."""
    assert model is not None
    counts = model.param_count()
    print(f"\n    feature_encoder : {counts['feature_encoder']:>12,}")
    print(f"    context_encoder : {counts['context_encoder']:>12,}")
    print(f"    gru             : {counts['gru']:>12,}")
    print(f"    total           : {counts['total']:>12,}")
    assert counts['total'] > 0


def test_images_load_correctly(sample_batch):
    """Real images load with correct shapes and value ranges."""
    left = sample_batch['left']
    right = sample_batch['right']
    disp  = sample_batch['disparity']

    print(f"\n    Image shape : {tuple(left.shape)}")
    print(f"    Disp shape  : {tuple(disp.shape)}")
    print(f"    Disp range  : [{disp.min():.2f}, {disp.max():.2f}]")

    assert left.shape[1]  == 3
    assert right.shape[1] == 3
    assert disp.shape[1]  == 1
    assert left.min() >= 0.0 and left.max() <= 1.0


def test_forward_pass_runs(model, sample_batch):
    """
    Full forward pass completes without errors on real images.
    This is the most important test.
    """
    left  = sample_batch['left'].to(DEVICE)
    right = sample_batch['right'].to(DEVICE)

    with torch.no_grad():
        predictions = model(left, right, n_iters=3, test_mode=False)

    assert predictions is not None,            "Model returned None"
    assert isinstance(predictions, (list, tuple)), \
        "Model should return a list of predictions"
    assert len(predictions) == 3,             \
        f"Expected 3 predictions, got {len(predictions)}"

    print(f"\n    GRU iterations   : {len(predictions)}")
    print(f"    Final pred shape : {tuple(predictions[-1].shape)}")


def test_output_shape(model, sample_batch):
    """
    Output disparity has correct shape.
    Model runs internally at 1/4 scale so output is (B, 1, H/4, W/4).
    """
    left  = sample_batch['left'].to(DEVICE)
    right = sample_batch['right'].to(DEVICE)

    H, W = left.shape[2], left.shape[3]

    with torch.no_grad():
        predictions = model(left, right, n_iters=3, test_mode=False)

    final = predictions[-1]

    assert final.ndim    == 4,  f"Expected 4D output, got {final.ndim}D"
    assert final.shape[0] == 1, "Batch size should be 1"
    assert final.shape[1] == 1, f"Expected 1 channel, got {final.shape[1]}"

    expected_h = H // 4
    expected_w = W // 4
    assert final.shape[2] == expected_h, \
        f"Expected height {expected_h}, got {final.shape[2]}"
    assert final.shape[3] == expected_w, \
        f"Expected width {expected_w}, got {final.shape[3]}"

    print(f"\n    Input  : ({H}, {W})")
    print(f"    Output : {tuple(final.shape[2:])}  (1/4 scale as expected)")


def test_output_values_are_finite(model, sample_batch):
    """
    No NaN or Inf values in any prediction.
    NaN = something exploded. Inf = overflow.
    """
    left  = sample_batch['left'].to(DEVICE)
    right = sample_batch['right'].to(DEVICE)

    with torch.no_grad():
        predictions = model(left, right, n_iters=3, test_mode=False)

    for i, pred in enumerate(predictions):
        assert torch.isfinite(pred).all(), \
            f"Iteration {i} contains NaN or Inf!"

    print(f"\n    All {len(predictions)} predictions are finite (no NaN/Inf)")


def test_output_range_is_sane(model, sample_batch):
    """
    Output values are in a numerically sane range.
    At random init weights values won't match ground truth yet —
    that is expected. We just verify no explosions.
    """
    left  = sample_batch['left'].to(DEVICE)
    right = sample_batch['right'].to(DEVICE)

    with torch.no_grad():
        predictions = model(left, right, n_iters=3, test_mode=False)

    final = predictions[-1]

    print(f"\n    Output range: [{final.min():.4f}, {final.max():.4f}]")
    print(f"    (random init — values won't match GT yet, that is normal)")

    assert final.min() > -10000, "Output values unreasonably negative"
    assert final.max() <  10000, "Output values unreasonably large"


def test_loss_computes_on_real_data(model, sample_batch):
    """
    Sequence loss computes a finite positive scalar on real data.
    Proves the full training pipeline is wired correctly end-to-end.
    """
    left    = sample_batch['left'].to(DEVICE)
    right   = sample_batch['right'].to(DEVICE)
    gt_disp = sample_batch['disparity'].to(DEVICE)  # (1, 1, H, W)

    loss_fn = SequenceLoss(gamma=0.9, max_disp=192)

    with torch.no_grad():
        predictions = model(left, right, n_iters=3, test_mode=False)

    # Ground truth must be downsampled to match 1/4 scale model output
    pred_h = predictions[-1].shape[2]
    pred_w = predictions[-1].shape[3]
    gt_h   = gt_disp.shape[2]
    gt_w   = gt_disp.shape[3]

    if (pred_h, pred_w) != (gt_h, gt_w):
        scale  = pred_h / gt_h          # same ratio for H and W
        gt_disp_scaled = torch.nn.functional.interpolate(
            gt_disp,
            size=(pred_h, pred_w),
            mode='nearest',
        ) * scale
    else:
        gt_disp_scaled = gt_disp

    loss = loss_fn(predictions, gt_disp_scaled)

    print(f"\n    GT disp range  : [{gt_disp.min():.2f}, {gt_disp.max():.2f}]")
    print(f"    Scaled GT range: [{gt_disp_scaled.min():.2f}, "
          f"{gt_disp_scaled.max():.2f}]")
    print(f"    Loss value     : {loss.item():.4f}")

    assert torch.isfinite(loss), "Loss is NaN or Inf!"
    assert loss.item() > 0,      "Loss should be positive"