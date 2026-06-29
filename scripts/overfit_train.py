"""
scripts/overfit_train.py
=========================
Intentionally overfits ClearDepth on the 6 sample pack images.

Goal: Prove the model CAN learn by memorizing a tiny dataset.
Expected outcome after 500 steps:
  - Loss drops from ~100 down to ~1-5
  - AvgErr drops from ~20px down to ~1-3px
  - Bad-1.0 drops from ~99% down to ~10-30%

This is NOT a generalization test. It is an architecture sanity check.
A model that cannot overfit 6 images has a fundamental bug.

Run with:
    conda activate cleardepth
    cd C:\\Users\\ragha\\cleardepth
    python scripts/overfit_train.py
"""

import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from cleardepth.data.sceneflow_sample import SceneFlowSampleDataset
from cleardepth.models.cleardepth_net import ClearDepthNet
from cleardepth.loss.sequence_loss import SequenceLoss
from cleardepth.evaluation.metrics import compute_metrics, format_metrics

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_ROOT    = "C:/Users/ragha/Downloads/Sampler/Sampler"
SUBSETS        = ["Monkaa", "FlyingThings3D"]
CHECKPOINT_DIR = "checkpoints/overfit"
CHECKPOINT_OUT = "checkpoints/overfit/overfit_final.pt"

TOTAL_STEPS    = 500
LOG_EVERY      = 50       # print metrics every 50 steps
LR             = 1e-4     # constant learning rate — no scheduler
GRU_ITERS      = 3        # keep small for speed
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CFG = dict(
    in_channels      = 3,
    embed_dim        = 64,
    depths           = [2, 2, 2, 2],
    num_heads        = [1, 2, 4, 8],
    reduction_ratios = [8, 4, 2, 1],
    mlp_ratio        = 4.0,
    drop_rate        = 0.0,
    drop_path_rate   = 0.0,   # disable stochastic depth for overfitting
    hidden_dim       = 128,
    n_gru_layers     = 3,
    n_gru_iters      = GRU_ITERS,
    corr_levels      = 4,
    corr_radius      = 4,
)


def main():
    log.info("=" * 60)
    log.info("ClearDepth — Overfit Training on Sample Pack")
    log.info("=" * 60)
    log.info(f"Device     : {DEVICE}")
    log.info(f"Steps      : {TOTAL_STEPS}")
    log.info(f"LR         : {LR} (constant, no scheduler)")
    log.info(f"GRU iters  : {GRU_ITERS}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = SceneFlowSampleDataset(
        root=SAMPLE_ROOT,
        subsets=SUBSETS,
        pass_name='RGB_cleanpass',
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    log.info(f"Dataset    : {len(dataset)} samples")

    # ── Model ──────────────────────────────────────────────────────────────
    model = ClearDepthNet(**MODEL_CFG).to(DEVICE)
    counts = model.param_count()
    log.info(f"Parameters : {counts['total']:,}")

    # ── Loss + Optimizer ───────────────────────────────────────────────────
    loss_fn   = SequenceLoss(gamma=0.9, max_disp=192.0)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    # No scheduler — constant LR is better for overfitting

    # ── Track loss history for summary ────────────────────────────────────
    loss_history = []

    # ── Training loop ──────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("Step      | Loss     | AvgErr  | Bad-1.0  | Bad-2.0  | LR")
    log.info("-" * 60)

    model.train()
    global_step = 0

    while global_step < TOTAL_STEPS:
        for batch in loader:
            if global_step >= TOTAL_STEPS:
                break

            left  = batch['left'].to(DEVICE)        # (1, 3, H, W)
            right = batch['right'].to(DEVICE)       # (1, 3, H, W)
            gt    = batch['disparity'].to(DEVICE)   # (1, 1, H, W)

            # Resize to small resolution for local GPU
            # 540x960 → 128x256 — 18x fewer pixels, fits in 4GB VRAM
            TRAIN_H, TRAIN_W = 128, 256
            left  = torch.nn.functional.interpolate(
                left,  size=(TRAIN_H, TRAIN_W), mode='bilinear',
                align_corners=False
            )
            right = torch.nn.functional.interpolate(
                right, size=(TRAIN_H, TRAIN_W), mode='bilinear',
                align_corners=False
            )
            # Scale disparity values proportionally when resizing
            disp_scale_resize = TRAIN_W / gt.shape[-1]
            gt = torch.nn.functional.interpolate(
                gt, size=(TRAIN_H, TRAIN_W), mode='nearest'
            ) * disp_scale_resize

            # Forward pass
            preds = model(left, right, n_iters=GRU_ITERS, test_mode=False)

            # Downsample GT to match 1/4 scale model output
            _, _, H_pred, W_pred = preds[0].shape
            scale = W_pred / gt.shape[-1]
            if gt.shape[-2] != H_pred or gt.shape[-1] != W_pred:
                gt_scaled = torch.nn.functional.interpolate(
                    gt, size=(H_pred, W_pred), mode='nearest'
                ) * scale
            else:
                gt_scaled = gt

            # Loss + backward
            loss = loss_fn(preds, gt_scaled)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_history.append(loss.item())
            global_step += 1

            # ── Logging ────────────────────────────────────────────────────
            if global_step % LOG_EVERY == 0 or global_step == 1:
                with torch.no_grad():
                    metrics = compute_metrics(
                        preds[-1].detach(), gt_scaled,
                        max_disp=192.0 * scale,
                    )
                log.info(
                    f"Step {global_step:>4d}/{TOTAL_STEPS} | "
                    f"Loss={loss.item():>8.4f} | "
                    f"AvgErr={metrics['avg_err']:>6.2f}px | "
                    f"Bad-1.0={metrics['bad_1.0']:>5.1f}% | "
                    f"Bad-2.0={metrics['bad_2.0']:>5.1f}% | "
                    f"LR={LR:.0e}"
                )

    # ── Save final checkpoint ──────────────────────────────────────────────
    torch.save({
        'step'       : global_step,
        'model_state': model.state_dict(),
        'model_cfg'  : MODEL_CFG,
    }, CHECKPOINT_OUT)
    log.info("-" * 60)
    log.info(f"Checkpoint saved: {CHECKPOINT_OUT}")

    # ── Final summary ──────────────────────────────────────────────────────
    first_10_avg = sum(loss_history[:10])  / len(loss_history[:10])
    last_10_avg  = sum(loss_history[-10:]) / len(loss_history[-10:])

    log.info("")
    log.info("=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    log.info(f"  First 10 steps avg loss : {first_10_avg:.4f}")
    log.info(f"  Last  10 steps avg loss : {last_10_avg:.4f}")
    reduction = (1 - last_10_avg / first_10_avg) * 100
    log.info(f"  Loss reduction          : {reduction:.1f}%")
    log.info("")
    if reduction > 50:
        log.info("  ✅ Model is learning — loss dropped significantly")
        log.info("  ✅ Run visualize_results.py to see disparity maps")
    elif reduction > 20:
        log.info("  ⚠️  Partial learning — some improvement but not full overfit")
        log.info("  Try running with more steps")
    else:
        log.info("  ❌ Model not learning — check architecture or learning rate")


if __name__ == '__main__':
    main()