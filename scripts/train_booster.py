"""
scripts/train_booster.py
========================
Fine-tune ClearDepthNet on the Booster transparent-object stereo dataset.

Key design choices aligned with the updated model:
  - ClearDepthNet now includes fuse_out_channels and ConvexUpsample;
    all parameters are loaded from configs/model/cleardepth.yaml.
  - training forward: returns a list of 1/4-scale disparity predictions.
  - GT is downsampled to 1/4 scale and divided by 4 before loss:
        gt_q = F.interpolate(gt, (H/4, W/4), mode='nearest') / 4
  - mask_00.png invalid pixels are zeroed in GT so SequenceLoss's
    (gt > 0) filter naturally excludes them.
  - Validation: last prediction at 1/4 scale vs downsampled GT (fast).
  - Checkpoints: best.pt (lowest val AvgErr), latest.pt, step_XXXXXX.pt.
  - MiT-B1 pretrained backbone weights optional via --pretrained flag.

Run:
    conda activate cleardepth
    cd /path/to/cleardepth
    python scripts/train_booster.py \\
        --data_root /data/booster_gt \\
        --batch_size 4 \\
        --max_steps 50000 \\
        --n_gru_iters 22 \\
        --pretrained \\
        --ckpt_dir checkpoints/booster

For T4 (16 GB VRAM) set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
before running to reduce memory fragmentation.
"""

import os
import sys
import time
import logging
import argparse
import platform
from pathlib import Path

# ── VRAM fragmentation fix (T4 / small-VRAM GPUs) ─────────────────────────
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ── Project root on path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from omegaconf import OmegaConf

from cleardepth.data.booster import BoosterDataset
from cleardepth.models.cleardepth_net import ClearDepthNet
from cleardepth.loss.sequence_loss import SequenceLoss
from cleardepth.evaluation.metrics import (
    compute_metrics, aggregate_metrics, format_metrics,
)

log = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Fine-tune ClearDepth on the Booster dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument('--data_root', required=True,
                   help='Path to Booster root, e.g. /data/booster_gt')
    p.add_argument('--height', type=int, default=360)
    p.add_argument('--width',  type=int, default=720)
    p.add_argument('--val_fraction', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_samples', type=int, default=None,
                   help='Cap dataset size (debugging)')

    # Training
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--num_workers',  type=int,   default=None,
                   help='DataLoader workers. Default: 0 on Windows, 4 on Linux.')
    p.add_argument('--max_steps',    type=int,   default=50_000)
    p.add_argument('--n_gru_iters',  type=int,   default=None,
                   help='GRU iterations per step. Defaults to config value.')
    p.add_argument('--lr',           type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip',    type=float, default=1.0)
    p.add_argument('--gamma',        type=float, default=0.9,
                   help='Sequence-loss exponential decay.')

    # Model
    p.add_argument('--config', default='configs/model/cleardepth.yaml',
                   help='OmegaConf model config file.')
    p.add_argument('--pretrained', action='store_true',
                   help='Load MiT-B1 ImageNet-1k weights before training.')

    # Checkpoints
    p.add_argument('--ckpt_dir',  default='checkpoints/booster')
    p.add_argument('--save_every', type=int, default=5_000,
                   help='Save checkpoint and run validation every N steps.')
    p.add_argument('--resume',    default=None,
                   help='Path to checkpoint to resume from.')

    # Logging
    p.add_argument('--log_every', type=int, default=100)

    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────

def build_model(cfg, n_gru_iters: int) -> ClearDepthNet:
    return ClearDepthNet(
        in_channels      = cfg.backbone.in_channels,
        embed_dim        = cfg.backbone.embed_dims[0],
        fuse_out_channels= cfg.backbone.fuse_out_channels,
        depths           = list(cfg.backbone.depths),
        num_heads        = list(cfg.backbone.num_heads),
        reduction_ratios = list(cfg.backbone.reduction_ratios),
        mlp_ratio        = cfg.backbone.mlp_ratio,
        drop_rate        = cfg.backbone.drop_rate,
        drop_path_rate   = cfg.backbone.drop_path_rate,
        hidden_dim       = cfg.gru.hidden_dim,
        n_gru_layers     = cfg.gru.n_gru_layers,
        n_gru_iters      = n_gru_iters,
        corr_levels      = cfg.correlation.num_levels,
        corr_radius      = cfg.correlation.radius,
        upsample_scale   = cfg.upsample.scale,
    )


def make_loader(dataset, batch_size: int, num_workers: int,
                shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = shuffle,
        persistent_workers = (num_workers > 0),
    )


def downsample_gt(gt: torch.Tensor, mask: torch.Tensor,
                  target_h: int, target_w: int):
    """
    Downsample GT disparity and mask to the 1/4-scale feature resolution.

    Args:
        gt, mask : (B, 1, H, W) at full training resolution.
        target_h, target_w : target spatial size (typically H/4, W/4).

    Returns:
        gt_q   : (B, 1, target_h, target_w) in 1/4-pixel units.
        mask_q : (B, 1, target_h, target_w) binary float.

    GT disparity is scaled by (target_w / W) so values remain in
    units of pixels at the *output* resolution.
    """
    W = gt.shape[-1]
    disp_scale = target_w / W                        # e.g. 0.25 for 1/4 scale
    gt_q = F.interpolate(
        gt, size=(target_h, target_w), mode='nearest'
    ) * disp_scale
    mask_q = F.interpolate(
        mask.float(), size=(target_h, target_w), mode='nearest'
    )
    return gt_q, mask_q


def save_ckpt(path: str, step: int, model, optimizer, scheduler,
              best_val_err: float):
    torch.save({
        'step'           : step,
        'model_state'    : model.state_dict(),
        'optimiser_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_err'   : best_val_err,
    }, path)


def load_ckpt(path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimiser_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    step         = ckpt.get('step', 0)
    best_val_err = ckpt.get('best_val_err', float('inf'))
    log.info(f"Resumed from step {step}: {path}")
    return step, best_val_err


# ── Validation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device: torch.device,
             n_gru_iters: int, max_disp: float) -> dict:
    """
    Quick validation using the last 1/4-scale GRU prediction.
    Uses test_mode=False to skip convex upsample overhead.
    """
    model.eval()
    all_metrics = []

    for batch in val_loader:
        left  = batch['left'].to(device)
        right = batch['right'].to(device)
        gt    = batch['disparity'].to(device)
        mask  = batch['mask'].to(device)

        # Forward: last prediction at 1/4 scale
        preds = model(left, right, n_iters=n_gru_iters, test_mode=False)
        pred_q = preds[-1]

        _, _, H_q, W_q = pred_q.shape
        gt_q, mask_q = downsample_gt(gt, mask, H_q, W_q)

        # Apply mask: zero out invalid pixels (excluded by > 0 in compute_metrics)
        gt_masked = gt_q * mask_q

        max_disp_q = max_disp * (W_q / gt.shape[-1])
        m = compute_metrics(pred_q, gt_masked, max_disp=max_disp_q)
        all_metrics.append(m)

    model.train()
    return aggregate_metrics(all_metrics)


# ── Main training loop ─────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    n_gru_iters = args.n_gru_iters or cfg.gru.n_gru_iters
    max_disp    = float(cfg.max_disp)
    log.info(f"n_gru_iters={n_gru_iters}  max_disp={max_disp}")

    # ── Workers ───────────────────────────────────────────────────────────
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if platform.system() == 'Windows' else 4
    log.info(f"DataLoader num_workers={num_workers}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = BoosterDataset(
        args.data_root,
        split        = 'train',
        height       = args.height,
        width        = args.width,
        augment      = True,
        max_samples  = args.max_samples,
        val_fraction = args.val_fraction,
        seed         = args.seed,
    )
    val_ds = BoosterDataset(
        args.data_root,
        split        = 'val',
        height       = args.height,
        width        = args.width,
        augment      = False,
        val_fraction = args.val_fraction,
        seed         = args.seed,
    )
    log.info(f"Train: {train_ds}")
    log.info(f"Val:   {val_ds}")

    train_loader = make_loader(train_ds, args.batch_size, num_workers, shuffle=True)
    val_loader   = make_loader(val_ds,   max(1, args.batch_size // 2),
                               num_workers, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg, n_gru_iters).to(device)
    counts = model.param_count()
    log.info(
        f"Parameters — feature_encoder={counts['feature_encoder']:,}  "
        f"context_encoder={counts['context_encoder']:,}  "
        f"gru={counts['gru']:,}  "
        f"convex_upsample={counts['convex_upsample']:,}  "
        f"total={counts['total']:,}"
    )

    # ── Pretrained weights ────────────────────────────────────────────────
    if args.pretrained:
        from cleardepth.models.backbone.pretrained import load_pretrained_encoders
        log.info("Loading MiT-B1 ImageNet-1k pretrained weights ...")
        load_pretrained_encoders(model.feature_encoder, model.context_encoder)

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_fn = SequenceLoss(gamma=args.gamma, max_disp=max_disp / 4.0)

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)

    warmup_steps = max(1, int(0.05 * args.max_steps))
    scheduler = OneCycleLR(
        optimizer,
        max_lr       = args.lr,
        total_steps  = args.max_steps + 1,
        pct_start    = warmup_steps / args.max_steps,
        anneal_strategy = 'cos',
    )

    # ── Resume ────────────────────────────────────────────────────────────
    global_step   = 0
    best_val_err  = float('inf')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.resume:
        global_step, best_val_err = load_ckpt(
            args.resume, model, optimizer, scheduler, device
        )
        # Fast-forward scheduler to the resumed step
        for _ in range(global_step):
            scheduler.step()

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    step_times: list = []

    log.info(f"Training for {args.max_steps} steps ...")
    log.info("-" * 70)

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            t0 = time.perf_counter()

            left  = batch['left'].to(device)
            right = batch['right'].to(device)
            gt    = batch['disparity'].to(device)   # (B, 1, H, W) full-res
            mask  = batch['mask'].to(device)         # (B, 1, H, W) binary

            # ── Forward pass ──────────────────────────────────────────────
            preds = model(left, right, n_iters=n_gru_iters, test_mode=False)

            # ── GT alignment: 1/4 scale + divide values by 4 ─────────────
            _, _, H_q, W_q = preds[0].shape
            gt_q, mask_q   = downsample_gt(gt, mask, H_q, W_q)

            # Mask out invalid pixels by zeroing GT there.
            # SequenceLoss filters valid = (gt > 0), so zeros are excluded.
            gt_masked = gt_q * mask_q

            # ── Sequence loss ─────────────────────────────────────────────
            loss = loss_fn(preds, gt_masked)

            # ── Backward ──────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step_time = time.perf_counter() - t0
            step_times.append(step_time)
            global_step += 1

            # ── Logging ───────────────────────────────────────────────────
            if global_step % args.log_every == 0:
                avg_ms  = 1000 * sum(step_times[-50:]) / len(step_times[-50:])
                lr_now  = scheduler.get_last_lr()[0]
                steps_remaining = args.max_steps - global_step
                eta_s   = steps_remaining * (avg_ms / 1000.0)
                eta_h   = eta_s / 3600.0

                log.info(
                    f"Step {global_step:>7d}/{args.max_steps} | "
                    f"Loss={loss.item():.4f} | "
                    f"LR={lr_now:.2e} | "
                    f"{avg_ms:.0f}ms/step | "
                    f"ETA={eta_h:.1f}h"
                )

            # ── Checkpoint + validation ───────────────────────────────────
            if global_step % args.save_every == 0:
                # Step checkpoint
                ckpt_path = os.path.join(
                    args.ckpt_dir, f'step_{global_step:07d}.pt'
                )
                save_ckpt(ckpt_path, global_step, model, optimizer,
                          scheduler, best_val_err)
                log.info(f"Saved: {ckpt_path}")

                # Latest checkpoint (overwrites)
                save_ckpt(
                    os.path.join(args.ckpt_dir, 'latest.pt'),
                    global_step, model, optimizer, scheduler, best_val_err,
                )

                # Validation
                log.info("Running validation ...")
                val_metrics = validate(
                    model, val_loader, device, n_gru_iters, max_disp
                )
                val_err = val_metrics.get('avg_err', float('inf'))
                log.info(
                    f"[Val @ step {global_step}]  {format_metrics(val_metrics)}"
                )

                # Best checkpoint
                if val_err < best_val_err:
                    best_val_err = val_err
                    save_ckpt(
                        os.path.join(args.ckpt_dir, 'best.pt'),
                        global_step, model, optimizer, scheduler, best_val_err,
                    )
                    log.info(
                        f"New best AvgErr={best_val_err:.4f} — saved best.pt"
                    )

    # ── Final save ────────────────────────────────────────────────────────
    save_ckpt(
        os.path.join(args.ckpt_dir, 'latest.pt'),
        global_step, model, optimizer, scheduler, best_val_err,
    )
    log.info(f"Training complete. Best val AvgErr={best_val_err:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s | %(levelname)s | %(message)s',
        datefmt= '%H:%M:%S',
    )
    train(parse_args())
