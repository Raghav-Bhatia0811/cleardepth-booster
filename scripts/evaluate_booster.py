"""
scripts/evaluate_booster.py
============================
Evaluate a trained ClearDepthNet checkpoint on the Booster val split.

What this script does:
  1. Loads a checkpoint (best.pt by default) onto CPU/CUDA.
  2. Runs test_mode=True inference → full-resolution disparity (H×W).
     With the updated model this uses ConvexUpsample instead of bilinear,
     so predictions are already at the target resolution (360×720).
  3. Computes paper metrics at full resolution against the 360×720 GT:
       AvgErr | RMS | Bad-0.5 | Bad-1.0 | Bad-2.0 | Bad-4.0
  4. Saves 4-panel PNG figures per sample (left | pred | GT | error).
  5. Writes evaluation_results.txt with per-scene and averaged metrics.

Aligned with updated ClearDepthNet:
  - test_mode=True returns (B, 1, H, W) at the full training resolution.
  - No manual upsampling needed; compare directly with full-res GT.
  - n_gru_iters uses cfg.gru.n_gru_iters_eval (=32) unless overridden.

Run:
    python scripts/evaluate_booster.py \\
        --ckpt checkpoints/booster/best.pt \\
        --data_root /data/booster_gt \\
        --output_dir eval_results/booster
"""

import os
import sys
import logging
import argparse
import platform
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from cleardepth.data.booster import BoosterDataset
from cleardepth.models.cleardepth_net import ClearDepthNet
from cleardepth.evaluation.metrics import (
    compute_metrics, aggregate_metrics, format_metrics,
)

log = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate ClearDepth on the Booster val split',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ckpt',        required=True,
                   help='Path to checkpoint file, e.g. checkpoints/booster/best.pt')
    p.add_argument('--data_root',   required=True,
                   help='Path to Booster root, e.g. /data/booster_gt')
    p.add_argument('--output_dir',  default='eval_results/booster')
    p.add_argument('--split',       default='val', choices=['train', 'val'])
    p.add_argument('--height',      type=int, default=360)
    p.add_argument('--width',       type=int, default=720)
    p.add_argument('--val_fraction',type=float, default=0.15)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--config',      default='configs/model/cleardepth.yaml')
    p.add_argument('--n_gru_iters', type=int, default=None,
                   help='GRU iterations at eval. Defaults to config n_gru_iters_eval.')
    p.add_argument('--max_vis',     type=int, default=None,
                   help='Max figures to save (None = save all).')
    p.add_argument('--batch_size',  type=int, default=1)
    p.add_argument('--num_workers', type=int, default=None)
    return p.parse_args()


# ── Model builder ──────────────────────────────────────────────────────────

def build_model(cfg, n_gru_iters: int) -> ClearDepthNet:
    return ClearDepthNet(
        in_channels      = cfg.backbone.in_channels,
        embed_dim        = cfg.backbone.embed_dims[0],
        fuse_out_channels= cfg.backbone.fuse_out_channels,
        depths           = list(cfg.backbone.depths),
        num_heads        = list(cfg.backbone.num_heads),
        reduction_ratios = list(cfg.backbone.reduction_ratios),
        mlp_ratio        = cfg.backbone.mlp_ratio,
        drop_rate        = 0.0,           # no dropout at eval
        drop_path_rate   = 0.0,
        hidden_dim       = cfg.gru.hidden_dim,
        n_gru_layers     = cfg.gru.n_gru_layers,
        n_gru_iters      = n_gru_iters,
        corr_levels      = cfg.correlation.num_levels,
        corr_radius      = cfg.correlation.radius,
        upsample_scale   = cfg.upsample.scale,
    )


# ── Visualisation ──────────────────────────────────────────────────────────

def _tensor_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) normalised [-1,1] tensor → (H, W, 3) uint8."""
    arr = ((img_tensor.cpu().float() + 1.0) / 2.0).clamp(0, 1)
    return (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _disp_to_color(disp: torch.Tensor, vmin: float = None,
                   vmax: float = None) -> np.ndarray:
    """
    Map (1, H, W) or (H, W) disparity tensor → (H, W, 3) uint8 RGBA.
    Uses 'magma' colormap: bright = high disparity (near), dark = far.
    """
    try:
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib is required for visualisation: pip install matplotlib")

    arr = disp.squeeze().cpu().float().numpy()
    if vmin is None:
        vmin = float(np.percentile(arr[arr > 0], 5)) if (arr > 0).any() else 0
    if vmax is None:
        vmax = float(np.percentile(arr[arr > 0], 95)) if (arr > 0).any() else 1

    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    colored = (cm.magma(norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def _error_to_color(error: torch.Tensor, max_err: float = 5.0) -> np.ndarray:
    """Map (H, W) absolute-error tensor → (H, W, 3) uint8 using 'hot' colormap."""
    try:
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib is required for visualisation")

    arr = error.squeeze().cpu().float().numpy()
    norm = np.clip(arr / max_err, 0, 1)
    return (cm.hot(norm)[:, :, :3] * 255).astype(np.uint8)


def save_figure(
    output_path: str,
    left: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    scene: str,
    illum: str,
):
    """
    Save a 4-panel figure: Left RGB | Predicted Disp | GT Disp | Error Map.

    All panels are at full training resolution (e.g. 360×720).
    Error map shows absolute error, capped at 5 pixels for contrast.
    Invalid pixels (mask=0) are shown in black on GT and error panels.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping figure generation.")
        return

    mask_np = mask.squeeze().cpu().float().numpy()

    # Shared disparity scale between pred and GT for fair comparison
    gt_np = gt.squeeze().cpu().float().numpy()
    valid_gt = gt_np[mask_np > 0.5]
    vmin = float(np.percentile(valid_gt, 2))  if len(valid_gt) > 0 else 0
    vmax = float(np.percentile(valid_gt, 98)) if len(valid_gt) > 0 else 1

    rgb      = _tensor_to_rgb(left)
    pred_col = _disp_to_color(pred, vmin=vmin, vmax=vmax)
    gt_col   = _disp_to_color(gt,   vmin=vmin, vmax=vmax)

    # Black-out invalid pixels on GT panel
    gt_col[mask_np < 0.5] = 0

    # Error map — compute only on valid pixels, others black
    err = (pred.squeeze() - gt.squeeze()).abs()
    err_masked = err.clone()
    err_masked[mask.squeeze() < 0.5] = 0
    error_col = _error_to_color(err_masked, max_err=5.0)
    error_col[mask_np < 0.5] = 0

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ['Left RGB', 'Predicted Disparity', 'Ground Truth', 'Error Map (cap 5px)']
    images = [rgb, pred_col, gt_col, error_col]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig.suptitle(f'{scene} / {illum}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Main evaluation loop ───────────────────────────────────────────────────

def evaluate(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Config & model ────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    n_gru_iters = args.n_gru_iters or cfg.gru.n_gru_iters_eval
    max_disp    = float(cfg.max_disp)
    log.info(f"n_gru_iters (eval)={n_gru_iters}  max_disp={max_disp}")

    model = build_model(cfg, n_gru_iters).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    step_saved = ckpt.get('step', '?')
    log.info(f"Loaded checkpoint from step {step_saved}: {args.ckpt}")
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if platform.system() == 'Windows' else 4

    dataset = BoosterDataset(
        args.data_root,
        split        = args.split,
        height       = args.height,
        width        = args.width,
        augment      = False,
        val_fraction = args.val_fraction,
        seed         = args.seed,
    )
    log.info(f"Dataset: {dataset}")

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    # ── Output directories ─────────────────────────────────────────────────
    vis_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(vis_dir, exist_ok=True)

    # ── Inference loop ────────────────────────────────────────────────────
    all_metrics: list = []
    per_sample_rows: list = []
    vis_count = 0

    log.info("Running inference ...")
    for batch_idx, batch in enumerate(loader):
        left  = batch['left'].to(device)
        right = batch['right'].to(device)
        gt    = batch['disparity'].to(device)   # (B, 1, H, W) full-res
        mask  = batch['mask'].to(device)         # (B, 1, H, W) binary

        with torch.no_grad():
            # test_mode=True → full-resolution disparity (B, 1, H, W)
            # thanks to ConvexUpsample; no manual rescaling needed
            pred = model(left, right, n_iters=n_gru_iters, test_mode=True)

        B = pred.shape[0]
        for b in range(B):
            scene = batch['scene'][b]
            illum = batch['illum'][b]

            pred_b = pred[b:b+1]        # (1, 1, H, W)
            gt_b   = gt[b:b+1]
            mask_b = mask[b:b+1]

            # Apply mask: invalid pixels → 0 (excluded by compute_metrics)
            gt_masked = gt_b * mask_b

            m = compute_metrics(pred_b, gt_masked, max_disp=max_disp)
            all_metrics.append(m)
            per_sample_rows.append((scene, illum, m))

            log.info(
                f"  {scene}/{illum}  {format_metrics(m)}"
            )

            # ── Visualisation ─────────────────────────────────────────────
            if args.max_vis is None or vis_count < args.max_vis:
                fig_path = os.path.join(vis_dir, f"{scene}_{illum}.png")
                save_figure(fig_path, left[b], pred_b.squeeze(0),
                            gt_b.squeeze(0), mask_b.squeeze(0), scene, illum)
                vis_count += 1

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg = aggregate_metrics(all_metrics)
    log.info("")
    log.info("=" * 60)
    log.info(f"FINAL RESULTS ({len(all_metrics)} samples)")
    log.info(f"  {format_metrics(agg)}")
    log.info("=" * 60)

    # ── Write results file ─────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"ClearDepth — Booster Evaluation\n")
        f.write(f"Checkpoint : {args.ckpt}  (step {step_saved})\n")
        f.write(f"Split      : {args.split}\n")
        f.write(f"Resolution : {args.height}×{args.width}\n")
        f.write(f"GRU iters  : {n_gru_iters}\n")
        f.write(f"Samples    : {len(all_metrics)}\n")
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("AVERAGED METRICS\n")
        f.write("=" * 60 + "\n")
        for k, v in agg.items():
            f.write(f"  {k:>12s}  {v:.4f}\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("PER-SAMPLE METRICS\n")
        f.write("-" * 60 + "\n")

        # Header
        metric_keys = list(all_metrics[0].keys()) if all_metrics else []
        header = f"{'scene':<25} {'illum':<8}  " + \
                 "  ".join(f"{k:>10}" for k in metric_keys)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for scene, illum, m in per_sample_rows:
            row = f"{scene:<25} {illum:<8}  " + \
                  "  ".join(f"{m[k]:>10.4f}" for k in metric_keys)
            f.write(row + "\n")

    log.info(f"Results written to: {results_path}")
    log.info(f"Figures saved to:   {vis_dir}/  ({vis_count} files)")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s | %(levelname)s | %(message)s',
        datefmt= '%H:%M:%S',
    )
    evaluate(parse_args())
