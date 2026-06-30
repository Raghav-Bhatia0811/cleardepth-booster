"""
scripts/evaluate_sceneflow.py
===============================
Evaluate a Scene Flow (Monkaa) pretraining checkpoint on the val split.

Unlike evaluate_booster.py (which evaluates at full resolution via plain
bilinear x4 upsampling under test_mode=True), this script evaluates at
1/4 scale — the native GRU output resolution — so that Scene Flow
pretraining metrics are directly comparable to the training-time
validation metrics logged by pretrain_sceneflow.py, without the
upsampling step's interpolation affecting the numbers.

What this script does:
  1. Loads a checkpoint (best.pt by default).
  2. Runs test_mode=False inference → last 1/4-scale disparity prediction.
  3. Downsamples GT to the same 1/4 scale (nearest + divide by 4), same
     convention as training.
  4. Computes paper metrics at 1/4 scale:
       AvgErr | RMS | Bad-0.5 | Bad-1.0 | Bad-2.0 | Bad-4.0
  5. Saves 4-panel PNG figures per sample — pred/GT bilinearly upsampled
     ×4 for DISPLAY ONLY (metrics are unaffected, computed pre-upsample).
  6. Writes evaluation_results.txt with per-scene and averaged metrics.

Run:
    python scripts/evaluate_sceneflow.py \\
        --ckpt /data/sceneflow_checkpoints/best.pt \\
        --data_root /data/monkaa \\
        --output_dir eval_results/sceneflow
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

from cleardepth.data.sceneflow_monkaa import SceneFlowMonkaaDataset
from cleardepth.models.cleardepth_net import ClearDepthNet
from cleardepth.evaluation.metrics import (
    compute_metrics, aggregate_metrics, format_metrics,
)

log = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate ClearDepth Scene Flow pretraining checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ckpt',        required=True,
                   help='Path to checkpoint, e.g. /data/sceneflow_checkpoints/best.pt')
    p.add_argument('--data_root',   required=True,
                   help='Path to Monkaa root, e.g. /data/monkaa')
    p.add_argument('--output_dir',  default='eval_results/sceneflow')
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


def downsample_gt(gt: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Same convention as pretrain_sceneflow.py: nearest + scale by width ratio."""
    W = gt.shape[-1]
    disp_scale = target_w / W
    return F.interpolate(gt, size=(target_h, target_w), mode='nearest') * disp_scale


# ── Visualisation ──────────────────────────────────────────────────────────

def _tensor_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) normalised [-1,1] tensor → (H, W, 3) uint8."""
    arr = ((img_tensor.cpu().float() + 1.0) / 2.0).clamp(0, 1)
    return (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _disp_to_color(disp: torch.Tensor, vmin: float = None,
                   vmax: float = None) -> np.ndarray:
    """Map (H, W) disparity tensor → (H, W, 3) uint8 using 'magma' colormap."""
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
    return (cm.magma(norm)[:, :, :3] * 255).astype(np.uint8)


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
    pred_q: torch.Tensor,
    gt_q: torch.Tensor,
    upsample_scale: int,
    scene: str,
    frame: str,
):
    """
    Save a 4-panel figure: Left RGB | Predicted Disparity | GT | Error Map.

    pred_q and gt_q are at 1/4-scale (the resolution metrics were computed
    at). They are bilinearly upsampled by upsample_scale here PURELY for
    side-by-side display next to the full-resolution left image — this
    does not affect the reported metrics, which are computed beforehand
    at native 1/4 scale.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping figure generation.")
        return

    # Upsample pred/gt ×scale for display only (disparity values are NOT
    # rescaled here — only the spatial grid is enlarged for visual parity
    # with the full-res left image; values stay in their native 1/4-scale
    # pixel units, consistent within the pred/gt pair for fair comparison).
    pred_disp = F.interpolate(
        pred_q.unsqueeze(0), scale_factor=upsample_scale,
        mode='bilinear', align_corners=False,
    ).squeeze(0)
    gt_disp = F.interpolate(
        gt_q.unsqueeze(0), scale_factor=upsample_scale,
        mode='bilinear', align_corners=False,
    ).squeeze(0)

    gt_np = gt_disp.squeeze().cpu().float().numpy()
    valid_gt = gt_np[gt_np > 0]
    vmin = float(np.percentile(valid_gt, 2))  if len(valid_gt) > 0 else 0
    vmax = float(np.percentile(valid_gt, 98)) if len(valid_gt) > 0 else 1

    rgb      = _tensor_to_rgb(left)
    pred_col = _disp_to_color(pred_disp, vmin=vmin, vmax=vmax)
    gt_col   = _disp_to_color(gt_disp,   vmin=vmin, vmax=vmax)

    # Error map computed at the upsampled display resolution (visual only)
    err = (pred_disp.squeeze() - gt_disp.squeeze()).abs()
    error_col = _error_to_color(err, max_err=5.0)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ['Left RGB', 'Predicted Disparity (×4 upsampled)',
              'Ground Truth (×4 upsampled)', 'Error Map (cap 5px)']
    images = [rgb, pred_col, gt_col, error_col]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig.suptitle(f'{scene} / {frame}', fontsize=12)
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
    upsample_scale = int(cfg.upsample.scale)
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

    dataset = SceneFlowMonkaaDataset(
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

    log.info("Running inference (metrics at 1/4 scale) ...")
    for batch_idx, batch in enumerate(loader):
        left  = batch['left'].to(device)
        right = batch['right'].to(device)
        gt    = batch['disparity'].to(device)   # (B, 1, H, W) full-res

        with torch.no_grad():
            # test_mode=False → list of 1/4-scale predictions; take last
            preds = model(left, right, n_iters=n_gru_iters, test_mode=False)
            pred_q = preds[-1]   # (B, 1, H/4, W/4)

        _, _, H_q, W_q = pred_q.shape
        gt_q = downsample_gt(gt, H_q, W_q)
        max_disp_q = max_disp * (W_q / gt.shape[-1])

        B = pred_q.shape[0]
        for b in range(B):
            scene = batch['scene'][b]
            frame = batch['frame'][b]

            pred_b = pred_q[b:b+1]   # (1, 1, H/4, W/4)
            gt_b   = gt_q[b:b+1]

            m = compute_metrics(pred_b, gt_b, max_disp=max_disp_q)
            all_metrics.append(m)
            per_sample_rows.append((scene, frame, m))

            log.info(f"  {scene}/{frame}  {format_metrics(m)}")

            # ── Visualisation (upsampled ×4 for display only) ─────────────
            if args.max_vis is None or vis_count < args.max_vis:
                fig_path = os.path.join(vis_dir, f"{scene}_{frame}.png")
                save_figure(fig_path, left[b], pred_b.squeeze(0),
                            gt_b.squeeze(0), upsample_scale, scene, frame)
                vis_count += 1

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg = aggregate_metrics(all_metrics)
    log.info("")
    log.info("=" * 60)
    log.info(f"FINAL RESULTS ({len(all_metrics)} samples, 1/4-scale metrics)")
    log.info(f"  {format_metrics(agg)}")
    log.info("=" * 60)

    # ── Write results file ─────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"ClearDepth — Scene Flow (Monkaa) Evaluation\n")
        f.write(f"Checkpoint : {args.ckpt}  (step {step_saved})\n")
        f.write(f"Split      : {args.split}\n")
        f.write(f"Resolution : {args.height}×{args.width}  "
                f"(metrics computed at 1/4 scale: "
                f"{args.height//4}×{args.width//4})\n")
        f.write(f"GRU iters  : {n_gru_iters}\n")
        f.write(f"Samples    : {len(all_metrics)}\n")
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("AVERAGED METRICS (1/4 scale)\n")
        f.write("=" * 60 + "\n")
        for k, v in agg.items():
            f.write(f"  {k:>12s}  {v:.4f}\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("PER-SAMPLE METRICS\n")
        f.write("-" * 60 + "\n")

        metric_keys = list(all_metrics[0].keys()) if all_metrics else []
        header = f"{'scene':<25} {'frame':<8}  " + \
                 "  ".join(f"{k:>10}" for k in metric_keys)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for scene, frame, m in per_sample_rows:
            row = f"{scene:<25} {frame:<8}  " + \
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
