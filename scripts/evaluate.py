"""
ClearDepth Evaluation Script
==============================
Loads a trained checkpoint, runs inference on a test set,
computes all metrics, and saves visualizations.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/step_0300000.pt \
        --data_root  /path/to/SceneFlow \
        --output_dir results/eval \
        --n_samples  100 \
        --save_viz

For smoke-test (no real data needed):
    python scripts/evaluate.py --smoke_test
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make sure the cleardepth package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cleardepth.models.cleardepth_net import ClearDepthNet
from cleardepth.evaluation.metrics import compute_metrics, aggregate_metrics, format_metrics
from cleardepth.evaluation.visualize import save_comparison_figure
from cleardepth.training.trainer import SyntheticStereoDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default model config — matches training config
# ---------------------------------------------------------------------------

DEFAULT_MODEL_CFG = dict(
    embed_dim=64,
    depths=[2, 2, 2, 2],
    num_heads=[1, 2, 4, 8],
    reduction_ratios=[8, 4, 2, 1],
    hidden_dim=128,
    n_gru_iters=22,
    corr_levels=4,
    corr_radius=4,
)

SMOKE_MODEL_CFG = dict(
    embed_dim=32,
    depths=[1, 1, 1, 1],
    num_heads=[1, 2, 4, 8],
    reduction_ratios=[8, 4, 2, 1],
    hidden_dim=64,
    n_gru_iters=4,
    corr_levels=4,
    corr_radius=4,
)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: ClearDepthNet,
    loader: DataLoader,
    device: torch.device,
    n_iters: int,
    max_disp: float,
    output_dir: str,
    save_viz: bool,
    n_viz: int = 8,
) -> dict:
    """
    Run evaluation over the full loader.

    Args:
        model      : Trained ClearDepthNet in eval mode.
        loader     : DataLoader yielding {'left', 'right', 'disparity'}.
        device     : Torch device.
        n_iters    : GRU iterations for inference.
        max_disp   : Maximum valid disparity for metric masking.
        output_dir : Directory to save visualizations.
        save_viz   : Whether to save comparison figures.
        n_viz      : How many samples to visualize.

    Returns:
        Aggregated metrics dict.
    """
    model.eval()
    all_metrics = []
    viz_count   = 0

    if save_viz:
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        left  = batch['left'].to(device)
        right = batch['right'].to(device)
        gt    = batch['disparity'].to(device)

        # Forward pass — inference mode returns final prediction only
        pred = model(left, right, n_iters=n_iters, test_mode=True)
        # pred: (B, 1, H_feat, W_feat) at 1/4 scale

        # Upsample prediction back to full image resolution for metrics
        _, _, H_full, W_full = gt.shape
        pred_full = F.interpolate(
            pred, size=(H_full, W_full),
            mode='bilinear', align_corners=False,
        )
        # Scale disparity values back up (1/4 scale → full scale)
        scale = W_full / pred.shape[-1]
        pred_full = pred_full * scale

        # Compute metrics for this batch
        metrics = compute_metrics(pred_full, gt, max_disp=max_disp)
        all_metrics.append(metrics)

        # Save visualizations for the first n_viz samples
        if save_viz and viz_count < n_viz:
            for i in range(left.shape[0]):
                if viz_count >= n_viz:
                    break
                save_comparison_figure(
                    img_left=left[i].cpu(),
                    disp_pred=pred_full[i].cpu(),
                    disp_gt=gt[i].cpu(),
                    path=os.path.join(viz_dir, f'sample_{viz_count:04d}.png'),
                    max_disp=max_disp,
                )
                viz_count += 1

        if (batch_idx + 1) % 10 == 0:
            log.info(
                f"Batch {batch_idx+1}/{len(loader)} | "
                f"{format_metrics(metrics)}"
            )

    return aggregate_metrics(all_metrics)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ClearDepth Evaluation')
    parser.add_argument('--checkpoint',  type=str,  default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data_root',   type=str,  default=None,
                        help='Path to dataset root directory')
    parser.add_argument('--output_dir',  type=str,  default='results/eval',
                        help='Directory to save results and visualizations')
    parser.add_argument('--n_samples',   type=int,  default=None,
                        help='Max samples to evaluate (None = full test set)')
    parser.add_argument('--n_iters',     type=int,  default=22,
                        help='GRU refinement iterations for inference')
    parser.add_argument('--batch_size',  type=int,  default=1,
                        help='Evaluation batch size')
    parser.add_argument('--max_disp',    type=float,default=192.0,
                        help='Maximum valid disparity for metric masking')
    parser.add_argument('--save_viz',    action='store_true',
                        help='Save comparison visualization figures')
    parser.add_argument('--n_viz',       type=int,  default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--smoke_test',  action='store_true',
                        help='Run with synthetic data (no real dataset needed)')
    parser.add_argument('--device',      type=str,  default='cuda',
                        help='Device: cuda or cpu')
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == 'cuda'
        else 'cpu'
    )
    log.info(f"Evaluating on: {device}")

    # ── Output directory ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────
    cfg = SMOKE_MODEL_CFG if args.smoke_test else DEFAULT_MODEL_CFG
    model = ClearDepthNet(**cfg).to(device)

    if args.checkpoint is not None:
        log.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device,
                          weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        log.info(f"Resumed from step {ckpt.get('step', '?')}")
    else:
        log.warning("No checkpoint provided — evaluating with random weights")

    params = model.param_count()
    log.info(f"Model parameters: {params['total']:,}")

    # ── Dataset ───────────────────────────────────────────────────────────
    if args.smoke_test:
        log.info("SMOKE TEST MODE — using synthetic data")
        n = args.n_samples or 16
        dataset = SyntheticStereoDataset(
            length=n, height=64, width=128, max_disp=64.0
        )
    else:
        if args.data_root is None:
            parser.error("--data_root is required unless --smoke_test is set")
        from cleardepth.data.sceneflow import SceneFlowDataset
        dataset = SceneFlowDataset(
            root=args.data_root,
            split='test',
            height=360, width=720,
            augment=False,
        )
        if args.n_samples is not None:
            from torch.utils.data import Subset
            dataset = Subset(dataset, range(min(args.n_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    log.info(f"Evaluating on {len(dataset)} samples")

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = evaluate(
        model=model,
        loader=loader,
        device=device,
        n_iters=args.n_iters if not args.smoke_test else cfg['n_gru_iters'],
        max_disp=args.max_disp,
        output_dir=args.output_dir,
        save_viz=args.save_viz,
        n_viz=args.n_viz,
    )

    # ── Print results ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(format_metrics(results))
    for k, v in results.items():
        log.info(f"  {k:12s}: {v:.4f}")
    log.info("=" * 60)

    # Save results to text file
    results_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(results_path, 'w') as f:
        f.write(format_metrics(results) + '\n')
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    log.info(f"Results saved to: {results_path}")

    if args.save_viz:
        log.info(f"Visualizations saved to: "
                 f"{os.path.join(args.output_dir, 'visualizations')}")


if __name__ == '__main__':
    main()