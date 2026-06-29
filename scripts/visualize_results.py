"""
scripts/visualize_results.py
==============================
Loads the overfit checkpoint and generates disparity visualizations
for all 6 sample pack images.

Output folder: outputs/visualizations/
  Each image produces one PNG:
    <frame_id>_comparison.png
      [ Left Image | Predicted Disparity | Ground Truth | Error Map ]

Run with:
    conda activate cleardepth
    cd C:\\Users\\ragha\\cleardepth
    python scripts/visualize_results.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cleardepth.data.sceneflow_sample import SceneFlowSampleDataset
from cleardepth.models.cleardepth_net import ClearDepthNet
from cleardepth.evaluation.metrics import compute_metrics, aggregate_metrics, format_metrics
from cleardepth.evaluation.visualize import save_comparison_figure

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_ROOT    = "C:/Users/ragha/Downloads/Sampler/Sampler"
SUBSETS        = ["Monkaa", "FlyingThings3D"]
CHECKPOINT_PATH = "checkpoints/overfit/overfit_final.pt"
OUTPUT_DIR     = "outputs/visualizations"
GRU_ITERS      = 3
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 60)
    print("ClearDepth — Disparity Visualization")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Checkpoint not found!")
        print("   Run scripts/overfit_train.py first.")
        return

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model_cfg = ckpt['model_cfg']
    print(f"Checkpoint from step: {ckpt['step']}")

    # ── Build model ────────────────────────────────────────────────────────
    model = ClearDepthNet(**model_cfg).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded: {model.param_count()['total']:,} parameters")

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = SceneFlowSampleDataset(
        root=SAMPLE_ROOT,
        subsets=SUBSETS,
        pass_name='RGB_cleanpass',
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} samples")
    print("-" * 60)

    # ── Run inference on every sample ──────────────────────────────────────
    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            left     = batch['left'].to(DEVICE)       # (1, 3, H, W)
            right    = batch['right'].to(DEVICE)      # (1, 3, H, W)
            gt       = batch['disparity'].to(DEVICE)  # (1, 1, H, W)
            filename = os.path.basename(batch['left_path'][0])
            frame_id = os.path.splitext(filename)[0]  # e.g. "0048"

            # Resize to training resolution before inference
            # Must match what the model was trained on
            TRAIN_H, TRAIN_W = 128, 256
            left_resized  = torch.nn.functional.interpolate(
                left,  size=(TRAIN_H, TRAIN_W), mode='bilinear',
                align_corners=False
            )
            right_resized = torch.nn.functional.interpolate(
                right, size=(TRAIN_H, TRAIN_W), mode='bilinear',
                align_corners=False
            )

            # Forward pass at training resolution
            preds = model(left_resized, right_resized,
                          n_iters=GRU_ITERS, test_mode=False)
            pred_disp = preds[-1]   # (1, 1, 32, 64) at 1/4 of 128x256

            # ── Upsample prediction to full resolution for visualization ──
            H_full, W_full = left.shape[2], left.shape[3]
            H_pred, W_pred = pred_disp.shape[2], pred_disp.shape[3]
            scale = W_full / W_pred

            pred_full = F.interpolate(
                pred_disp,
                size=(H_full, W_full),
                mode='bilinear',
                align_corners=False,
            ) * scale   # scale disparity values back to full resolution

            # ── Compute metrics at model output scale ──────────────────────
            scale_down = W_pred / W_full
            gt_scaled  = F.interpolate(
                gt, size=(H_pred, W_pred), mode='nearest'
            ) * scale_down

            metrics = compute_metrics(
                pred_disp, gt_scaled,
                max_disp=192.0 * scale_down,
            )
            all_metrics.append(metrics)

            # ── Save comparison figure ─────────────────────────────────────
            # Figure shows full resolution for visual clarity
            out_path = os.path.join(OUTPUT_DIR, f"{frame_id}_comparison.png")
            save_comparison_figure(
                img_left  = left[0],        # (3, H, W)
                disp_pred = pred_full[0],   # (1, H, W) upsampled
                disp_gt   = gt[0],          # (1, H, W) full res GT
                path      = out_path,
                max_disp  = gt[0].max().item(),
            )

            print(
                f"  [{i+1}/6] {frame_id} | "
                f"AvgErr={metrics['avg_err']:>6.2f}px | "
                f"Bad-1.0={metrics['bad_1.0']:>5.1f}% | "
                f"Saved: {os.path.basename(out_path)}"
            )

    # ── Print final aggregated metrics ────────────────────────────────────
    agg = aggregate_metrics(all_metrics)
    print("-" * 60)
    print("FINAL METRICS (averaged over all 6 images):")
    print(f"  {format_metrics(agg)}")
    print("-" * 60)
    print(f"\nVisualization images saved to: {OUTPUT_DIR}/")
    print("Each PNG contains 4 panels:")
    print("  [Left Image | Predicted Disparity | Ground Truth | Error Map]")
    print("\nOpen them in VS Code or Windows Photos to inspect results.")


if __name__ == '__main__':
    main()