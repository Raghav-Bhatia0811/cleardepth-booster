"""
Evaluation Metrics
==================
AvgErr, RMS, and Bad-N metrics for stereo disparity evaluation.

Paper reference: Section IV (Experiments), Table I
  Reports AvgErr, RMS, Bad-0.5, Bad-1.0, Bad-2.0, Bad-3.0
  on the SynClearDepth test set.
"""

import torch
from typing import Dict, List


def compute_metrics(
    disp_pred: torch.Tensor,
    disp_gt: torch.Tensor,
    max_disp: float = 192.0,
    bad_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for one batch.

    Args:
        disp_pred       : Predicted disparity (B, 1, H, W).
        disp_gt         : Ground truth disparity (B, 1, H, W).
                          Invalid pixels have value <= 0 or > max_disp.
        max_disp        : Maximum valid disparity (used for masking).
        bad_thresholds  : Error thresholds for Bad-N metrics (pixels).

    Returns:
        Dictionary with keys:
          'avg_err'  : Mean absolute error (pixels)
          'rms'      : Root mean squared error (pixels)
          'bad_0.5'  : % pixels with error > 0.5
          'bad_1.0'  : % pixels with error > 1.0
          'bad_2.0'  : % pixels with error > 2.0
          'bad_3.0'  : % pixels with error > 3.0
    """
    # Valid pixel mask
    valid = (disp_gt > 0) & (disp_gt < max_disp)

    if valid.sum() == 0:
        # No valid pixels — return zeros to avoid division by zero
        metrics = {'avg_err': 0.0, 'rms': 0.0}
        for t in bad_thresholds:
            metrics[f'bad_{t}'] = 0.0
        return metrics

    # Error at valid pixels only
    error = (disp_pred - disp_gt).abs()[valid]    # (num_valid,)
    error_sq = ((disp_pred - disp_gt) ** 2)[valid]

    results = {}

    # AvgErr: mean absolute error
    results['avg_err'] = error.mean().item()

    # RMS: root mean squared error
    results['rms'] = error_sq.mean().sqrt().item()

    # Bad-N: percentage of pixels exceeding each threshold
    n_valid = valid.sum().item()
    for t in bad_thresholds:
        bad_count = (error > t).sum().item()
        results[f'bad_{t}'] = 100.0 * bad_count / n_valid

    return results


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average metrics across multiple batches.

    Args:
        metrics_list : List of metric dicts, one per batch.

    Returns:
        Dict with mean value for each metric key.
    """
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    return {
        k: sum(m[k] for m in metrics_list) / len(metrics_list)
        for k in keys
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as a human-readable string for logging."""
    parts = []
    if 'avg_err' in metrics:
        parts.append(f"AvgErr={metrics['avg_err']:.4f}")
    if 'rms' in metrics:
        parts.append(f"RMS={metrics['rms']:.4f}")
    for t in [0.5, 1.0, 2.0, 4.0]:
        key = f'bad_{t}'
        if key in metrics:
            parts.append(f"Bad-{t}={metrics[key]:.2f}%")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W = 2, 16, 32

    # Simulate a prediction close to ground truth
    gt   = torch.rand(B, 1, H, W) * 100
    pred = gt + torch.randn_like(gt) * 2.0   # ~2px noise

    # Mark some pixels as invalid (value = 0)
    gt[:, :, :2, :] = 0.0

    metrics = compute_metrics(pred, gt)
    print("Metrics:")
    print(f"  {format_metrics(metrics)}")

    # Sanity checks
    assert metrics['avg_err'] >= 0
    assert metrics['rms']     >= metrics['avg_err']   # RMS >= mean
    assert 0 <= metrics['bad_1.0'] <= 100

    # Test aggregation
    m2 = compute_metrics(pred * 0.9, gt)
    agg = aggregate_metrics([metrics, m2])
    assert 'avg_err' in agg
    print(f"  Aggregated: {format_metrics(agg)}")

    print("✅ Metrics smoke test passed.")  