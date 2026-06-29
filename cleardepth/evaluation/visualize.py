"""
Disparity Visualization Utilities
===================================
Converts disparity tensors to colour images and comparison figures.

Colour convention (jet colormap):
  Warm (red/yellow) = small disparity = FAR objects
  Cool (blue)       = large disparity = NEAR objects

  This is the standard convention in stereo depth papers.

Functions:
  disp_to_color  : Single disparity map → RGB colour image (numpy)
  error_to_color : Absolute error map → RGB colour image (numpy)
  make_comparison_figure : Side-by-side: image | pred | gt | error
  save_disp_image : Save a disparity tensor directly to a PNG file
"""

import numpy as np
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# Core colorization
# ---------------------------------------------------------------------------

def disp_to_color(
    disp: torch.Tensor,
    vmin: float = None,
    vmax: float = None,
    colormap: str = 'plasma',
) -> np.ndarray:
    """
    Convert a disparity map to an RGB colour image using a colormap.

    Args:
        disp     : Disparity tensor (1, H, W) or (H, W). Float, pixels.
        vmin     : Min disparity value for normalisation.
                   If None, uses disp.min().
        vmax     : Max disparity value for normalisation.
                   If None, uses disp.max().
        colormap : Matplotlib colormap name ('plasma', 'jet', 'magma').

    Returns:
        rgb : (H, W, 3) uint8 numpy array in range [0, 255].
    """
    import matplotlib.cm as cm

    # Handle (1, H, W) or (H, W)
    if disp.dim() == 3:
        disp = disp.squeeze(0)

    disp_np = disp.detach().cpu().float().numpy()

    # Normalise to [0, 1]
    v_min = disp_np.min() if vmin is None else vmin
    v_max = disp_np.max() if vmax is None else vmax
    if v_max - v_min < 1e-6:
        v_max = v_min + 1.0   # avoid division by zero on flat maps

    disp_norm = np.clip((disp_np - v_min) / (v_max - v_min), 0.0, 1.0)

    # Apply colormap: returns (H, W, 4) RGBA in [0, 1]
    import matplotlib
    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(disp_norm)

    # Convert to uint8 RGB
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def error_to_color(
    pred: torch.Tensor,
    gt: torch.Tensor,
    max_error: float = 5.0,
) -> np.ndarray:
    """
    Visualise absolute disparity error as a colour image.

    Low error → blue/black, High error → red/white.

    Args:
        pred      : Predicted disparity (1, H, W) or (H, W).
        gt        : Ground truth disparity (1, H, W) or (H, W).
        max_error : Clamp errors above this value (pixels).
                    Default 5.0 pixels — typical for stereo methods.

    Returns:
        rgb : (H, W, 3) uint8 numpy array.
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if gt.dim() == 3:
        gt = gt.squeeze(0)

    error = (pred - gt).abs().detach().cpu().float().numpy()
    return disp_to_color(
        torch.from_numpy(error),
        vmin=0.0, vmax=max_error,
        colormap='hot',
    )


def tensor_to_rgb(img: torch.Tensor) -> np.ndarray:
    """
    Convert an image tensor to a uint8 RGB numpy array.

    Args:
        img : (3, H, W) float tensor. Values in [-1, 1] or [0, 1].

    Returns:
        rgb : (H, W, 3) uint8 numpy array.
    """
    if img.dim() == 3:
        img = img.permute(1, 2, 0)   # (H, W, 3)
    img_np = img.detach().cpu().float().numpy()

    # Normalise from [-1,1] or [0,1] to [0,255]
    if img_np.min() < -0.01:
        img_np = (img_np + 1.0) / 2.0   # [-1,1] → [0,1]
    img_np = np.clip(img_np, 0.0, 1.0)
    return (img_np * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def make_comparison_figure(
    img_left: torch.Tensor,
    disp_pred: torch.Tensor,
    disp_gt: torch.Tensor,
    max_disp: float = None,
) -> np.ndarray:
    """
    Create a side-by-side comparison figure:
      [Left Image | Predicted Disparity | Ground Truth | Error Map]

    Args:
        img_left  : Left RGB image (3, H, W).
        disp_pred : Predicted disparity (1, H, W).
        disp_gt   : Ground truth disparity (1, H, W).
        max_disp  : Max disparity for consistent colour scale.
                    If None, uses gt max.

    Returns:
        figure : (H, 4*W, 3) uint8 numpy array. Ready to save with PIL.
    """
    # Determine shared colour scale from ground truth
    valid_mask = disp_gt > 0
    if valid_mask.any():
        v_max = disp_gt[valid_mask].max().item()
    else:
        v_max = disp_pred.max().item()

    if max_disp is not None:
        v_max = min(v_max, max_disp)
    v_min = 0.0

    # Generate each panel
    panel_img  = tensor_to_rgb(img_left)
    panel_pred = disp_to_color(disp_pred, vmin=v_min, vmax=v_max)
    panel_gt   = disp_to_color(disp_gt,   vmin=v_min, vmax=v_max)
    panel_err  = error_to_color(disp_pred, disp_gt, max_error=max(3.0, v_max * 0.05))

    # Concatenate horizontally
    figure = np.concatenate(
        [panel_img, panel_pred, panel_gt, panel_err], axis=1
    )
    return figure


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_disp_image(
    disp: torch.Tensor,
    path: str,
    vmin: float = None,
    vmax: float = None,
    colormap: str = 'plasma',
):
    """
    Save a disparity tensor directly to a PNG file.

    Args:
        disp     : Disparity tensor (1, H, W) or (H, W).
        path     : Output file path (should end in .png).
        vmin/vmax: Colour scale range.
        colormap : Matplotlib colormap name.
    """
    from PIL import Image
    rgb = disp_to_color(disp, vmin=vmin, vmax=vmax, colormap=colormap)
    Image.fromarray(rgb).save(path)


def save_comparison_figure(
    img_left: torch.Tensor,
    disp_pred: torch.Tensor,
    disp_gt: torch.Tensor,
    path: str,
    max_disp: float = None,
):
    """
    Save the 4-panel comparison figure to a PNG file.

    Args:
        img_left, disp_pred, disp_gt : As in make_comparison_figure.
        path     : Output file path.
        max_disp : Max disparity for colour scale.
    """
    from PIL import Image
    figure = make_comparison_figure(img_left, disp_pred, disp_gt, max_disp)
    Image.fromarray(figure).save(path)
    return path


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H, W = 64, 128

    img   = torch.rand(3, H, W)
    pred  = torch.rand(1, H, W) * 80 + 1
    gt    = torch.rand(1, H, W) * 80 + 1

    rgb   = disp_to_color(pred)
    err   = error_to_color(pred, gt)
    fig   = make_comparison_figure(img, pred, gt)

    assert rgb.shape == (H, W, 3)
    assert err.shape == (H, W, 3)
    assert fig.shape == (H, W * 4, 3)
    assert rgb.dtype == np.uint8

    print(f"disp_to_color:       {rgb.shape}")
    print(f"error_to_color:      {err.shape}")
    print(f"comparison figure:   {fig.shape}")
    print("✅ visualize smoke test passed.")