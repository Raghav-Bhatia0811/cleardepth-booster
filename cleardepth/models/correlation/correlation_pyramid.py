"""
Correlation Pyramid
===================
Computes multi-scale dot-product similarity between left and right
feature maps, producing a lookup volume used by the GRU at each iteration.

How it works:
  1. Build a 4-level pyramid of right features by average pooling.
  2. At each level, for every left pixel position (x, y), compute dot
     products against right pixels in a horizontal window of radius r
     centered at (x - current_disparity, y).
  3. Concatenate lookups from all levels → correlation feature vector.

Why a pyramid?
  Coarse levels (heavily pooled) capture large displacement matches —
  useful when the initial disparity estimate is far off.
  Fine levels capture precise sub-pixel corrections.
  Together they give the GRU both a wide search range and fine precision.

Why only horizontal search?
  Epipolar geometry: for a rectified stereo pair, correspondences always
  lie on the same horizontal scanline. This reduces the 2D search space
  to 1D, cutting computation by a factor of image width.

Output per pixel: (2*radius + 1) values per level × num_levels values
  Default: (2*4 + 1) * 4 = 36 correlation features per pixel.

Paper reference: Fig. 2 (Correlation Pyramid C1...C4), Section III-C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelationPyramid(nn.Module):
    """
    Multi-scale correlation pyramid for stereo matching.

    Args:
        num_levels : Number of pyramid levels (default 4).
        radius     : Search radius at each level (default 4).
                     Each level contributes 2*radius+1 = 9 values.
    """

    def __init__(self, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

    @property
    def out_channels(self) -> int:
        """Total correlation channels per pixel fed into the GRU."""
        return self.num_levels * (2 * self.radius + 1)

    def _build_pyramid(self, feat_right: torch.Tensor) -> list:
        """
        Build a multi-scale pyramid of right features.

        Args:
            feat_right : Right feature map (B, C, H, W).

        Returns:
            List of tensors, level l has shape (B, C, H, W / 2^l).
            Level 0 = original, level 1 = 2× pooled, etc.
        """
        pyramid = [feat_right]
        for _ in range(self.num_levels - 1):
            # Pool only along width (horizontal / disparity direction)
            # keepdim-style: kernel=(1,2), stride=(1,2) halves W but keeps H
            feat_right = F.avg_pool2d(feat_right, kernel_size=(1, 2),
                                      stride=(1, 2))
            pyramid.append(feat_right)
        return pyramid

    def _lookup_level(
        self,
        feat_left: torch.Tensor,
        feat_right_level: torch.Tensor,
        disparity: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """
        Correlation lookup at one pyramid level.

        For each left pixel at position (x, y) with current disparity d:
          - The estimated right-image x position is x - d (disparity shifts left)
          - We sample right features at positions (x - d + offset) for
            offset in [-radius, ..., +radius]
          - Dot product with the left feature gives the similarity score

        Args:
            feat_left        : (B, C, H, W)
            feat_right_level : (B, C, H, W_l) — right features at this level
            disparity        : (B, 1, H, W) — current disparity estimate
            level            : Pyramid level index (used to scale disparity)

        Returns:
            corr : (B, 2*radius+1, H, W) — correlation values at this level
        """
        B, C, H, W = feat_left.shape
        _, _, H_r, W_r = feat_right_level.shape

        # Scale disparity to this level's resolution
        # Level l has W_r = W / 2^l, so disparity in level-l coords = d / 2^l
        scale = 2 ** level
        disp_scaled = disparity / scale   # (B, 1, H, W)

        # Build sampling grid for the right feature map.
        # grid_sample expects coordinates in [-1, 1] normalised to feat size.
        #
        # Base x coordinate in right image (in pixels, level-l coords):
        #   x_right = x_left - disp_scaled
        # Then we add offset ∈ [-radius, ..., +radius]

        # Pixel coordinates of left image positions
        # xs: (W,), ys: (H,) — regular pixel grid
        xs = torch.arange(W, device=feat_left.device, dtype=torch.float32)
        ys = torch.arange(H, device=feat_left.device, dtype=torch.float32)

        # Meshgrid: (H, W) grids
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        # Expand to batch: (1, H, W)
        grid_x = grid_x.unsqueeze(0)   # pixel x coords in left image
        grid_y = grid_y.unsqueeze(0)   # pixel y coords (same in both images)

        # Collect correlation for each offset
        corr_list = []
        for offset in range(-self.radius, self.radius + 1):
            # x position in right image at this level (pixel coords)
            x_right = grid_x - disp_scaled[:, 0] + offset   # (B, H, W)

            # Normalise to [-1, 1] for grid_sample
            # grid_sample convention: -1 = left edge, +1 = right edge
            x_norm = 2.0 * x_right / max(W_r - 1, 1) - 1.0
            y_norm = 2.0 * grid_y.expand(B, -1, -1) / max(H - 1, 1) - 1.0

            # Stack into (B, H, W, 2) grid — grid_sample expects (x, y) order
            grid = torch.stack([x_norm, y_norm], dim=-1)   # (B, H, W, 2)

            # Sample right features at these positions (bilinear interpolation)
            # feat_right_level: (B, C, H, W_r)
            # sampled: (B, C, H, W)
            sampled = F.grid_sample(
                feat_right_level, grid,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )

            # Dot product: sum over channel dim C
            # feat_left: (B, C, H, W), sampled: (B, C, H, W)
            # result: (B, 1, H, W)
            dot = (feat_left * sampled).sum(dim=1, keepdim=True)
            corr_list.append(dot)

        # Stack offsets: (B, 2*radius+1, H, W)
        corr = torch.cat(corr_list, dim=1)
        return corr

    def forward(
        self,
        feat_left: torch.Tensor,
        feat_right: torch.Tensor,
        disparity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-scale correlation features.

        Args:
            feat_left  : Left  features (B, C, H, W).
            feat_right : Right features (B, C, H, W).
            disparity  : Current disparity estimate (B, 1, H, W).
                         Initialised to zeros at the start of GRU iterations.

        Returns:
            corr_features : (B, num_levels * (2*radius+1), H, W)
                            = (B, 36, H, W) with default settings.
        """
        # Build right-image pyramid once per forward pass
        pyramid = self._build_pyramid(feat_right)

        # Collect correlations from all levels
        corr_all = []
        for level, feat_right_l in enumerate(pyramid):
            corr_l = self._lookup_level(feat_left, feat_right_l,
                                        disparity, level)
            corr_all.append(corr_l)

        # Concatenate all levels: (B, num_levels*(2r+1), H, W)
        return torch.cat(corr_all, dim=1)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 1, 64, 16, 32   # Feature map dims at 1/4 scale

    corr_pyramid = CorrelationPyramid(num_levels=4, radius=4)
    feat_l = torch.randn(B, C, H, W)
    feat_r = torch.randn(B, C, H, W)
    disp   = torch.zeros(B, 1, H, W)   # Initial disparity = 0

    out = corr_pyramid(feat_l, feat_r, disp)
    expected_channels = 4 * (2 * 4 + 1)   # 36

    assert out.shape == (B, expected_channels, H, W), \
        f"Expected ({B}, {expected_channels}, {H}, {W}), got {tuple(out.shape)}"

    print(f"Correlation output: {list(out.shape)}")
    print(f"Channels per pixel: {expected_channels} "
          f"(4 levels × 9 offsets)")
    print("✅ CorrelationPyramid smoke test passed.")