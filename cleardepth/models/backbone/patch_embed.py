"""
Overlap Patch Embedding
=======================
Converts a raw RGB image (B, 3, H, W) into a sequence of patch tokens
(B, N, C) where N = (H/stride) * (W/stride).

Key design choice: overlapping patches (kernel > stride) preserve local
boundary information at patch borders. This is crucial for transparent
objects where structural edges are the primary signal.

Paper reference: Section III-B
  "Our backbone begins with overlap patch embedding for initial
   tokenization, preserving local features."
"""

import torch
import torch.nn as nn
from einops import rearrange


class OverlapPatchEmbed(nn.Module):
    """
    Overlap patch embedding via a strided convolution.

    Args:
        in_channels  : Number of input image channels (3 for RGB).
        embed_dim    : Output channel dimension C.
        patch_size   : Convolution kernel size (default 7 for Stage 1,
                       3 for subsequent stages).
        stride       : Convolution stride that controls spatial downsampling.
                       Stage 1 uses stride=4 to produce 1/4-scale features.
        padding      : Padding to keep output size = ceil(H/stride).
                       For kernel=7, stride=4: padding=3.
                       For kernel=3, stride=2: padding=1.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ):
        super().__init__()

        # A single conv layer does the patch extraction + channel projection
        # in one shot. The overlap comes from patch_size > stride.
        #
        # Shape trace:
        #   Input:  (B, in_channels, H, W)
        #   Output: (B, embed_dim, H/stride, W/stride)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )

        # LayerNorm applied channel-wise after flattening to (B, N, C).
        # This stabilises training by normalising each token's features.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Image tensor of shape (B, C_in, H, W).

        Returns:
            tokens : Patch tokens of shape (B, N, embed_dim)
                     where N = (H/stride) * (W/stride).
            H_out  : Spatial height of the token grid (= H/stride).
            W_out  : Spatial width  of the token grid (= W/stride).

        We return H_out and W_out so the next stage knows how to reshape
        tokens back into a 2-D grid (needed for the Conv2d inside
        EfficientAttention and MixFFN).
        """
        x = self.proj(x)                          # (B, embed_dim, H', W')
        B, C, H_out, W_out = x.shape

        # Flatten spatial dims into a sequence of tokens:
        #   (B, C, H', W') → (B, H'*W', C)
        # einops makes the intention explicit and avoids shape bugs.
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.norm(x)                          # (B, N, C) — normalised
        return x, H_out, W_out


# ---------------------------------------------------------------------------
# Quick self-test (only runs if you execute this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W = 1, 64, 128   # Smoke-test resolution

    # Stage 1: 7×7 kernel, stride 4 → 1/4 scale
    embed = OverlapPatchEmbed(in_channels=3, embed_dim=64,
                              patch_size=7, stride=4, padding=3)
    x = torch.randn(B, 3, H, W)
    tokens, h_out, w_out = embed(x)

    print(f"Input shape:   {list(x.shape)}")
    print(f"Output tokens: {list(tokens.shape)}")   # expect (1, 16*32, 64) = (1, 512, 64)
    print(f"Token grid:    {h_out} x {w_out}")      # expect 16 x 32
    assert tokens.shape == (B, h_out * w_out, 64)
    print("✅ OverlapPatchEmbed smoke test passed.")