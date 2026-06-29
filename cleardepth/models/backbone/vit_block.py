"""
ViT Block
=========
One complete transformer block assembling EfficientSelfAttention + MixFFN.

Structure (pre-norm style):
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

"Pre-norm" means LayerNorm is applied BEFORE the sublayer, not after.
This is the standard in modern ViTs (SegFormer, DINOv2) and trains more
stably than post-norm at large scale.

Why stochastic depth (DropPath)?
  During training, each block is randomly "skipped" (its output replaced
  with zero) with probability drop_path_rate. This regularises deep
  networks the same way Dropout regularises wide networks — it forces the
  model not to over-rely on any single block.
  At inference time, DropPath is disabled (standard nn.Module eval mode).

Paper reference: Section III-B (implied by "four transformer blocks").
"""

import torch
import torch.nn as nn
from timm.layers import DropPath   # standard ViT utility from timm

from .efficient_attention import EfficientSelfAttention
from .mix_ffn import MixFFN


class ViTBlock(nn.Module):
    """
    A single transformer block with efficient attention and Mix-FFN.

    Args:
        dim              : Token channel dimension C.
        num_heads        : Number of attention heads.
        reduction_ratio  : Sequence reduction factor R for this stage.
        mlp_ratio        : Hidden dim multiplier for MixFFN (default 4).
        attn_drop        : Dropout on attention weights.
        drop             : Dropout on FFN output.
        drop_path_rate   : Stochastic depth rate (0 = disabled).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        reduction_ratio: int = 1,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Pre-norm before attention
        self.norm1 = nn.LayerNorm(dim)

        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            reduction_ratio=reduction_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Pre-norm before FFN
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = MixFFN(
            dim=dim,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

        # DropPath: stochastic depth regularisation.
        # nn.Identity() when drop_path_rate=0 (no overhead at inference).
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x : Token sequence (B, N, C) where N = H * W.
            H : Spatial height of token grid (needed for Conv2d inside
                EfficientAttention and MixFFN).
            W : Spatial width of token grid.

        Returns:
            Tensor of shape (B, N, C).
        """
        # ── Attention sublayer (pre-norm + residual + stochastic depth) ────
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))

        # ── FFN sublayer (pre-norm + residual + stochastic depth) ──────────
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))

        return x


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W, C = 1, 16, 32, 64   # Smoke-test resolution

    # Test all 4 stage configurations
    configs = [
        dict(num_heads=1, reduction_ratio=8),   # Stage 1
        dict(num_heads=2, reduction_ratio=4),   # Stage 2
        dict(num_heads=4, reduction_ratio=2),   # Stage 3
        dict(num_heads=8, reduction_ratio=1),   # Stage 4
    ]

    for i, cfg in enumerate(configs):
        # Each stage uses half the H and W of the previous
        h = H // (2 ** i)
        w = W // (2 ** i)
        c = C * (2 ** i)   # dim doubles each stage (typical ViT scaling)

        block = ViTBlock(dim=c, drop_path_rate=0.1, **cfg)
        x = torch.randn(B, h * w, c)
        out = block(x, h, w)
        assert out.shape == x.shape, f"Stage {i+1} shape mismatch"
        print(f"✅ Stage {i+1}: (R={cfg['reduction_ratio']}, "
              f"heads={cfg['num_heads']}) "
              f"→ {list(x.shape)} → {list(out.shape)}")

    print("✅ ViTBlock smoke test passed.")