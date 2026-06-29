"""
Efficient Self-Attention with Sequence Reduction
=================================================
Implements equations (4) and (5) from the ClearDepth paper:

    K̂ = Reshape(N/R, C·R)(K)       # compress the key/value sequence
    K  = Linear(C·R, C)(K̂)          # project back to C channels

This reduces attention cost from O(N²) to O(N × N/R) = O(N²/R).

The queries remain at full length N so the output still has N tokens
(one per spatial position). Only keys and values are compressed.

Reduction ratios per stage: R = {8, 4, 2, 1}
  Stage 1 (1/4 scale,  most tokens) → R=8, 8× cheaper
  Stage 2 (1/8 scale)               → R=4
  Stage 3 (1/16 scale)              → R=2
  Stage 4 (1/32 scale, fewest tokens)→ R=1 (standard attention)

Paper reference: Section III-B, Equations (4) and (5).
"""

import torch
import torch.nn as nn
from einops import rearrange


class EfficientSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional sequence reduction on K and V.

    Args:
        dim              : Token channel dimension C.
        num_heads        : Number of attention heads.
        reduction_ratio  : Sequence reduction factor R.
                           R=1 means standard attention (no reduction).
        attn_drop        : Dropout probability on attention weights.
        proj_drop        : Dropout probability on output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        reduction_ratio: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Scale factor for dot-product attention: 1/sqrt(head_dim)
        # Without this, large head_dims push softmax into saturation.
        self.scale = self.head_dim ** -0.5

        self.reduction_ratio = reduction_ratio

        # Q projection: full sequence length N → stays at N
        self.q = nn.Linear(dim, dim)

        # K, V projections: applied AFTER sequence reduction
        self.kv = nn.Linear(dim, dim * 2)   # combined for efficiency

        # If R > 1, we need a 2D conv to compress the spatial sequence.
        # stride=R halves (or more) the token count.
        # LayerNorm stabilises the compressed representation.
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio,
                                stride=reduction_ratio)
            self.sr_norm = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)        # output projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x : Token sequence of shape (B, N, C) where N = H * W.
            H : Spatial height of the token grid.
            W : Spatial width  of the token grid.

        Returns:
            Tensor of shape (B, N, C) — same as input.
        """
        B, N, C = x.shape

        # ── Query ──────────────────────────────────────────────────────────
        # Shape: (B, N, C) → split into heads → (B, num_heads, N, head_dim)
        q = self.q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)

        # ── Key / Value with optional sequence reduction ───────────────────
        if self.reduction_ratio > 1:
            # Step 1: reshape tokens back to 2D spatial grid for Conv2d
            #   (B, N, C) → (B, C, H, W)
            x_spatial = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

            # Step 2: Conv2d with stride=R compresses spatial dims
            #   (B, C, H, W) → (B, C, H/R, W/R)
            x_reduced = self.sr(x_spatial)

            # Step 3: flatten back to sequence of length N/R
            #   (B, C, H/R, W/R) → (B, N/R, C)
            x_reduced = rearrange(x_reduced, 'b c h w -> b (h w) c')
            x_reduced = self.sr_norm(x_reduced)
        else:
            # R=1: no compression, use original tokens
            x_reduced = x

        # Project compressed sequence to K and V
        # kv shape: (B, N_reduced, 2*C)
        kv = self.kv(x_reduced)
        # Split into K and V, each (B, num_heads, N_reduced, head_dim)
        k, v = rearrange(
            kv, 'b n (two h d) -> two b h n d',
            two=2, h=self.num_heads
        ).unbind(dim=0)

        # ── Scaled dot-product attention ───────────────────────────────────
        # attn[b, head, i, j] = dot(q_i, k_j) / sqrt(head_dim)
        # Shape: (B, num_heads, N, N_reduced)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values weighted by attention
        # (B, num_heads, N, N_reduced) × (B, num_heads, N_reduced, head_dim)
        # → (B, num_heads, N, head_dim) → (B, N, C)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Output projection + dropout
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W, C = 1, 16, 32, 64   # Smoke-test: 1/4 scale of 64×128

    for R in [8, 4, 2, 1]:
        attn = EfficientSelfAttention(dim=C, num_heads=1, reduction_ratio=R)
        x = torch.randn(B, H * W, C)
        out = attn(x, H, W)
        assert out.shape == (B, H * W, C), f"Shape mismatch for R={R}"
        print(f"✅ R={R}: input {list(x.shape)} → output {list(out.shape)}")

    print("✅ EfficientSelfAttention smoke test passed.")