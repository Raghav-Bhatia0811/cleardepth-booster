"""
Mix-FFN (Feed-Forward Network with Depthwise Convolution)
=========================================================
Implements Equation (6) from the ClearDepth paper:

    x_out = MLP(GELU(Conv3×3_dw(MLP(x_in)))) + x_in

This replaces the fixed positional embedding table of standard ViTs.
The 3×3 depthwise convolution implicitly encodes position through local
neighbourhood context — each token "knows" where it is relative to its
8 neighbours without needing a global lookup table.

Why this matters for ClearDepth:
  - Training resolution: 360×720 (N ≈ 16,200 tokens at 1/4 scale)
  - A fixed position table trained at this resolution breaks if we ever
    run inference at a different resolution.
  - Mix-FFN works at ANY resolution because convolutions are translation-
    equivariant by nature.

Architecture:
  Linear(C → C*mlp_ratio)    [expand channels — "inner MLP"]
  Conv3×3_depthwise           [local position encoding]
  GELU activation
  Linear(C*mlp_ratio → C)    [project back — "outer MLP"]
  + residual (x_in)

"Depthwise" = each of the C*mlp_ratio channels gets its own 3×3 kernel.
  groups = C*mlp_ratio  means C*mlp_ratio independent convolutions.
  Much cheaper than a full conv (no cross-channel mixing).

Paper reference: Section III-B, Equation (6).
"""

import torch
import torch.nn as nn
from einops import rearrange


class MixFFN(nn.Module):
    """
    Mix-FFN: positional-embedding-free feed-forward network.

    Args:
        dim       : Token channel dimension C.
        mlp_ratio : Hidden dim multiplier. Default 4 → hidden = 4 * C.
        drop      : Dropout probability on the output.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        # First linear: C → hidden_dim  (channel expansion)
        self.fc1 = nn.Linear(dim, hidden_dim)

        # 3×3 depthwise conv: operates on hidden_dim channels independently.
        # padding=1 keeps spatial dimensions unchanged (H stays H, W stays W).
        # groups=hidden_dim makes it depthwise.
        self.dw_conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_dim,   # ← this makes it depthwise
            bias=True,
        )

        self.act = nn.GELU()

        # Second linear: hidden_dim → C  (channel compression)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x : Token sequence (B, N, C) where N = H * W.
            H : Spatial height of token grid.
            W : Spatial width  of token grid.

        Returns:
            Tensor of shape (B, N, C) — same as input.
        """
        # ── Inner MLP: expand channels ─────────────────────────────────────
        x = self.fc1(x)               # (B, N, hidden_dim)

        # ── Depthwise Conv: encode local position ──────────────────────────
        # Conv2d expects spatial format → reshape tokens to 2D grid
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.dw_conv(x)           # (B, hidden_dim, H, W) — unchanged size
        # Flatten back to token sequence
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.act(x)
        x = self.drop(x)

        # ── Outer MLP: compress channels ───────────────────────────────────
        x = self.fc2(x)               # (B, N, C)
        x = self.drop(x)

        # No residual here — the skip connection lives in ViTBlock,
        # matching the original SegFormer/MiT design: x = x + drop_path(ffn(norm(x)))
        return x


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W, C = 1, 16, 32, 64    # Smoke-test resolution

    ffn = MixFFN(dim=C, mlp_ratio=4.0)
    x = torch.randn(B, H * W, C)
    out = ffn(x, H, W)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(out.shape)}")
    print("✅ MixFFN smoke test passed.")