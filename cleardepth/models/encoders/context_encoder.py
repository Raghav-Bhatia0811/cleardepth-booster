"""
Context Encoder
===============
Wraps the CascadedViT backbone and runs it on the left image only.
Produces three structural feature tensors (c_k, c_r, c_h) that are
injected as additive biases into the Post-Fusion GRU gates.

Design: Strategy B — three separate 1×1 conv projection heads.
Each head independently projects the backbone output to the GRU's
hidden_dim, giving each gate its own learned structural representation.

Paper reference: Section III-C, Equations (8), (9), (10)
  c_k biases the update gate z
  c_r biases the reset gate r
  c_h biases the candidate hidden state h_tilde

The structural features are computed ONCE before the GRU iteration loop
and reused at every iteration step. They are constants w.r.t. the loop.
"""

import torch
import torch.nn as nn
from ..backbone.cascaded_vit import CascadedViT


class ContextEncoder(nn.Module):
    """
    Context encoder: extracts structural priors from the left image.

    Args:
        in_channels    : Input image channels (3 for RGB).
        embed_dim      : Backbone output channel dimension.
        hidden_dim     : GRU hidden state dimension.
                         The three output tensors will have this many channels.
        depths         : ViT blocks per stage.
        num_heads      : Attention heads per stage.
        reduction_ratios : Sequence reduction ratios per stage.
        mlp_ratio      : MixFFN hidden dim multiplier.
        drop_rate      : Dropout rate.
        drop_path_rate : Max stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        depths: list = [2, 2, 2, 2],
        num_heads: list = [1, 2, 4, 8],
        reduction_ratios: list = [8, 4, 2, 1],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        fuse_out_channels: int = None,
    ):
        super().__init__()

        self.backbone = CascadedViT(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            reduction_ratios=reduction_ratios,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            fuse_out_channels=fuse_out_channels,
        )

        # The backbone outputs fuse_out_channels (or embed_dim if not set).
        # Projection heads must match that output dimension.
        feat_channels = self.backbone.out_channels

        # Three independent 1×1 conv projection heads.
        # Each projects backbone features (feat_channels) →
        # hidden_dim channels to match the GRU's gate dimensions.
        # Tanh bounds the bias values — prevents structural features from
        # dominating gate activations across many GRU iterations.
        def make_proj_head(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True),
                nn.Tanh(),
            )

        self.proj_k = make_proj_head(feat_channels, hidden_dim)  # → c_k
        self.proj_r = make_proj_head(feat_channels, hidden_dim)  # → c_r
        self.proj_h = make_proj_head(feat_channels, hidden_dim)  # → c_h

    def forward(self, img_left: torch.Tensor):
        """
        Args:
            img_left : Left image tensor (B, 3, H, W).

        Returns:
            c_k : Structural bias for update gate   (B, hidden_dim, H/4, W/4).
            c_r : Structural bias for reset gate    (B, hidden_dim, H/4, W/4).
            c_h : Structural bias for candidate     (B, hidden_dim, H/4, W/4).

        These are injected into the GRU as per Equations (8)-(10):
            z = σ(Conv([h_prev, x], W_z) + c_k)
            r = σ(Conv([h_prev, x], W_r) + c_r)
            h̃ = tanh(Conv([r⊙h_prev, x], W_h) + c_h)
        """
        # Shared structural feature extraction
        feat = self.backbone(img_left)   # (B, embed_dim, H/4, W/4)

        # Three independent projections — each gate gets its own view
        c_k = self.proj_k(feat)          # (B, hidden_dim, H/4, W/4)
        c_r = self.proj_r(feat)          # (B, hidden_dim, H/4, W/4)
        c_h = self.proj_h(feat)          # (B, hidden_dim, H/4, W/4)

        return c_k, c_r, c_h


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W = 1, 64, 128
    EMBED_C, HIDDEN_DIM = 64, 128

    encoder = ContextEncoder(embed_dim=EMBED_C, hidden_dim=HIDDEN_DIM)
    left = torch.randn(B, 3, H, W)

    c_k, c_r, c_h = encoder(left)
    expected = (B, HIDDEN_DIM, H // 4, W // 4)

    assert c_k.shape == torch.Size(expected)
    assert c_r.shape == torch.Size(expected)
    assert c_h.shape == torch.Size(expected)

    print(f"c_k: {list(c_k.shape)}")
    print(f"c_r: {list(c_r.shape)}")
    print(f"c_h: {list(c_h.shape)}")
    print("✅ ContextEncoder smoke test passed.")