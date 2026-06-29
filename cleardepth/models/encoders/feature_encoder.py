"""
Feature Encoder
===============
Wraps the CascadedViT backbone and applies it to both left and right images
using shared weights. Produces appearance features for correlation matching.

Paper reference: Section III-B, Fig. 2
  "The feature encoder extracts appearance features from both left and
   right images"
"""

import torch
import torch.nn as nn
from ..backbone.cascaded_vit import CascadedViT


class FeatureEncoder(nn.Module):
    """
    Shared-weight feature encoder for stereo image pairs.

    The same CascadedViT backbone processes both images. Weight sharing
    ensures left and right features live in the same embedding space,
    which is a prerequisite for meaningful dot-product correlation.

    Args:
        in_channels  : Input image channels (3 for RGB).
        embed_dim    : Backbone output channel dimension.
        depths       : ViT blocks per stage.
        num_heads    : Attention heads per stage.
        reduction_ratios : Sequence reduction ratios per stage.
        mlp_ratio    : MixFFN hidden dim multiplier.
        drop_rate    : Dropout rate.
        drop_path_rate : Max stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        depths: list = [2, 2, 2, 2],
        num_heads: list = [1, 2, 4, 8],
        reduction_ratios: list = [8, 4, 2, 1],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        fuse_out_channels: int = None,
    ):
        super().__init__()

        # Single backbone instance — both images pass through this
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
        self.out_channels = self.backbone.out_channels

    def forward(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
    ):
        """
        Args:
            img_left  : Left image  (B, 3, H, W).
            img_right : Right image (B, 3, H, W).

        Returns:
            feat_left  : Left  features (B, embed_dim, H/4, W/4).
            feat_right : Right features (B, embed_dim, H/4, W/4).
        """
        feat_left  = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        return feat_left, feat_right


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W, C = 1, 64, 128, 64

    encoder = FeatureEncoder(embed_dim=C)
    left  = torch.randn(B, 3, H, W)
    right = torch.randn(B, 3, H, W)

    fl, fr = encoder(left, right)
    assert fl.shape == (B, C, H // 4, W // 4)
    assert fr.shape == (B, C, H // 4, W // 4)
    print(f"Left feat:  {list(fl.shape)}")
    print(f"Right feat: {list(fr.shape)}")
    print("✅ FeatureEncoder smoke test passed.")