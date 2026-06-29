"""
ClearDepth — Full Model Assembly
=================================
Wires together all components into one end-to-end stereo depth network.

Forward pass:
  1. FeatureEncoder  → feat_left, feat_right  (appearance features, shared weights)
  2. ContextEncoder  → c_k, c_r, c_h          (structural priors from left only)
  3. PostFusionGRU   → [d_1, ..., d_N], h_0   (iterative refinement + finest hidden)
  4. ConvexUpsample  → d_full_res             (inference only: 1/4 → full resolution)

Training (test_mode=False):
  Returns all N disparity predictions at 1/4-scale for sequence loss.

Inference (test_mode=True):
  Returns only the final full-resolution prediction via convex upsampling.

Paper reference: Fig. 2, Section III overall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

from .encoders.feature_encoder import FeatureEncoder
from .encoders.context_encoder import ContextEncoder
from .correlation.correlation_pyramid import CorrelationPyramid
from .gru.post_fusion_gru import PostFusionGRU
from .upsample.convex_upsample import ConvexUpsample


class ClearDepthNet(nn.Module):
    """
    Full ClearDepth stereo depth estimation network.

    Args:
        # Backbone (shared by feature and context encoders)
        in_channels      : Input image channels (3 for RGB).
        embed_dim        : Backbone base channel dim (C) — stages use C,2C,4C,8C.
        fuse_out_channels: Output channels after multi-scale fusion. Paper config: 256.
                           If None, defaults to embed_dim (backward-compatible).
        depths           : ViT blocks per stage.
        num_heads        : Attention heads per stage.
        reduction_ratios : Sequence reduction ratios per stage.
        mlp_ratio        : MixFFN expansion ratio.
        drop_rate        : Dropout rate.
        drop_path_rate   : Max stochastic depth rate.

        # GRU
        hidden_dim       : GRU hidden state dimension.
        n_gru_layers     : Number of GRU scales (default 3).
        n_gru_iters      : Refinement iterations during training.

        # Correlation
        corr_levels      : Correlation pyramid levels (default 4).
        corr_radius      : Search radius per level (default 4).

        # Upsampling
        upsample_scale   : Factor to upsample from 1/4 to full res (default 4).
    """

    def __init__(
        self,
        # Backbone
        in_channels: int = 3,
        embed_dim: int = 64,
        fuse_out_channels: int = 256,
        depths: list = [2, 2, 2, 2],
        num_heads: list = [1, 2, 4, 8],
        reduction_ratios: list = [8, 4, 2, 1],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        # GRU
        hidden_dim: int = 128,
        n_gru_layers: int = 3,
        n_gru_iters: int = 22,
        # Correlation
        corr_levels: int = 4,
        corr_radius: int = 4,
        # Upsampling
        upsample_scale: int = 4,
    ):
        super().__init__()

        self.n_gru_iters = n_gru_iters

        # ── Feature Encoder ────────────────────────────────────────────────
        # Shared-weight backbone for left + right appearance features
        self.feature_encoder = FeatureEncoder(
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

        # ── Context Encoder ────────────────────────────────────────────────
        # Separate backbone for structural priors from left image only
        self.context_encoder = ContextEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            depths=depths,
            num_heads=num_heads,
            reduction_ratios=reduction_ratios,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            fuse_out_channels=fuse_out_channels,
        )

        # ── Correlation Pyramid ────────────────────────────────────────────
        # Pure computation — no learned parameters
        self.corr_pyramid = CorrelationPyramid(
            num_levels=corr_levels,
            radius=corr_radius,
        )

        # ── Post-Fusion GRU ────────────────────────────────────────────────
        corr_channels = corr_levels * (2 * corr_radius + 1)  # 36
        self.gru = PostFusionGRU(
            corr_channels=corr_channels,
            hidden_dim=hidden_dim,
            n_gru_layers=n_gru_layers,
        )

        # ── Convex Upsample ────────────────────────────────────────────────
        # Upsamples 1/4-scale disparity to full resolution at inference.
        # The mask is predicted from the GRU's finest-scale hidden state
        # (upsampled to 1/4 to match the disparity resolution).
        self.convex_upsample = ConvexUpsample(
            hidden_dim=hidden_dim,
            scale=upsample_scale,
        )

    def forward(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
        n_iters: int = None,
        test_mode: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Full forward pass.

        Args:
            img_left  : Left  image (B, 3, H, W). Values in [-1, 1].
            img_right : Right image (B, 3, H, W).
            n_iters   : Override number of GRU iterations (default: self.n_gru_iters).
            test_mode : If True, apply convex upsampling and return single full-res map.
                        If False (training), return all 1/4-scale predictions for loss.

        Returns:
            test_mode=True  : Full-resolution disparity (B, 1, H, W).
            test_mode=False : List of N disparity maps, each (B, 1, H/4, W/4).
        """
        if n_iters is None:
            n_iters = self.n_gru_iters

        H, W = img_left.shape[-2], img_left.shape[-1]

        # ── Step 1: Extract appearance features ───────────────────────────
        feat_left, feat_right = self.feature_encoder(img_left, img_right)
        # feat_left, feat_right: (B, fuse_out_channels, H/4, W/4)
        _, _, H_feat, W_feat = feat_left.shape

        # ── Step 2: Extract structural priors ─────────────────────────────
        c_k, c_r, c_h = self.context_encoder(img_left)
        # c_k, c_r, c_h: (B, hidden_dim, H/4, W/4)

        # ── Step 3: Iterative GRU refinement ──────────────────────────────
        disp_predictions, finest_hidden = self.gru(
            feat_left=feat_left,
            feat_right=feat_right,
            c_k=c_k,
            c_r=c_r,
            c_h=c_h,
            corr_fn=self.corr_pyramid,
            n_iters=n_iters,
        )
        # disp_predictions: list of (B, 1, H_feat, W_feat)
        # finest_hidden:    (B, hidden_dim, H_feat/2, W_feat/2)

        if not test_mode:
            return disp_predictions   # all predictions for sequence loss

        # ── Step 4 (inference only): Convex upsample to full resolution ───
        # The GRU's finest scale runs at (H_feat, W_feat) = (H/4, W/4), so
        # finest_hidden is already at the same resolution as final_disp.
        final_disp = disp_predictions[-1]   # (B, 1, H/4, W/4)

        # Convex upsample: H/4 → H (4× factor, mask predicted from GRU hidden state)
        disp_full = self.convex_upsample(final_disp, finest_hidden)
        return disp_full   # (B, 1, H, W)

    def param_count(self) -> dict:
        """Return parameter counts per submodule for inspection."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            'feature_encoder'  : count(self.feature_encoder),
            'context_encoder'  : count(self.context_encoder),
            'gru'              : count(self.gru),
            'convex_upsample'  : count(self.convex_upsample),
            'total'            : count(self),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W = 1, 64, 128

    model = ClearDepthNet(
        embed_dim=64,
        fuse_out_channels=256,
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        reduction_ratios=[8, 4, 2, 1],
        hidden_dim=128,
        n_gru_iters=4,
        upsample_scale=4,
    )

    left  = torch.randn(B, 3, H, W)
    right = torch.randn(B, 3, H, W)

    # Training mode: all 1/4-scale predictions
    preds = model(left, right, n_iters=4, test_mode=False)
    assert len(preds) == 4
    assert all(p.shape == (B, 1, H // 4, W // 4) for p in preds), \
        [p.shape for p in preds]

    # Inference mode: full-resolution disparity
    final = model(left, right, n_iters=4, test_mode=True)
    assert final.shape == (B, 1, H, W), final.shape

    # Parameter breakdown
    counts = model.param_count()
    print("\nParameter counts:")
    for name, n in counts.items():
        print(f"  {name:20s}: {n:>12,}")

    print(f"\nTraining output: {len(preds)} predictions, "
          f"each {list(preds[0].shape)}")
    print(f"Inference output (full-res): {list(final.shape)}")
    print("ClearDepthNet smoke test passed.")
