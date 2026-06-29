"""
Cascaded ViT Backbone
=====================
Implements the 4-stage hierarchical Vision Transformer backbone described
in Section III-B of the ClearDepth paper.

Each stage:
  1. OverlapPatchEmbed  — tokenise at progressively coarser spatial scales
  2. Stack of ViTBlocks — transformer processing with efficient attention
  3. Reshape            — convert token sequence back to 2D feature map

Feature fusion (paper: "concatenate multi-scale feature maps ... upsampling
them to a unified scale of 1/4 ... 1×1 convolution"):
  - feat2, feat3, feat4 are bilinearly upsampled to match feat1's 1/4 scale
  - All four maps are concatenated along the channel dimension
  - A 1×1 conv projects to the final embed_dim output channels

Output: single feature map at 1/4 input resolution, shape (B, embed_dim, H/4, W/4).
This is consumed by both the Feature Encoder and the Context Encoder.

Paper reference: Section III-B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .patch_embed import OverlapPatchEmbed
from .vit_block import ViTBlock


class CascadedViT(nn.Module):
    """
    4-stage cascaded Vision Transformer backbone.

    Args:
        in_channels      : Input image channels (3 for RGB).
        embed_dim        : Base channel dimension C for Stage 1.
                           Stages 2/3/4 use 2C/4C/8C automatically.
        depths           : Number of ViTBlocks per stage, e.g. [2, 2, 2, 2].
        num_heads        : Attention heads per stage, e.g. [1, 2, 4, 8].
        reduction_ratios : Sequence reduction R per stage, e.g. [8, 4, 2, 1].
        mlp_ratio        : Hidden dim multiplier in MixFFN (default 4.0).
        drop_rate        : Dropout rate in FFN and attention projections.
        drop_path_rate   : Maximum stochastic depth rate (linearly scheduled).
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

        # Channel dims for each stage: C, 2C, 4C, 8C
        dims = [embed_dim * (2 ** i) for i in range(4)]  # [64, 128, 256, 512]

        # ── Stage patch embeddings ─────────────────────────────────────────
        # Stage 1: large 7×7 kernel, stride 4 → 1/4 scale
        # Stages 2-4: smaller 3×3 kernel, stride 2 → halve each time
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(in_channels, dims[0],
                              patch_size=7, stride=4, padding=3),   # 1/4
            OverlapPatchEmbed(dims[0], dims[1],
                              patch_size=3, stride=2, padding=1),   # 1/8
            OverlapPatchEmbed(dims[1], dims[2],
                              patch_size=3, stride=2, padding=1),   # 1/16
            OverlapPatchEmbed(dims[2], dims[3],
                              patch_size=3, stride=2, padding=1),   # 1/32
        ])

        # ── DropPath schedule ──────────────────────────────────────────────
        # Linearly space drop rates across ALL blocks in ALL stages combined.
        # e.g. with depths=[2,2,2,2] → 8 blocks total, rates 0..drop_path_rate
        total_blocks = sum(depths)
        dp_rates = [
            r.item()
            for r in torch.linspace(0, drop_path_rate, total_blocks)
        ]

        # ── ViT block stacks per stage ─────────────────────────────────────
        self.stages = nn.ModuleList()
        block_idx = 0
        for stage_i in range(4):
            blocks = nn.ModuleList([
                ViTBlock(
                    dim=dims[stage_i],
                    num_heads=num_heads[stage_i],
                    reduction_ratio=reduction_ratios[stage_i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path_rate=dp_rates[block_idx + j],
                )
                for j in range(depths[stage_i])
            ])
            self.stages.append(blocks)
            block_idx += depths[stage_i]

        # LayerNorms applied after each stage's block stack
        # (one norm per stage, normalising the final token sequence)
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i]) for i in range(4)
        ])

        # ── Feature Fusion ─────────────────────────────────────────────────
        # After upsampling stages 2-4 to 1/4 scale, concatenate all 4 maps.
        # Total channels = C + 2C + 4C + 8C = 15C
        total_fused_channels = sum(dims)   # 64+128+256+512 = 960

        # fuse_out_channels controls the fusion output independently of embed_dim.
        # Paper config specifies 256; defaults to embed_dim if not set.
        _fuse_out = fuse_out_channels if fuse_out_channels is not None else embed_dim
        self.out_channels = _fuse_out

        # 1×1 conv projects 15C → fuse_out_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_fused_channels, _fuse_out,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(_fuse_out),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Standard ViT weight initialisation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Truncated normal init — standard for ViT weights
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _run_stage(
        self,
        x: torch.Tensor,
        patch_embed: OverlapPatchEmbed,
        blocks: nn.ModuleList,
        norm: nn.LayerNorm,
    ):
        """
        Run one complete stage: patch embed → transformer blocks → norm → reshape.

        Args:
            x           : Input feature map (B, C_in, H_in, W_in).
            patch_embed : This stage's OverlapPatchEmbed.
            blocks      : This stage's list of ViTBlocks.
            norm        : This stage's LayerNorm.

        Returns:
            feat : 2D feature map (B, C_out, H_out, W_out).
            H_out, W_out : Spatial dimensions of the output.
        """
        # Tokenise: (B, C_in, H_in, W_in) → (B, N, C_out), plus grid dims
        tokens, H_out, W_out = patch_embed(x)

        # Pass through each ViTBlock in sequence
        for block in blocks:
            tokens = block(tokens, H_out, W_out)

        # Final normalisation
        tokens = norm(tokens)

        # Reshape token sequence back to 2D spatial feature map
        # (B, N, C_out) → (B, C_out, H_out, W_out)
        feat = rearrange(tokens, 'b (h w) c -> b c h w', h=H_out, w=W_out)
        return feat, H_out, W_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input image tensor (B, 3, H, W).

        Returns:
            Fused feature map (B, embed_dim, H/4, W/4).
        """
        # ── Run all 4 stages ───────────────────────────────────────────────
        features = []
        current = x
        for i in range(4):
            feat, H_out, W_out = self._run_stage(
                current,
                self.patch_embeds[i],
                self.stages[i],
                self.norms[i],
            )
            features.append(feat)
            # Feed this stage's output as input to the next stage
            current = feat

        feat1, feat2, feat3, feat4 = features
        # Shapes at smoke-test resolution (64×128):
        #   feat1: (B, C,   16, 32)   ← 1/4  scale
        #   feat2: (B, 2C,   8, 16)   ← 1/8  scale
        #   feat3: (B, 4C,   4,  8)   ← 1/16 scale
        #   feat4: (B, 8C,   2,  4)   ← 1/32 scale

        # ── Upsample coarser features to 1/4 scale ────────────────────────
        # Target size = feat1's spatial dimensions
        target_size = (feat1.shape[2], feat1.shape[3])

        feat2_up = F.interpolate(feat2, size=target_size,
                                 mode='bilinear', align_corners=False)
        feat3_up = F.interpolate(feat3, size=target_size,
                                 mode='bilinear', align_corners=False)
        feat4_up = F.interpolate(feat4, size=target_size,
                                 mode='bilinear', align_corners=False)

        # ── Concatenate and fuse ───────────────────────────────────────────
        # All four maps now have shape (B, *, H/4, W/4)
        # Concatenated: (B, C+2C+4C+8C, H/4, W/4) = (B, 15C, H/4, W/4)
        fused = torch.cat([feat1, feat2_up, feat3_up, feat4_up], dim=1)

        # 1×1 conv: (B, 15C, H/4, W/4) → (B, embed_dim, H/4, W/4)
        out = self.fusion_conv(fused)
        return out


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, H, W = 1, 64, 128   # Smoke-test resolution

    backbone = CascadedViT(
        in_channels=3,
        embed_dim=64,
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        reduction_ratios=[8, 4, 2, 1],
    )

    x = torch.randn(B, 3, H, W)
    out = backbone(x)

    expected = (B, 64, H // 4, W // 4)   # (1, 64, 16, 32)
    assert out.shape == torch.Size(expected), \
        f"Expected {expected}, got {tuple(out.shape)}"

    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"Input:       {list(x.shape)}")
    print(f"Output:      {list(out.shape)}")
    print(f"Parameters:  {total_params:,}")
    print("✅ CascadedViT smoke test passed.")