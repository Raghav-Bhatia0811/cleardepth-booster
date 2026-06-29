"""
Milestone 3 Smoke Test — Cascaded ViT Backbone
===============================================
Tests the full 4-stage backbone at smoke-test resolution (64×128, batch=1).

Run with: pytest tests/test_m3_backbone.py -v
"""

import torch
import pytest
from cleardepth.models.backbone.cascaded_vit import CascadedViT


BATCH   = 1
H_IMG   = 64
W_IMG   = 128
EMBED_C = 64


def make_backbone(**kwargs):
    """Helper: build backbone with paper defaults, allow overrides."""
    defaults = dict(
        in_channels=3,
        embed_dim=EMBED_C,
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        reduction_ratios=[8, 4, 2, 1],
        mlp_ratio=4.0,
        drop_path_rate=0.1,
    )
    defaults.update(kwargs)
    return CascadedViT(**defaults)


class TestCascadedViT:

    def test_output_shape(self):
        """Output must be (B, embed_dim, H/4, W/4)."""
        backbone = make_backbone()
        x = torch.randn(BATCH, 3, H_IMG, W_IMG)
        out = backbone(x)
        expected = (BATCH, EMBED_C, H_IMG // 4, W_IMG // 4)
        assert out.shape == torch.Size(expected), (
            f"Expected {expected}, got {tuple(out.shape)}"
        )

    def test_output_shape_odd_divisible(self):
        """Should work on any resolution divisible by 32."""
        backbone = make_backbone()
        # 96×192 is divisible by 32 — a common training resolution
        x = torch.randn(BATCH, 3, 96, 192)
        out = backbone(x)
        assert out.shape == (BATCH, EMBED_C, 96 // 4, 192 // 4)

    def test_gradient_flow(self):
        """Gradients must flow from output all the way back to the input."""
        backbone = make_backbone()
        x = torch.randn(BATCH, 3, H_IMG, W_IMG, requires_grad=True)
        out = backbone(x)
        out.mean().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any(), "NaN gradients detected"
        assert x.grad.abs().sum() > 0, "Zero gradients — something is detached"

    def test_eval_mode_deterministic(self):
        """In eval mode, identical inputs must give identical outputs."""
        backbone = make_backbone(drop_path_rate=0.5, drop_rate=0.1)
        backbone.eval()
        x = torch.randn(BATCH, 3, H_IMG, W_IMG)
        with torch.no_grad():
            out1 = backbone(x)
            out2 = backbone(x)
        torch.testing.assert_close(out1, out2)

    def test_different_embed_dims(self):
        """Backbone should work with any even embed_dim."""
        for dim in [32, 64]:
            backbone = CascadedViT(
                embed_dim=dim,
                depths=[1, 1, 1, 1],      # 1 block per stage for speed
                num_heads=[1, 2, 4, 8],
                reduction_ratios=[8, 4, 2, 1],
            )
            x = torch.randn(BATCH, 3, H_IMG, W_IMG)
            out = backbone(x)
            assert out.shape == (BATCH, dim, H_IMG // 4, W_IMG // 4), \
                f"Failed for embed_dim={dim}"

    def test_no_nan_in_output(self):
        """Output must not contain NaN or Inf values."""
        backbone = make_backbone()
        x = torch.randn(BATCH, 3, H_IMG, W_IMG)
        out = backbone(x)
        assert not torch.isnan(out).any(), "NaN in backbone output"
        assert not torch.isinf(out).any(), "Inf in backbone output"

    def test_parameter_count_reasonable(self):
        """
        ClearDepth reports 99.45M total params (Table VI).
        The backbone alone should be well under that — a sanity bound check.
        With embed_dim=64 and depths=[2,2,2,2], expect roughly 3-8M params.
        """
        backbone = make_backbone()
        total = sum(p.numel() for p in backbone.parameters())
        print(f"\n  Backbone parameters: {total:,}")
        # Sanity bounds — not too tiny, not larger than full model
        assert total > 100_000,    f"Suspiciously few params: {total:,}"
        assert total < 99_000_000, f"Backbone alone exceeds full model: {total:,}"

    def test_feature_fusion_output_channels(self):
        """
        The fusion conv must project 15*embed_dim → embed_dim.
        Verify by checking the fusion_conv weight shapes.
        """
        backbone = make_backbone()
        fusion_weight = backbone.fusion_conv[0].weight  # Conv2d weight
        in_ch  = fusion_weight.shape[1]
        out_ch = fusion_weight.shape[0]
        assert out_ch == EMBED_C, \
            f"Fusion output channels: expected {EMBED_C}, got {out_ch}"
        assert in_ch == EMBED_C * 15, \
            f"Fusion input channels: expected {EMBED_C * 15}, got {in_ch}"