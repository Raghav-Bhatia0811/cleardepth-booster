"""
Milestone 2 Smoke Test — Backbone Primitives
=============================================
Tests all four backbone primitives at smoke-test resolution (64×128, batch=1).

Each test:
  1. Instantiates the module with paper-specified hyperparameters
  2. Runs a forward pass with a random tensor
  3. Asserts the output shape is exactly correct
  4. Checks that gradients flow (backward pass doesn't crash)

Run with: pytest tests/test_m2_primitives.py -v
"""

import torch
import pytest
from cleardepth.models.backbone.patch_embed import OverlapPatchEmbed
from cleardepth.models.backbone.efficient_attention import EfficientSelfAttention
from cleardepth.models.backbone.mix_ffn import MixFFN
from cleardepth.models.backbone.vit_block import ViTBlock


# ── Smoke-test constants ───────────────────────────────────────────────────
BATCH   = 1
H_IMG   = 64    # Full image height at smoke-test resolution
W_IMG   = 128   # Full image width  at smoke-test resolution
EMBED_C = 64    # Base channel dimension (embed_dim from config)


# ===========================================================================
# Test Group 1: OverlapPatchEmbed
# ===========================================================================

class TestOverlapPatchEmbed:

    def test_output_shape_stage1(self):
        """Stage 1: 7x7 kernel, stride 4 → output at 1/4 scale."""
        embed = OverlapPatchEmbed(in_channels=3, embed_dim=EMBED_C,
                                  patch_size=7, stride=4, padding=3)
        x = torch.randn(BATCH, 3, H_IMG, W_IMG)
        tokens, h_out, w_out = embed(x)

        expected_h = H_IMG // 4   # = 16
        expected_w = W_IMG // 4   # = 32
        assert tokens.shape == (BATCH, expected_h * expected_w, EMBED_C), (
            f"Expected ({BATCH}, {expected_h * expected_w}, {EMBED_C}), "
            f"got {tuple(tokens.shape)}"
        )
        assert h_out == expected_h and w_out == expected_w

    def test_output_shape_inter_stage(self):
        """Inter-stage: 3x3 kernel, stride 2 → output halved spatially."""
        # Used between stages 2→3, 3→4 in the cascaded backbone
        embed = OverlapPatchEmbed(in_channels=EMBED_C, embed_dim=EMBED_C * 2,
                                  patch_size=3, stride=2, padding=1)
        H_in, W_in = H_IMG // 4, W_IMG // 4   # Input at 1/4 scale
        x = torch.randn(BATCH, EMBED_C, H_in, W_in)
        tokens, h_out, w_out = embed(x)

        assert h_out == H_in // 2
        assert w_out == W_in // 2

    def test_gradient_flow(self):
        """Gradients must flow through the patch embedding."""
        embed = OverlapPatchEmbed(in_channels=3, embed_dim=EMBED_C,
                                  patch_size=7, stride=4, padding=3)
        x = torch.randn(BATCH, 3, H_IMG, W_IMG, requires_grad=True)
        tokens, _, _ = embed(x)
        loss = tokens.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_returns_normalised_tokens(self):
        """LayerNorm output should have near-zero mean and near-unit std."""
        embed = OverlapPatchEmbed(in_channels=3, embed_dim=EMBED_C,
                                  patch_size=7, stride=4, padding=3)
        x = torch.randn(BATCH, 3, H_IMG, W_IMG) * 100   # deliberately large
        tokens, _, _ = embed(x)
        # After LayerNorm, each token's features should be normalised
        # (mean ≈ 0, std ≈ 1 per token vector)
        mean = tokens.mean(dim=-1).abs().mean().item()
        assert mean < 1.0, f"LayerNorm not working; mean = {mean:.4f}"


# ===========================================================================
# Test Group 2: EfficientSelfAttention
# ===========================================================================

# Stage 1→4 configurations: (num_heads, reduction_ratio)
ATTN_CONFIGS = [
    pytest.param(1, 8, id="stage1_R8"),
    pytest.param(2, 4, id="stage2_R4"),
    pytest.param(4, 2, id="stage3_R2"),
    pytest.param(8, 1, id="stage4_R1"),
]


class TestEfficientSelfAttention:

    @pytest.mark.parametrize("num_heads,R", ATTN_CONFIGS)
    def test_output_shape(self, num_heads, R):
        """Output must match input shape for all reduction ratios."""
        H, W = H_IMG // 4, W_IMG // 4   # 16 x 32
        C = EMBED_C
        attn = EfficientSelfAttention(dim=C, num_heads=num_heads,
                                      reduction_ratio=R)
        x = torch.randn(BATCH, H * W, C)
        out = attn(x, H, W)
        assert out.shape == x.shape, (
            f"R={R}: expected {tuple(x.shape)}, got {tuple(out.shape)}"
        )

    @pytest.mark.parametrize("num_heads,R", ATTN_CONFIGS)
    def test_gradient_flow(self, num_heads, R):
        """Gradients must flow through attention for all R values."""
        H, W = H_IMG // 4, W_IMG // 4
        C = EMBED_C
        attn = EfficientSelfAttention(dim=C, num_heads=num_heads,
                                      reduction_ratio=R)
        x = torch.randn(BATCH, H * W, C, requires_grad=True)
        out = attn(x, H, W)
        out.mean().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any(), f"NaN gradient at R={R}"

    def test_r1_equals_standard_attention_shape(self):
        """R=1 should behave like standard multi-head self-attention."""
        H, W = H_IMG // 4, W_IMG // 4
        C = EMBED_C
        attn = EfficientSelfAttention(dim=C, num_heads=8, reduction_ratio=1)
        x = torch.randn(BATCH, H * W, C)
        out = attn(x, H, W)
        assert out.shape == x.shape


# ===========================================================================
# Test Group 3: MixFFN
# ===========================================================================

class TestMixFFN:

    def test_output_shape(self):
        """Output shape must match input shape exactly."""
        H, W = H_IMG // 4, W_IMG // 4
        ffn = MixFFN(dim=EMBED_C, mlp_ratio=4.0)
        x = torch.randn(BATCH, H * W, EMBED_C)
        out = ffn(x, H, W)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """With zero-initialised weights, output ≈ input (residual check)."""
        import torch.nn as nn
        H, W = H_IMG // 4, W_IMG // 4
        ffn = MixFFN(dim=EMBED_C, mlp_ratio=4.0)
        # Zero out all learnable parameters
        for p in ffn.parameters():
            nn.init.zeros_(p)
        x = torch.randn(BATCH, H * W, EMBED_C)
        out = ffn(x, H, W)
        # With zero params, fc2(GELU(0)) = 0, so out = x + 0 = x
        torch.testing.assert_close(out, x)

    def test_gradient_flow(self):
        H, W = H_IMG // 4, W_IMG // 4
        ffn = MixFFN(dim=EMBED_C, mlp_ratio=4.0)
        x = torch.randn(BATCH, H * W, EMBED_C, requires_grad=True)
        out = ffn(x, H, W)
        out.mean().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_resolutions(self):
        """Mix-FFN should work at multiple resolutions (unlike pos embed)."""
        ffn = MixFFN(dim=EMBED_C, mlp_ratio=4.0)
        for h, w in [(16, 32), (8, 16), (4, 8), (90, 180)]:
            x = torch.randn(BATCH, h * w, EMBED_C)
            out = ffn(x, h, w)
            assert out.shape == x.shape, f"Failed at resolution {h}×{w}"


# ===========================================================================
# Test Group 4: ViTBlock
# ===========================================================================

# Full per-stage configs for ViTBlock tests
VIT_STAGE_CONFIGS = [
    pytest.param(dict(dim=64,  num_heads=1, reduction_ratio=8), 16, 32, id="stage1"),
    pytest.param(dict(dim=128, num_heads=2, reduction_ratio=4),  8, 16, id="stage2"),
    pytest.param(dict(dim=256, num_heads=4, reduction_ratio=2),  4,  8, id="stage3"),
    pytest.param(dict(dim=512, num_heads=8, reduction_ratio=1),  2,  4, id="stage4"),
]


class TestViTBlock:

    @pytest.mark.parametrize("cfg,H,W", VIT_STAGE_CONFIGS)
    def test_output_shape(self, cfg, H, W):
        """Output shape = input shape for all 4 stage configurations."""
        block = ViTBlock(**cfg)
        x = torch.randn(BATCH, H * W, cfg['dim'])
        out = block(x, H, W)
        assert out.shape == x.shape, (
            f"Expected {tuple(x.shape)}, got {tuple(out.shape)}"
        )

    @pytest.mark.parametrize("cfg,H,W", VIT_STAGE_CONFIGS)
    def test_gradient_flow(self, cfg, H, W):
        """Gradients must flow end-to-end through a ViT block."""
        block = ViTBlock(**cfg)
        x = torch.randn(BATCH, H * W, cfg['dim'], requires_grad=True)
        out = block(x, H, W)
        out.mean().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_drop_path_disabled_at_eval(self):
        """DropPath must be inactive during eval mode (no randomness)."""
        block = ViTBlock(dim=EMBED_C, num_heads=1, reduction_ratio=8,
                         drop_path_rate=0.5)
        x = torch.randn(BATCH, 16 * 32, EMBED_C)
        block.eval()
        with torch.no_grad():
            out1 = block(x, 16, 32)
            out2 = block(x, 16, 32)
        # In eval mode, identical inputs must produce identical outputs
        torch.testing.assert_close(out1, out2)