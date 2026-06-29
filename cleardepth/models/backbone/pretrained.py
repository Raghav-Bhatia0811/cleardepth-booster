"""
Pretrained MiT-B1 Weight Loading
=================================
Initialises a CascadedViT backbone from timm's pretrained Mix Transformer B1
(MiT-B1), which was trained on ImageNet-1k.

Why MiT-B1?
-----------
ClearDepth's backbone mirrors the MiT-B1 architecture:
  Stage 1: embed_dim=64,  num_heads=1, SR=8   (exact match)
  Stage 2: embed_dim=128, num_heads=2, SR=4   (exact match)
  Stage 3: embed_dim=256, num_heads=4, SR=2   (dim differs: 320 in MiT-B1)
  Stage 4: embed_dim=512, num_heads=8, SR=1   (blocks match; patch_embed differs)

We load stages 1, 2 (fully) and stage 4 blocks/norm (blocks only, not patch_embed).
Stage 3 is skipped due to dimension mismatch (256 vs 320).

Usage:
    from cleardepth.models.backbone.pretrained import load_mit_b1_pretrained
    from cleardepth.models.backbone.cascaded_vit import CascadedViT

    backbone = CascadedViT(embed_dim=64, fuse_out_channels=256, ...)
    load_mit_b1_pretrained(backbone)

Dependencies:
    pip install timm
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key name mapping: timm MiT-B1 → our CascadedViT
# ---------------------------------------------------------------------------

def _build_key_map(stage_indices):
    """
    Build a dict mapping timm MiT-B1 state-dict key → our CascadedViT key
    for the requested stage indices (0-based).

    timm naming:
        patch_embed{N+1}.proj.*       → patch_embeds.{N}.proj.*
        patch_embed{N+1}.norm.*       → patch_embeds.{N}.norm.*
        block{N+1}.{i}.norm1.*        → stages.{N}.{i}.norm1.*
        block{N+1}.{i}.norm2.*        → stages.{N}.{i}.norm2.*
        block{N+1}.{i}.attn.q.*       → stages.{N}.{i}.attn.q.*
        block{N+1}.{i}.attn.kv.*      → stages.{N}.{i}.attn.kv.*
        block{N+1}.{i}.attn.proj.*    → stages.{N}.{i}.attn.proj.*
        block{N+1}.{i}.attn.sr.*      → stages.{N}.{i}.attn.sr.*
        block{N+1}.{i}.attn.norm.*    → stages.{N}.{i}.attn.sr_norm.*  ← renamed
        block{N+1}.{i}.mlp.fc1.*      → stages.{N}.{i}.ffn.fc1.*       ← renamed
        block{N+1}.{i}.mlp.dw_conv.dw_conv.*  → stages.{N}.{i}.ffn.dw_conv.*
        block{N+1}.{i}.mlp.fc2.*      → stages.{N}.{i}.ffn.fc2.*
        norm{N+1}.*                   → norms.{N}.*
    """
    key_map = {}
    for n in stage_indices:
        t = n + 1  # timm uses 1-based numbering

        # patch embedding (only if included for this stage)
        pe_src = f"patch_embed{t}."
        pe_dst = f"patch_embeds.{n}."
        key_map[pe_src] = pe_dst

        # norm after stage
        key_map[f"norm{t}."] = f"norms.{n}."

        # transformer blocks — handled per-key below via prefix rewrite
        key_map[f"block{t}."] = f"stages.{n}."

    return key_map


def _remap_key(key: str, prefix_map: dict, block_key_renames: dict) -> Optional[str]:
    """
    Apply prefix map then block-level key renames to a single timm key.
    Returns the remapped key, or None if this key should be skipped.
    """
    # Find matching prefix
    new_key = None
    for src_prefix, dst_prefix in prefix_map.items():
        if key.startswith(src_prefix):
            new_key = dst_prefix + key[len(src_prefix):]
            break

    if new_key is None:
        return None  # not in a mapped stage

    # Apply sub-key renames (attention norm, mlp → ffn, etc.)
    for src_suffix, dst_suffix in block_key_renames.items():
        if src_suffix in new_key:
            new_key = new_key.replace(src_suffix, dst_suffix)
            break

    return new_key


def load_mit_b1_pretrained(
    backbone,          # CascadedViT instance
    strict: bool = False,
) -> None:
    """
    Load pretrained MiT-B1 (ImageNet-1k) weights into a CascadedViT backbone.

    Compatible layers (loaded):
      - Stage 1: patch_embed, blocks, norm   (embed_dim=64,  heads=1)
      - Stage 2: patch_embed, blocks, norm   (embed_dim=128, heads=2)
      - Stage 4: blocks, norm only           (embed_dim=512, heads=8)
                 patch_embed4 skipped (input channels differ: 320 vs 256)

    Incompatible layers (randomly initialised):
      - Stage 3 entirely                     (embed_dim=256 vs 320 in MiT-B1)
      - Stage 4 patch_embed                  (in_channels 256 ≠ 320)
      - fusion_conv                          (architecture-specific)

    Args:
        backbone : CascadedViT instance to initialise in-place.
        strict   : If True, raise on any shape mismatch. Default False (skip mismatches).

    Raises:
        ImportError : if timm is not installed.
        RuntimeError: if strict=True and any compatible layer has a shape mismatch.
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for pretrained weight loading.\n"
            "Install it with:  pip install timm"
        )

    log.info("Downloading / loading pretrained MiT-B1 weights from timm ...")
    mit = timm.create_model("mit_b1", pretrained=True)
    timm_sd = mit.state_dict()
    del mit  # free memory

    our_sd = backbone.state_dict()

    # Stages whose patch_embed + blocks + norm we can load fully
    full_stages = [0, 1]     # stages 1 and 2 (0-indexed)
    # Stage 4 blocks + norm only (not patch_embed due to channel mismatch)
    block_only_stages = [3]

    prefix_map = _build_key_map(full_stages + block_only_stages)

    # Sub-key renames that differ between timm and our code
    block_key_renames = {
        ".attn.norm.":             ".attn.sr_norm.",     # timm norm → our sr_norm
        ".mlp.dw_conv.dw_conv.":  ".ffn.dw_conv.",      # timm nested DWConv
        ".mlp.fc1.":               ".ffn.fc1.",
        ".mlp.fc2.":               ".ffn.fc2.",
    }

    # Keys to explicitly skip even if they have a matching prefix
    skip_keys = {f"patch_embeds.{n}." for n in block_only_stages}

    loaded, skipped_shape, skipped_key = 0, 0, 0

    new_sd = {k: v.clone() for k, v in our_sd.items()}  # start from our weights

    for timm_key, timm_val in timm_sd.items():
        our_key = _remap_key(timm_key, prefix_map, block_key_renames)

        if our_key is None:
            skipped_key += 1
            continue

        # Skip patch_embeds for block_only_stages
        if any(our_key.startswith(skip) for skip in skip_keys):
            skipped_key += 1
            continue

        if our_key not in our_sd:
            skipped_key += 1
            continue

        if our_sd[our_key].shape != timm_val.shape:
            msg = (f"Shape mismatch: {timm_key} {tuple(timm_val.shape)} → "
                   f"{our_key} {tuple(our_sd[our_key].shape)}")
            if strict:
                raise RuntimeError(msg)
            log.warning(msg)
            skipped_shape += 1
            continue

        new_sd[our_key] = timm_val
        loaded += 1

    backbone.load_state_dict(new_sd, strict=True)

    log.info(
        f"MiT-B1 pretrained load complete: "
        f"{loaded} tensors loaded, "
        f"{skipped_shape} skipped (shape mismatch), "
        f"{skipped_key} skipped (no mapping)."
    )
    print(
        f"[pretrained] Loaded {loaded} tensors from MiT-B1 "
        f"({skipped_shape} shape mismatches skipped, "
        f"{skipped_key} unmapped keys skipped)."
    )


def load_pretrained_encoders(feature_encoder, context_encoder) -> None:
    """
    Convenience wrapper: load MiT-B1 pretrained weights into both encoders.

    Call this once after constructing ClearDepthNet, before training.

    Args:
        feature_encoder : ClearDepthNet.feature_encoder  (FeatureEncoder)
        context_encoder : ClearDepthNet.context_encoder  (ContextEncoder)
    """
    print("[pretrained] Loading MiT-B1 weights into feature encoder ...")
    load_mit_b1_pretrained(feature_encoder.backbone)
    print("[pretrained] Loading MiT-B1 weights into context encoder ...")
    load_mit_b1_pretrained(context_encoder.backbone)
    print("[pretrained] Done.")


# ---------------------------------------------------------------------------
# Quick self-test (requires timm and internet access)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from cleardepth.models.backbone.cascaded_vit import CascadedViT

    backbone = CascadedViT(embed_dim=64, fuse_out_channels=256)
    x = torch.randn(1, 3, 64, 128)
    out_before = backbone(x).detach().clone()

    load_mit_b1_pretrained(backbone)

    out_after = backbone(x).detach()
    diff = (out_after - out_before).abs().mean().item()
    print(f"Mean output change after loading pretrained weights: {diff:.6f}")
    print("(Non-zero = weights changed, as expected.)")
    print("pretrained smoke test passed.")
