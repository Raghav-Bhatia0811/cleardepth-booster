"""
scripts/train_sample.py
========================
Short local training run using the Scene Flow sample pack.

Purpose: Verify that
  - Loss decreases over steps (gradients flowing correctly)
  - No NaN/Inf errors during training
  - Checkpointing works
  - Logging works

This is NOT full training. It is a sanity check before cloud training.

Run with:
    conda activate cleardepth
    cd C:\\Users\\ragha\\cleardepth
    python scripts/train_sample.py
"""

import os
import sys
import logging

# Make sure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from cleardepth.data.sceneflow_sample import SceneFlowSampleDataset
from cleardepth.training.trainer import Trainer

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────

SAMPLE_ROOT = "C:/Users/ragha/Downloads/Sampler/Sampler"
SUBSETS     = ["Monkaa", "FlyingThings3D"]

# Small model config matching your cleardepth.yaml
MODEL_CFG = dict(
    in_channels      = 3,
    embed_dim        = 64,
    depths           = [2, 2, 2, 2],
    num_heads        = [1, 2, 4, 8],
    reduction_ratios = [8, 4, 2, 1],
    mlp_ratio        = 4.0,
    drop_rate        = 0.0,
    drop_path_rate   = 0.1,
    hidden_dim       = 128,
    n_gru_layers     = 3,
    n_gru_iters      = 3,    # keep small for fast local steps
    corr_levels      = 4,
    corr_radius      = 4,
)

TRAIN_CFG = dict(
    lr            = 2e-4,
    weight_decay  = 1e-5,
    max_steps     = 20,
    n_gru_iters   = 3,
    grad_clip     = 1.0,
    gamma         = 0.9,
    max_disp      = 192.0,
    batch_size    = 1,
    num_workers   = 0,
    save_every    = 9999,
    log_every     = 1,
    use_wandb     = False,
    smoke_test    = False,
    device        = 'cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir= 'checkpoints/sample_run',
)


def main():
    log.info("=" * 60)
    log.info("ClearDepth — Short Local Training Run")
    log.info("=" * 60)
    log.info(f"Device : {TRAIN_CFG['device']}")
    log.info(f"Steps  : {TRAIN_CFG['max_steps']}")
    log.info(f"Data   : {SAMPLE_ROOT}")

    # ── Build dataset ──────────────────────────────────────────────────────
    log.info("Loading sample pack dataset...")
    dataset = SceneFlowSampleDataset(
        root=SAMPLE_ROOT,
        subsets=SUBSETS,
        pass_name='RGB_cleanpass',
    )
    log.info(f"Dataset size: {len(dataset)} samples")

    # ── Build trainer ──────────────────────────────────────────────────────
    log.info("Building trainer...")
    trainer = Trainer(
        model_cfg     = MODEL_CFG,
        train_dataset = dataset,
        **TRAIN_CFG,
    )

    # ── Print parameter count ──────────────────────────────────────────────
    counts = trainer.model.param_count()
    log.info("Parameter counts:")
    log.info(f"  feature_encoder : {counts['feature_encoder']:>12,}")
    log.info(f"  context_encoder : {counts['context_encoder']:>12,}")
    log.info(f"  gru             : {counts['gru']:>12,}")
    log.info(f"  total           : {counts['total']:>12,}")

    # ── Run training ───────────────────────────────────────────────────────
    log.info("Starting training...")
    log.info("-" * 60)
    trainer.train()
    log.info("-" * 60)

    # ── Check loss decreased ───────────────────────────────────────────────
    log.info("Training complete!")
    log.info("")
    log.info("What to look for in the output above:")
    log.info("  ✅ Loss should trend DOWNWARD over 20 steps")
    log.info("  ✅ No NaN or Inf in loss values")
    log.info("  ✅ LR should start small, rise, then begin to fall")
    log.info("  ✅ No CUDA out of memory errors")


if __name__ == '__main__':
    main()