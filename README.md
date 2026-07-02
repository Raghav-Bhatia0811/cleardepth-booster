# ClearDepth — Stereo Depth Estimation for Transparent Objects

PyTorch implementation of **ClearDepth** ([arXiv:2409.08926v3](https://arxiv.org/abs/2409.08926)), a stereo depth estimation network designed to handle transparent and reflective objects. The model combines a SegFormer-style multi-scale backbone with RAFT-Stereo-inspired iterative GRU refinement.

---

## Architecture Overview

```
Left + Right stereo pair
        │
        ├─► FeatureEncoder (shared weights) ──► feat_left, feat_right
        │         └─ MixViT backbone: 4-stage ViT with MixFFN
        │
        ├─► ContextEncoder (left only) ──────► c_k (context key)
        │                                      c_r (context residual)
        │                                      c_h (initial hidden state)
        │
        ├─► CorrelationPyramid ──────────────► 4-level cost volume
        │
        └─► PostFusionGRU (3-scale, coarse-to-fine)
                  └─► [d_1, ..., d_N] at 1/4 scale  ←── training output
                  └─► d_final bilinear ×4 to full HxW ←── inference output
```

Key design decisions:
- **Training**: returns all N disparity predictions at 1/4-scale for sequence loss (γ=0.9)
- **Inference** (`test_mode=True`): returns only the final prediction, bilinearly upsampled ×4 — exactly as stated in the Architecture Report ("bilinear upsampled 4× to full H×W")
- **No learned upsampling module** — a ConvexUpsample (RAFT-Stereo style) was evaluated but has no basis in the paper's equations (Eqs. 7–15) or Architecture Report and was removed

---

## Repository Structure

```
cleardepth-booster/
│
├── cleardepth/                    # Main package
│   ├── models/
│   │   ├── cleardepth_net.py      # Full model assembly (entry point)
│   │   ├── backbone/
│   │   │   ├── cascaded_vit.py    # 4-stage SegFormer-style backbone
│   │   │   ├── mix_ffn.py         # MixFFN (depthwise conv + FFN)
│   │   │   └── pretrained.py      # Pretrained weight loading utilities
│   │   ├── encoders/
│   │   │   ├── feature_encoder.py # Shared feature encoder (left + right)
│   │   │   └── context_encoder.py # Context encoder (left only → c_k, c_r, c_h)
│   │   ├── correlation/
│   │   │   └── correlation_pyramid.py  # 4-level cost volume
│   │   └── gru/
│   │       └── post_fusion_gru.py # 3-scale coarse-to-fine GRU refinement
│   │
│   ├── data/
│   │   ├── booster.py             # Booster GT dataset (scene-level train/val split)
│   │   ├── sceneflow_monkaa.py    # Scene Flow Monkaa dataset (PFM reader + splits)
│   │   ├── sceneflow.py           # Scene Flow multi-subset dataset
│   │   ├── sceneflow_sample.py    # Scene Flow sample-pack dataset
│   │   └── transforms.py          # Shared augmentation transforms
│   │
│   ├── loss/
│   │   └── sequence_loss.py       # SequenceLoss: Σ γ^(N-1-i) · L1(pred, gt)
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # AvgErr, RMS, Bad-0.5/1.0/2.0/4.0
│   │   └── visualize.py           # Disparity colormaps and comparison figures
│   │
│   └── training/
│       └── trainer.py             # Training loop utilities
│
├── scripts/
│   ├── pretrain_sceneflow.py      # Stage 1: pretrain on Scene Flow Monkaa
│   ├── train_booster.py           # Stage 2: fine-tune on Booster GT
│   ├── evaluate_sceneflow.py      # Evaluate pretrain checkpoint on Monkaa val
│   ├── evaluate_booster.py        # Evaluate fine-tuned checkpoint on Booster val
│   ├── train_sample.py            # Training on Scene Flow sample pack
│   ├── overfit_train.py           # Overfit smoke-test script
│   ├── evaluate.py                # Generic evaluation entry point
│   └── visualize_results.py       # Standalone result visualiser
│
├── configs/
│   ├── model/cleardepth.yaml      # Architecture hyperparameters (backbone, GRU, corr)
│   ├── data/sceneflow.yaml        # Data config for Scene Flow sample-pack dataset
│   └── training/default.yaml      # Optimizer, LR schedule, batch size, etc.
│
├── tests/                         # pytest test suite (170 tests)
│   ├── test_m1_skeleton.py        # Module import / instantiation smoke tests
│   ├── test_m2_primitives.py      # MixFFN, attention, and low-level ops
│   ├── test_m3_backbone.py        # CascadedViT output shapes and multi-scale fusion
│   ├── test_m4_encoders.py        # Feature and context encoder contracts
│   ├── test_m5_gru.py             # PostFusionGRU shape and iteration count
│   ├── test_m6_full_model.py      # End-to-end model: training vs inference mode
│   ├── test_m7_loss_metrics.py    # SequenceLoss and metric correctness
│   ├── test_m8_data.py            # Dataset split, shapes, and normalisation
│   ├── test_m9_trainer.py         # Trainer integration
│   ├── test_m10_eval.py           # Evaluation pipeline smoke test
│   ├── test_pretrain_sceneflow.py # PFM reader, Monkaa dataset, GT downsampling
│   ├── test_real_forward_pass.py  # Real data forward pass (env-specific)
│   └── test_sample_dataloader.py  # Sample dataloader (env-specific)
│
├── environment.yml                # Conda environment spec
├── pyproject.toml                 # Package build config (pip install -e .)
└── verify_env.py                  # Environment sanity check script
```

---

## Two-Stage Training Pipeline

```
Scene Flow Monkaa          Booster GT
(synthetic, dense GT)      (real transparent objects)
        │                        │
        ▼                        ▼
pretrain_sceneflow.py  ──►  train_booster.py
  50K steps, AdamW            20K steps, AdamW
  1/4-scale sequence loss      fine-tune from SceneFlow ckpt
        │                        │
        ▼                        ▼
sceneflow_checkpoints/       booster_checkpoints/
  best.pt                      best.pt
        │                        │
        ▼                        ▼
evaluate_sceneflow.py        evaluate_booster.py
  (1/4-scale metrics)          (full-res metrics)
```

---

## Environment Setup

### Requirements

- CUDA-capable GPU (paper uses NVIDIA A6000)
- CUDA 12.1
- Conda (recommended) or pip

### Install with Conda (recommended)

```bash
conda env create -f environment.yml
conda activate cleardepth
pip install -e .
```

### Install with pip only

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install einops==0.7.0 timm==0.9.12 omegaconf==2.3.0 hydra-core==1.3.2 \
            wandb==0.16.0 accelerate==0.25.0 opencv-python==4.8.1.78 \
            matplotlib==3.8.0 tqdm==4.66.1 pytest==7.4.3 tensorboard==2.15.0
pip install -e .
```

### Verify installation

```bash
python verify_env.py
pytest tests/ -v --ignore=tests/test_real_forward_pass.py \
                  --ignore=tests/test_sample_dataloader.py
```

Expected: **~170 tests passed**. The two ignored test files require local dataset paths and are environment-specific.

---

## Dataset Setup

### Scene Flow — Monkaa subset (Stage 1 pretraining)

Download from the [Scene Flow dataset page](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).

Required layout:
```
/data/monkaa/
  frames_cleanpass/
    scene_name_1/
      left/   0001.png  0002.png  ...
      right/  0001.png  0002.png  ...
  disparity/
    scene_name_1/
      left/   0001.pfm  0002.pfm  ...
```

Disparities are stored as PFM files (bottom-to-top, endianness from scale sign). The dataset class handles reading and vertical flip automatically.

### Booster GT (Stage 2 fine-tuning)

Download from the [Booster dataset page](https://cvlab-unibo.github.io/booster-web/).

Required layout:
```
/data/booster_gt/
  train/
    scene_name/
      camera_00/
        frames/
          left/  00000.png  00001.png  ...
          right/ 00000.png  00001.png  ...
        disp_00.npy      # float32, shape (3008, 4112)
        mask_00.png      # white = valid pixel
```

The dataset performs scene-level train/val split (default 85%/15%), disparity rescaled by `width/4112` to match the resized image.

---

## Training

### Stage 1 — Pretrain on Scene Flow Monkaa

```bash
python scripts/pretrain_sceneflow.py \
    --data_root /data/monkaa \
    --batch_size 4 \
    --max_steps 300000 \
    --n_gru_iters 22 \
    --lr 2e-4 \
    --ckpt_dir /data/sceneflow_checkpoints \
    --pretrained          # use SegFormer ImageNet weights for backbone
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | required | Path to Monkaa dataset root |
| `--batch_size` | 4 | Per-GPU batch size |
| `--max_steps` | 300000 | Total gradient steps |
| `--n_gru_iters` | 22 | GRU iterations during training |
| `--lr` | 2e-4 | Peak learning rate (OneCycleLR) |
| `--grad_clip` | 1.0 | Gradient clipping norm |
| `--gamma` | 0.9 | Sequence loss decay factor |
| `--ckpt_dir` | `/data/sceneflow_checkpoints` | Checkpoint save directory |
| `--save_every` | 10000 | Save checkpoint every N steps |
| `--resume` | None | Resume from an interrupted run (full state restore) |
| `--val_fraction` | 0.15 | Fraction of scenes held out for validation |

### Stage 2 — Fine-tune on Booster GT

```bash
python scripts/train_booster.py \
    --data_root /data/booster_gt \
    --init_from /data/sceneflow_checkpoints/best.pt \
    --batch_size 4 \
    --max_steps 50000 \
    --ckpt_dir /data/booster_checkpoints
```

Key options (beyond Stage 1):
| Flag | Description |
|------|-------------|
| `--init_from PATH` | Load SceneFlow weights only; step counter and LR schedule start fresh. Use for cross-task fine-tuning. |
| `--resume PATH` | Full state restore (step, optimizer, scheduler). Use to continue an interrupted Booster run. |

> `--init_from` and `--resume` are mutually exclusive.

---

## Evaluation

### Scene Flow Monkaa val split

```bash
python scripts/evaluate_sceneflow.py \
    --ckpt /data/sceneflow_checkpoints/best.pt \
    --data_root /data/monkaa \
    --output_dir eval_results/sceneflow \
    --split val
```

Metrics are computed at **1/4 scale** (the native GRU output resolution) so they match training-time validation logs exactly. Figures are upsampled ×4 for display only.

### Booster GT val split

```bash
python scripts/evaluate_booster.py \
    --ckpt /data/booster_checkpoints/best.pt \
    --data_root /data/booster_gt \
    --output_dir eval_results/booster \
    --split val
```

Metrics computed at **full resolution** (360×720) after bilinear ×4 upsampling (`test_mode=True`).

Both scripts output:
- `evaluation_results.txt` — averaged and per-sample metrics
- `figures/` — 4-panel PNG per sample (Left RGB | Predicted Disp | GT Disp | Error Map)

### Reported metrics

| Metric | Description |
|--------|-------------|
| `AvgErr` | Mean absolute disparity error (pixels) |
| `RMS` | Root mean squared error (pixels) |
| `Bad-0.5` | % pixels with error > 0.5 px |
| `Bad-1.0` | % pixels with error > 1.0 px |
| `Bad-2.0` | % pixels with error > 2.0 px |
| `Bad-4.0` | % pixels with error > 4.0 px |

Invalid pixels (Booster mask, or GT ≤ 0 / > max_disp) are excluded from all metrics.

---

## Configuration

Architecture and training hyperparameters live in `configs/`:

**`configs/model/cleardepth.yaml`** — architecture (do not change unless redesigning the network):
```yaml
backbone:
  embed_dim: 64            # Base channel dim C; stages use C, 2C, 4C, 8C
  depths: [2, 2, 2, 2]    # ViT blocks per stage
  num_heads: [1, 2, 4, 8]
  reduction_ratios: [8, 4, 2, 1]
  mlp_ratio: 4.0
  fuse_out_channels: 256

gru:
  hidden_dim: 128
  n_gru_iters: 22          # Training iterations (paper: 22)
  n_gru_iters_eval: 32     # Inference iterations (paper: 32)
  n_gru_layers: 3          # Multi-scale levels: 1/8, 1/16, 1/32

correlation:
  num_levels: 4
  radius: 4                # Search radius → 9 values per level, 36 total channels

max_disp: 192              # Maximum valid disparity (pixels)

upsample:
  scale: 4                 # Bilinear upsample factor (1/4-scale → full-res)
```

**`configs/training/default.yaml`** — optimizer and schedule:
```yaml
optimizer: adamw
lr: 0.0002
weight_decay: 0.00001
scheduler: onecycle
max_steps: 300000
batch_size: 8
loss_gamma: 0.9
```

---

## Test Suite

```bash
# Full suite (170 tests, ~2 min on CPU)
pytest tests/ -v

# Skip environment-specific tests (no local dataset needed)
pytest tests/ -v --ignore=tests/test_real_forward_pass.py \
                  --ignore=tests/test_sample_dataloader.py

# Run a specific module
pytest tests/test_m6_full_model.py -v
```

Test modules cover:
- Model component shapes and contracts (`test_m1`–`test_m5`)
- End-to-end forward pass in both training and inference mode (`test_m6`)
- Loss function and metric correctness (`test_m7`)
- Booster dataset splits, shapes, and masking (`test_m8`)
- Training loop integration (`test_m9`)
- Evaluation pipeline (`test_m10`)
- PFM reader round-trip, Monkaa dataset, GT downsampling (`test_pretrain_sceneflow`)

---

## Paper Reference

> **ClearDepth: Enhanced Stereo Perception of Transparent Objects for Robotic Manipulation**  
> arXiv:2409.08926v3  
> https://arxiv.org/abs/2409.08926
