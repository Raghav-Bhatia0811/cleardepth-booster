# Checkpoint Registry

All `.pt` files are excluded from git via `.gitignore` (GitHub 100 MB file-size
limit; each checkpoint is ~349 MB). This file documents what was saved, where,
and what performance it achieved so the checkpoints can be reconstructed if lost.

---

## Scene Flow Monkaa — Pretraining

| Field | Value |
|-------|-------|
| Path on EC2 | `/data/sceneflow_checkpoints/best.pt` |
| Training script | `scripts/pretrain_sceneflow.py` |
| Step | 48,000 |
| Val AvgErr (1/4-scale px) | **0.127 px** |
| Val RMS | 0.332 px |
| Val Bad-0.5 | 4.27 % |
| Data | Scene Flow Monkaa (`/data/monkaa`), val_fraction=0.15, seed=42 |
| Resolution | 360 × 720 (GRU output at 90 × 180) |
| GRU iters (train) | 22 |
| Max steps run | ~48,000 of 300,000 planned |
| Hardware | Tesla T4 GPU |

**Checkpoint format** (keys in the `.pt` file):
```
step, model_state, optimiser_state, scheduler_state, best_val_err
```

**To resume pretraining from this checkpoint**:
```bash
python scripts/pretrain_sceneflow.py \
    --data_root /data/monkaa \
    --batch_size 4 \
    --max_steps 300000 \
    --resume /data/sceneflow_checkpoints/best.pt \
    --ckpt_dir /data/sceneflow_checkpoints
```

---

## Booster GT — Fine-tuning

| Field | Value |
|-------|-------|
| Path on EC2 | `/data/booster_checkpoints/best.pt` |
| Training script | `scripts/train_booster.py` |
| Step | 4,000 |
| Val AvgErr (full-scale px) | **2.12 px** |
| Val RMS | 3.78 px |
| Val Bad-0.5 | 60.93 % |
| Val Bad-1.0 | 40.17 % |
| Val Bad-2.0 | 24.67 % |
| Val Bad-4.0 | 15.38 % |
| Data | Booster balanced (`/data/datasets/booster`), val_fraction=0.15, seed=42 |
| Resolution | 256 × 512 |
| GRU iters (train) | 22 |
| Initialised from | `/data/sceneflow_checkpoints/best.pt` (weights only, via `--init_from`) |
| Max steps run | 20,000 |
| Hardware | Tesla T4 GPU |

> **Note**: `best.pt` was saved at step 4,000 (earliest val checkpoint). The final
> step 20,000 checkpoint is at `/data/booster_checkpoints/step_0020000.pt`.
> Re-evaluate both to determine which to use as the canonical best.

**Checkpoint format** (keys in the `.pt` file):
```
step, model_state, optimiser_state, scheduler_state, best_val_err
```

**To fine-tune further from the step 20,000 checkpoint**:
```bash
python scripts/train_booster.py \
    --data_root /data/datasets/booster \
    --height 256 --width 512 \
    --batch_size 4 \
    --max_steps 50000 \
    --resume /data/booster_checkpoints/step_0020000.pt \
    --ckpt_dir /data/booster_checkpoints
```

**To initialise a fresh Booster run from the SceneFlow pretrain**:
```bash
python scripts/train_booster.py \
    --data_root /data/datasets/booster \
    --height 256 --width 512 \
    --batch_size 4 \
    --max_steps 50000 \
    --init_from /data/sceneflow_checkpoints/best.pt \
    --ckpt_dir /data/booster_checkpoints
```

---

## Git LFS

`git lfs version` was checked at session close. If LFS is available and you want
to track checkpoints in git, run:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add /path/to/best.pt
git commit -m "Add best.pt checkpoint via Git LFS"
git push origin main
```
