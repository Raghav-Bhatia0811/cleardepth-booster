"""
ClearDepth Trainer
==================
Full training loop with:
  - Sequence loss + gradient clipping
  - OneCycleLR scheduler
  - Checkpoint saving / resuming
  - Weights & Biases logging
  - Smoke-test mode (synthetic data, 10 steps, no dataset required)

Paper reference: Section IV-A (Training Details)
  - AdamW optimiser, lr=2e-4, weight_decay=1e-5
  - OneCycleLR scheduler
  - Gradient clipping at max_norm=1.0
  - 300K pre-training steps on Scene Flow + CREStereo
  - 14K fine-tuning steps on SynClearDepth
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional
import logging

from ..models.cleardepth_net import ClearDepthNet
from ..loss.sequence_loss import SequenceLoss
from ..evaluation.metrics import compute_metrics, format_metrics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic dataset for smoke-test mode
# ---------------------------------------------------------------------------

class SyntheticStereoDataset(Dataset):
    """
    Generates random stereo pairs on-the-fly.
    Used for smoke-test mode — no real data needed.

    Args:
        length      : Number of samples to generate.
        height      : Image height.
        width       : Image width.
        max_disp    : Maximum disparity value for synthetic gt.
    """

    def __init__(self, length: int = 32, height: int = 64,
                 width: int = 128, max_disp: float = 64.0):
        self.length   = length
        self.height   = height
        self.width    = width
        self.max_disp = max_disp

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        left  = torch.randn(3, self.height, self.width)
        right = torch.randn(3, self.height, self.width)
        # Disparity: random values in (1, max_disp) — all valid
        disp  = torch.rand(1, self.height, self.width) * (self.max_disp - 1) + 1
        return {'left': left, 'right': right, 'disparity': disp}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full ClearDepth training loop.

    Args:
        # Model
        model_cfg       : Dict of kwargs for ClearDepthNet.__init__.

        # Training
        lr              : Peak learning rate for OneCycleLR.
        weight_decay    : AdamW weight decay.
        max_steps       : Total training steps.
        n_gru_iters     : GRU iterations per forward pass (training).
        grad_clip       : Max gradient norm (default 1.0).
        gamma           : Sequence loss decay factor (default 0.9).
        max_disp        : Max valid disparity for loss masking.

        # Data
        train_dataset   : PyTorch Dataset. If None and smoke_test=True,
                          a synthetic dataset is used automatically.
        batch_size      : Training batch size.
        num_workers     : DataLoader worker processes.

        # Checkpointing
        checkpoint_dir  : Directory to save checkpoints.
        save_every      : Save checkpoint every N steps.
        resume_from     : Path to checkpoint file to resume from.

        # Logging
        use_wandb       : Whether to log to Weights & Biases.
        wandb_project   : W&B project name.
        wandb_run_name  : W&B run name.
        log_every       : Log metrics every N steps.

        # Mode
        smoke_test      : If True, use synthetic data and run only
                          max_steps (default 10) steps at small resolution.
        device          : 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        # Model
        model_cfg: dict = None,
        # Training
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        max_steps: int = 300_000,
        n_gru_iters: int = 22,
        grad_clip: float = 1.0,
        gamma: float = 0.9,
        max_disp: float = 192.0,
        # Data
        train_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        # Checkpointing
        checkpoint_dir: str = 'checkpoints',
        save_every: int = 5000,
        resume_from: Optional[str] = None,
        # Logging
        use_wandb: bool = False,
        wandb_project: str = 'cleardepth',
        wandb_run_name: Optional[str] = None,
        log_every: int = 100,
        # Mode
        smoke_test: bool = False,
        device: str = 'cuda',
    ):
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.max_steps      = max_steps
        self.n_gru_iters    = n_gru_iters
        self.grad_clip      = grad_clip
        self.gamma          = gamma
        self.max_disp       = max_disp
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.checkpoint_dir = checkpoint_dir
        self.save_every     = save_every
        self.log_every      = log_every
        self.use_wandb      = use_wandb
        self.smoke_test     = smoke_test

        # Device setup
        self.device = torch.device(
            device if torch.cuda.is_available() and device == 'cuda'
            else 'cpu'
        )
        log.info(f"Training on: {self.device}")

        # ── Model ──────────────────────────────────────────────────────────
        if model_cfg is None:
            model_cfg = {}
        self.model = ClearDepthNet(**model_cfg).to(self.device)
        log.info(f"Model parameters: {self.model.param_count()['total']:,}")

        # ── Loss ───────────────────────────────────────────────────────────
        self.loss_fn = SequenceLoss(gamma=self.gamma, max_disp=self.max_disp)

        # ── Dataset ────────────────────────────────────────────────────────
        if smoke_test:
            train_dataset = SyntheticStereoDataset(
                length=max(32, batch_size * 4),
                height=64, width=128,
                max_disp=64.0,
            )
            self.num_workers = 0   # No multiprocessing needed for synthetic data

        if train_dataset is None:
            raise ValueError(
                "train_dataset must be provided when smoke_test=False. "
                "Pass a SceneFlowDataset (or similar) instance."
            )

        self.loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=True,
        )

        # ── Optimiser ──────────────────────────────────────────────────────
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # ── Scheduler ──────────────────────────────────────────────────────
        # OneCycleLR requires warmup phase to be at least 1 step.
        # pct_start=0.05 can round to 0 for very small max_steps (e.g. 20).
        # We clamp it so warmup is always at least 1 step.
        warmup_steps = max(1, int(0.05 * max_steps))
        pct_start    = warmup_steps / max_steps

        self.scheduler = OneCycleLR(
            self.optimiser,
            max_lr=lr,
            total_steps=max_steps + 1,
            pct_start=pct_start,
            anneal_strategy='cos',
        )

        self.global_step = 0

        # ── Resume from checkpoint ─────────────────────────────────────────
        if resume_from is not None:
            self._load_checkpoint(resume_from)

        # ── W&B ────────────────────────────────────────────────────────────
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        'lr': lr, 'batch_size': batch_size,
                        'max_steps': max_steps, 'n_gru_iters': n_gru_iters,
                        'gamma': gamma, **model_cfg,
                    }
                )
                self.wandb = wandb
                log.info("Weights & Biases logging enabled.")
            except ImportError:
                log.warning("wandb not installed — logging disabled.")
                self.use_wandb = False
        else:
            self.wandb = None

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Core training step ─────────────────────────────────────────────────

    def _step(self, batch: dict) -> dict:
        """
        Run one forward + backward pass.

        Args:
            batch : Dict with keys 'left', 'right', 'disparity'.

        Returns:
            Dict with 'loss' and any metrics.
        """
        left  = batch['left'].to(self.device)       # (B, 3, H, W)
        right = batch['right'].to(self.device)      # (B, 3, H, W)
        gt    = batch['disparity'].to(self.device)  # (B, 1, H, W)

        # Forward pass — training mode returns all N predictions
        preds = self.model(
            left, right,
            n_iters=self.n_gru_iters,
            test_mode=False,
        )

        # Model outputs at 1/4 scale — downsample gt to match.
        # Use nearest interpolation to preserve exact disparity values,
        # then scale the disparity magnitudes proportionally.
        _, _, H_pred, W_pred = preds[0].shape
        disp_scale = W_pred / gt.shape[-1]   # e.g. 0.25 for 1/4 scale
        if gt.shape[-2] != H_pred or gt.shape[-1] != W_pred:
            gt = torch.nn.functional.interpolate(
                gt, size=(H_pred, W_pred), mode='nearest'
            ) * disp_scale

        # Sequence loss across all GRU iterations
        loss = self.loss_fn(preds, gt)

        # Backward pass
        self.optimiser.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients in GRU
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimiser.step()
        self.scheduler.step()

        # Compute metrics on final prediction (no gradient needed)
        # max_disp also scaled to match the downsampled gt resolution
        with torch.no_grad():
            metrics = compute_metrics(
                preds[-1].detach(), gt,
                max_disp=self.max_disp * disp_scale,
            )

        return {
            'loss': loss.item(),
            'lr'  : self.scheduler.get_last_lr()[0],
            **metrics,
        }
    # ── Checkpoint management ──────────────────────────────────────────────

    def _save_checkpoint(self):
        path = os.path.join(
            self.checkpoint_dir, f'step_{self.global_step:07d}.pt'
        )
        torch.save({
            'step'           : self.global_step,
            'model_state'    : self.model.state_dict(),
            'optimiser_state': self.optimiser.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }, path)
        log.info(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimiser.load_state_dict(ckpt['optimiser_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.global_step = ckpt['step']
        log.info(f"Resumed from step {self.global_step}: {path}")

    # ── Main training loop ─────────────────────────────────────────────────

    def train(self):
        """Run the full training loop up to max_steps."""
        self.model.train()
        log.info(
            f"Starting training — "
            f"{'SMOKE TEST MODE' if self.smoke_test else 'FULL TRAINING'} | "
            f"max_steps={self.max_steps} | device={self.device}"
        )

        step_times = []

        while self.global_step < self.max_steps:
            for batch in self.loader:
                if self.global_step >= self.max_steps:
                    break

                t0 = time.perf_counter()
                info = self._step(batch)
                step_time = time.perf_counter() - t0
                step_times.append(step_time)

                self.global_step += 1

                # ── Logging ────────────────────────────────────────────────
                if self.global_step % self.log_every == 0 or self.smoke_test:
                    avg_time = sum(step_times[-50:]) / len(step_times[-50:])
                    log.info(
                        f"Step {self.global_step:>7d}/{self.max_steps} | "
                        f"Loss={info['loss']:.4f} | "
                        f"LR={info['lr']:.2e} | "
                        f"{format_metrics(info)} | "
                        f"{avg_time*1000:.0f}ms/step"
                    )

                    if self.use_wandb and self.wandb is not None:
                        self.wandb.log(info, step=self.global_step)

                # ── Checkpointing ──────────────────────────────────────────
                if (self.global_step % self.save_every == 0 and
                        not self.smoke_test):
                    self._save_checkpoint()

        # Save final checkpoint
        if not self.smoke_test:
            self._save_checkpoint()

        log.info("Training complete.")

        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()