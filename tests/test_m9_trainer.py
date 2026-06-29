"""
Milestone 9 Smoke Test — Training Loop
=======================================
Tests the trainer in smoke-test mode: synthetic data, 3 steps, CPU.
Does NOT require a real dataset or GPU.

Run with: pytest tests/test_m9_trainer.py -v
"""

import os
import torch
import tempfile
import pytest
from cleardepth.training.trainer import Trainer, SyntheticStereoDataset


# Minimal model config for fast testing
SMOKE_MODEL_CFG = dict(
    embed_dim=32,
    depths=[1, 1, 1, 1],
    num_heads=[1, 2, 4, 8],
    reduction_ratios=[8, 4, 2, 1],
    hidden_dim=64,
    n_gru_iters=2,
    corr_levels=4,
    corr_radius=4,
)


def make_trainer(tmpdir, **kwargs):
    defaults = dict(
        model_cfg=SMOKE_MODEL_CFG,
        lr=1e-3,
        max_steps=3,
        n_gru_iters=2,
        batch_size=1,
        num_workers=0,
        checkpoint_dir=str(tmpdir),
        save_every=999999,   # Don't save during tests
        use_wandb=False,
        smoke_test=True,
        device='cpu',
        log_every=1,
    )
    defaults.update(kwargs)
    return Trainer(**defaults)


class TestSyntheticDataset:

    def test_length(self):
        ds = SyntheticStereoDataset(length=10, height=64, width=128)
        assert len(ds) == 10

    def test_sample_shapes(self):
        ds = SyntheticStereoDataset(length=4, height=64, width=128)
        sample = ds[0]
        assert sample['left'].shape      == (3, 64, 128)
        assert sample['right'].shape     == (3, 64, 128)
        assert sample['disparity'].shape == (1, 64, 128)

    def test_disparity_all_valid(self):
        """Synthetic disparity must be > 0 (all pixels valid for loss)."""
        ds = SyntheticStereoDataset(length=4, height=64, width=128,
                                    max_disp=64.0)
        sample = ds[0]
        assert sample['disparity'].min() > 0


class TestTrainer:

    def test_smoke_train_completes(self, tmp_path):
        """Smoke-test training must complete max_steps without crashing."""
        trainer = make_trainer(tmp_path)
        trainer.train()   # Should not raise
        assert trainer.global_step == 3

    def test_loss_is_finite(self, tmp_path):
        """Loss must be finite (not NaN or Inf) at every step."""
        losses = []
        original_step = Trainer._step

        def patched_step(self, batch):
            info = original_step(self, batch)
            losses.append(info['loss'])
            return info

        Trainer._step = patched_step
        try:
            trainer = make_trainer(tmp_path)
            trainer.train()
        finally:
            Trainer._step = original_step

        assert len(losses) == 3
        for i, l in enumerate(losses):
            assert torch.isfinite(torch.tensor(l)), \
                f"Non-finite loss at step {i}: {l}"

    def test_model_weights_change_after_training(self, tmp_path):
        """Weights must be updated — if they don't, backward is broken."""
        trainer = make_trainer(tmp_path)

        # Snapshot initial weights
        param = next(trainer.model.parameters())
        before = param.data.clone()

        trainer.train()

        after = param.data
        assert not torch.allclose(before, after), \
            "Model weights did not change — optimiser or backward is broken"

    def test_checkpoint_save_and_load(self, tmp_path):
        """Saving then loading a checkpoint must restore exact state."""
        trainer = make_trainer(tmp_path, save_every=1, max_steps=2)

        # Manually trigger a save
        trainer.global_step = 1
        trainer._save_checkpoint()

        ckpt_path = os.path.join(str(tmp_path), 'step_0000001.pt')
        assert os.path.exists(ckpt_path), "Checkpoint file not created"

        # Load into a fresh trainer
        trainer2 = make_trainer(tmp_path, resume_from=ckpt_path)
        assert trainer2.global_step == 1

        # Verify model weights match
        for p1, p2 in zip(trainer.model.parameters(),
                          trainer2.model.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_lr_changes_over_steps(self, tmp_path):
        """OneCycleLR must update the learning rate across steps."""
        lrs = []
        original_step = Trainer._step

        def patched_step(self, batch):
            info = original_step(self, batch)
            lrs.append(info['lr'])
            return info

        Trainer._step = patched_step
        try:
            trainer = make_trainer(tmp_path, max_steps=5)
            trainer.train()
        finally:
            Trainer._step = original_step

        # LR should vary (OneCycleLR changes it every step)
        assert len(set(round(lr, 10) for lr in lrs)) > 1, \
            "Learning rate never changed — scheduler not working"

    def test_no_dataset_without_smoke_raises(self, tmp_path):
        """Passing smoke_test=False with no dataset must raise ValueError."""
        with pytest.raises(ValueError, match="train_dataset must be provided"):
            Trainer(
                model_cfg=SMOKE_MODEL_CFG,
                smoke_test=False,
                train_dataset=None,
                use_wandb=False,
                checkpoint_dir=str(tmp_path),
                device='cpu',
            )