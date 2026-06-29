"""
Milestone 1 Smoke Test
======================
Verifies that the project skeleton is correctly set up:
  - All package directories exist with __init__.py
  - Config YAML files exist and load correctly
  - Hydra config composition works (smoke_test overrides defaults)
  - All packages are importable

Run with: pytest tests/test_m1_skeleton.py -v
"""

import os
import pytest


# ---------------------------------------------------------------------------
# Test 1: Package structure
# ---------------------------------------------------------------------------
# Why: Python needs __init__.py in every folder to treat it as a package.
#      If any are missing, imports will fail in later milestones.

EXPECTED_PACKAGES = [
    "cleardepth",
    "cleardepth/models",
    "cleardepth/models/backbone",
    "cleardepth/models/encoders",
    "cleardepth/models/correlation",
    "cleardepth/models/gru",
    "cleardepth/loss",
    "cleardepth/data",
    "cleardepth/training",
    "cleardepth/evaluation",
]


@pytest.mark.parametrize("package_path", EXPECTED_PACKAGES)
def test_package_has_init(package_path):
    """Each package directory must have an __init__.py file."""
    init_file = os.path.join(package_path.replace("/", os.sep), "__init__.py")
    assert os.path.exists(init_file), f"Missing: {init_file}"


# ---------------------------------------------------------------------------
# Test 2: Config files exist
# ---------------------------------------------------------------------------
# Why: If a YAML file is missing or has a typo, Hydra will crash when
#      we try to build the model. Catch it now, not in Milestone 6.

EXPECTED_CONFIGS = [
    "configs/model/cleardepth.yaml",
    "configs/training/default.yaml",
    "configs/smoke_test.yaml",
]


@pytest.mark.parametrize("config_path", EXPECTED_CONFIGS)
def test_config_file_exists(config_path):
    """Each config YAML file must exist."""
    path = config_path.replace("/", os.sep)
    assert os.path.exists(path), f"Missing config: {path}"


# ---------------------------------------------------------------------------
# Test 3: Config files load correctly
# ---------------------------------------------------------------------------
# Why: A YAML file might exist but have bad syntax (wrong indentation,
#      missing colon, etc.). OmegaConf catches these errors.

def test_model_config_loads():
    """Model config must load and contain expected keys."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join("configs", "model", "cleardepth.yaml"))

    # Check that top-level keys exist
    assert "backbone" in cfg, "Model config missing 'backbone' section"
    assert "gru" in cfg, "Model config missing 'gru' section"
    assert "correlation" in cfg, "Model config missing 'correlation' section"
    assert "max_disp" in cfg, "Model config missing 'max_disp'"


def test_training_config_loads():
    """Training config must load and contain expected keys."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join("configs", "training", "default.yaml"))

    assert "lr" in cfg, "Training config missing 'lr'"
    assert "batch_size" in cfg, "Training config missing 'batch_size'"
    assert "max_steps" in cfg, "Training config missing 'max_steps'"


def test_smoke_test_config_loads():
    """Smoke test config must load and contain override values."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join("configs", "smoke_test.yaml"))

    assert "training" in cfg, "Smoke test config missing 'training' overrides"
    assert cfg.training.batch_size == 1, "Smoke test batch_size should be 1"
    assert cfg.training.image_height == 64, "Smoke test height should be 64"
    assert cfg.training.image_width == 128, "Smoke test width should be 128"


# ---------------------------------------------------------------------------
# Test 4: Config composition (smoke_test overrides defaults)
# ---------------------------------------------------------------------------
# Why: The whole point of Hydra is that smoke_test.yaml OVERRIDES
#      default.yaml. We need to verify that merging works correctly.
#      After merging, batch_size should be 1 (from smoke_test),
#      but lr should still be 0.0002 (from default, not overridden).

def test_config_composition():
    """Smoke test config must correctly override training defaults."""
    from omegaconf import OmegaConf

    # Load base configs
    model_cfg = OmegaConf.load(
        os.path.join("configs", "model", "cleardepth.yaml")
    )
    training_cfg = OmegaConf.load(
        os.path.join("configs", "training", "default.yaml")
    )
    smoke_cfg = OmegaConf.load(
        os.path.join("configs", "smoke_test.yaml")
    )

    # Merge: training defaults + smoke test overrides
    merged_training = OmegaConf.merge(training_cfg, smoke_cfg.get("training", {}))

    # Smoke test overrides should win
    assert merged_training.batch_size == 1, "Override failed: batch_size should be 1"
    assert merged_training.image_height == 64, "Override failed: height should be 64"
    assert merged_training.num_workers == 0, "Override failed: num_workers should be 0"

    # Non-overridden values should remain from defaults
    assert merged_training.lr == 0.0002, "Default lr should be preserved"
    assert merged_training.optimizer == "adamw", "Default optimizer should be preserved"

    # Model overrides
    merged_model = OmegaConf.merge(model_cfg, smoke_cfg.get("model", {}))
    assert merged_model.gru.n_gru_iters == 3, "GRU iters should be overridden to 3"
    assert merged_model.backbone.depths == [2, 2, 2, 2], "Backbone depths should be preserved"


# ---------------------------------------------------------------------------
# Test 5: All packages are importable
# ---------------------------------------------------------------------------
# Why: Even with __init__.py files present, there could be path issues.
#      This test actually tries to import each package.

def test_import_cleardepth():
    """Top-level package must be importable."""
    import cleardepth

def test_import_models():
    """Models sub-packages must be importable."""
    import cleardepth.models
    import cleardepth.models.backbone
    import cleardepth.models.encoders
    import cleardepth.models.correlation
    import cleardepth.models.gru

def test_import_other_packages():
    """Loss, data, training, evaluation packages must be importable."""
    import cleardepth.loss
    import cleardepth.data
    import cleardepth.training
    import cleardepth.evaluation


# ---------------------------------------------------------------------------
# Test 6: PyTorch and GPU sanity check
# ---------------------------------------------------------------------------
# Why: Every future milestone needs GPU. If this fails, nothing else matters.

def test_pytorch_gpu():
    """PyTorch must detect CUDA GPU."""
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    assert torch.cuda.device_count() > 0, "No GPU found"

    # Verify GPU computation works
    x = torch.randn(4, 4, device="cuda")
    y = torch.randn(4, 4, device="cuda")
    z = x @ y
    assert z.shape == (4, 4), "GPU matrix multiply failed"