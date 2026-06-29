"""
ClearDepth Environment Verification Script
Run this to confirm your setup is complete and working.

Usage: python verify_env.py
"""

import sys


def check_python():
    """Check Python version is 3.10.x"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major != 3 or version.minor != 10:
        print(f"  WARNING: Expected Python 3.10.x, got {version.major}.{version.minor}")
        return False
    print("  OK")
    return True


def check_torch():
    """Check PyTorch and CUDA are working"""
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")

        if not torch.cuda.is_available():
            print("  WARNING: CUDA is not available. GPU training won't work.")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram:.1f} GB")

        # Quick GPU computation test
        x = torch.randn(2, 3).cuda()
        y = torch.randn(3, 2).cuda()
        z = x @ y  # Matrix multiply on GPU
        assert z.shape == (2, 2), "GPU computation failed"
        print("  GPU computation: OK")
        return True

    except ImportError:
        print("  ERROR: PyTorch not installed")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_libraries():
    """Check all required libraries can be imported"""
    libraries = {
        "einops": "Tensor reshaping (used in ViT backbone)",
        "timm": "PyTorch Image Models (ViT utilities)",
        "hydra": "Config management",
        "omegaconf": "Config objects",
        "cv2": "OpenCV — image I/O",
        "matplotlib": "Plotting disparity maps",
        "tqdm": "Progress bars",
        "pytest": "Unit testing",
    }

    all_ok = True
    print("Checking libraries:")

    for lib_name, purpose in libraries.items():
        try:
            __import__(lib_name)
            print(f"  {lib_name}: OK  ({purpose})")
        except ImportError:
            print(f"  {lib_name}: MISSING  ({purpose})")
            all_ok = False

    return all_ok


def check_configs():
    """Check that config YAML files exist and can be loaded"""
    from omegaconf import OmegaConf
    import os

    config_files = [
        "configs/model/cleardepth.yaml",
        "configs/training/default.yaml",
        "configs/smoke_test.yaml",
    ]

    all_ok = True
    print("Checking config files:")

    for config_path in config_files:
        # Handle both forward and backslash for Windows
        config_path_os = config_path.replace("/", os.sep)

        if not os.path.exists(config_path_os):
            print(f"  {config_path}: MISSING")
            all_ok = False
            continue

        try:
            cfg = OmegaConf.load(config_path_os)
            print(f"  {config_path}: OK")
        except Exception as e:
            print(f"  {config_path}: ERROR — {e}")
            all_ok = False

    return all_ok


def check_package_structure():
    """Check that all __init__.py files exist"""
    import os

    packages = [
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

    all_ok = True
    print("Checking package structure:")

    for package in packages:
        init_path = os.path.join(package.replace("/", os.sep), "__init__.py")
        if os.path.exists(init_path):
            print(f"  {package}/: OK")
        else:
            print(f"  {package}/: MISSING __init__.py")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("  ClearDepth Environment Verification")
    print("=" * 60)
    print()

    results = []

    results.append(("Python", check_python()))
    print()
    results.append(("PyTorch + CUDA", check_torch()))
    print()
    results.append(("Libraries", check_libraries()))
    print()
    results.append(("Config files", check_configs()))
    print()
    results.append(("Package structure", check_package_structure()))
    print()

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "X"
        print(f"  [{icon}] {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All checks passed! Ready for Milestone 1 smoke test.")
    else:
        print("  Some checks failed. Fix the issues above before continuing.")

    print("=" * 60)