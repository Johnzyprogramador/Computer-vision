#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

cache_root = Path(".cache").resolve()
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:
        return f"MISSING ({exc})"


def main():
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Platform:     {platform.platform()}")
    for module in (
        "torch",
        "torchvision",
        "cv2",
        "ultralytics",
        "segmentation_models_pytorch",
        "matplotlib",
        "gradio",
    ):
        print(f"{module:14}{version(module)}")

    try:
        import torch

        if torch.cuda.is_available():
            device = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "Apple Metal (MPS)"
        else:
            device = "CPU"
        print(f"Training device: {device}")
    except ImportError:
        print("Training device: unavailable (PyTorch missing)")


if __name__ == "__main__":
    main()
