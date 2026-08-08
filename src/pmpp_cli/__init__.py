"""GPU-free bootstrap for PM++ production command-line entry points."""

import os
import sys
from typing import NoReturn


def main() -> NoReturn:
    """Re-exec the H200 supervisor without exposing CUDA to package imports."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["JAX_PLATFORMS"] = "cpu"
    # JAX's CUDA plugin probes cuInit even when no devices are visible. The
    # supervisor does not execute JAX, and workers remove this temporary
    # bootstrap-only setting before exposing their selected GPUs.
    environment["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    os.execve(sys.executable, [sys.executable, "-m", "pmpp.forward_cli", *sys.argv[1:]], environment, )
