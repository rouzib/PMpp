"""pytest configuration file.

This file is loaded by pytest before any test modules, so it is the right place
to set environment variables that must be in place before JAX is first imported.
"""

import os

# Disable XLA runtime fusion to work around a cuFFT "scratch allocator" failure
# observed on certain GPU hardware (e.g. GTX 1650 / cuFFT 12) when a large
# jit+grad computation contains irfftn calls.  The flag is a no-op on hardware
# that does not exhibit the issue, so it is safe to set unconditionally.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_use_runtime_fusion=false")

# Do not preallocate all GPU memory; allows multiple test processes to coexist.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
