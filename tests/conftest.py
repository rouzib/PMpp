"""pytest configuration file.

This file is loaded by pytest before any test modules, so it is the right place
to set environment variables that must be in place before JAX is first imported.
"""

import os

# Pylians needs the HDF5 compression filters to be registered before h5py is
# imported.  pytest imports this file before collecting any test modules, which
# prevents collection order from deciding whether test_power_spectrum works.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    # The power-spectrum tests will report the missing dev dependency directly.
    pass

# The PMWD baseline lives under tests as comparison support code.  Its helper
# module follows PMWD's upstream naming, but is not itself a pytest test module.
collect_ignore = ["pmwd/test_util.py"]

# Do not preallocate all GPU memory; allows multiple test processes to coexist.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
