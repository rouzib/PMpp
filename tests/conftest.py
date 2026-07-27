"""pytest configuration file.

This file is loaded by pytest before any test modules, so it is the right place
to set environment variables that must be in place before JAX is first imported.
"""

import os

# The PMWD baseline lives under tests as comparison support code.  Its helper
# module follows PMWD's upstream naming, but is not itself a pytest test module.
collect_ignore = ["pmwd/test_util.py"]

# Do not preallocate all GPU memory; allows multiple test processes to coexist.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
