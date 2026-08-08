"""PM++ public Python package.

The project is named PM++, while the importable package name is ``pmpp``.
"""

from .core import Configuration
from .distributed import MultiGPUConfiguration

__all__ = ["Configuration", "MultiGPUConfiguration", ]
