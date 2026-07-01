"""PM++ public Python package.

The project is named PM++, while the importable package name is ``pmpp``.
"""

from .configuration import Configuration
from .multigpu_configuration import MultiGPUConfiguration

__all__ = ["Configuration", "MultiGPUConfiguration"]
