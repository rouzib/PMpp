"""Sphinx configuration for PM++ documentation.

This file is intentionally CPU-safe for Read the Docs. Notebook execution is
disabled in the Jupyter Book configuration and no GPU runtime is initialized
here.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "PM++"
author = "rouzib"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

# Keep autodoc import-safe on Read the Docs.  Some modules expose optional IO
# and neural-network correction helpers whose dependencies are not required for
# the core package or for rendering the documentation.
autodoc_mock_imports = [
    "h5py",
    "haiku",
    "optax",
]
nb_execution_mode = "off"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,
    "github_user": "rouzib",
    "github_repo": "PMpp_v2",
    "github_version": "master",
    "conf_py_path": "/docs/source/",
}
