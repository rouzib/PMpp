"""Sphinx configuration for PM++ documentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "PM++"
author = "rouzib"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "description"
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

autodoc_mock_imports = ["h5py", "haiku", "optax"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_config.yml",
    "_toc.yml",
    "api.md",
    "citation.md",
    "contributing.md",
    "development/*",
    "examples/*",
    "faq.md",
    "getting_started.md",
    "glossary.md",
    "how_to/*",
    "installation.md",
    "intro.md",
    "introduction/*",
    "license.md",
    "multigpu/*",
    "quickstart.md",
    "references.md",
    "release.md",
    "theory/*",
    "tutorials/*",
]
html_theme = "pydata_sphinx_theme"
html_logo = "https://images.nicolaspayot.ca/others/83E87541-E945-4755-B4DD-ED4CE0E130D4.png"
html_static_path = ["_static"]
