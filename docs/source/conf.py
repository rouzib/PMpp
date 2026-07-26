"""Canonical Sphinx configuration for the PM++ documentation site."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "PM++"
author = "PM++ contributors"
copyright = "2026, PM++ contributors"

try:
    release = version("pmpp")
except PackageNotFoundError:
    release = "0.1.3"
version = release

extensions = [
    # myst_nb registers the MyST Markdown parser; registering the base parser
    # again causes duplicate-parser warnings.
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
source_encoding = "utf-8"
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    # Retired Jupyter Book / generated-API entry points. These can be removed
    # after downstream branches no longer carry them.
    "_config.yml",
    "_toc.yml",
    "api.md",
]

# Read the Docs renders committed outputs. Notebook execution is an explicit
# local validation step because the hosted builder has no multi-GPU runtime.
nb_execution_mode = "off"
nb_execution_raise_on_error = True
nb_merge_streams = True

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autodoc_class_signature = "separated"
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

# Optional correction/data-adapter dependencies are imported lazily by PM++.
# Mocking them keeps the core API reference buildable without expanding the
# documentation environment into a training stack.
autodoc_mock_imports = ["h5py", "haiku", "optax"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Opt in to the Sphinx 8 timeout status now.  Timeouts still fail the link-check
# builder, but Sphinx 7 no longer emits its default-transition warning.
linkcheck_report_timeouts_as_broken = False

html_theme = "pydata_sphinx_theme"
html_logo = "_static/pmpp-logo.png"
html_favicon = "_static/pmpp-logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_sidebars = {
    # The PyData theme's stock sidebar starts below the current top-level
    # section.  PM++ uses a persistent master tree instead so readers can move
    # between the five documentation areas without returning to the header.
    "**": ["sidebar-master.html"],
}
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "show_nav_level": 3,
    "show_toc_level": 2,
    "sidebar_includehidden": True,
    # The root toctree contains exactly five major sections.  Keep all five in
    # the header instead of moving the last one into a More dropdown.
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rouzib/PMpp",
            "icon": "fa-brands fa-github",
        },
    ],
}
html_title = f"PM++ {release} documentation"
html_show_sourcelink = True
html_last_updated_fmt = "%Y-%m-%d"

# Sphinx defaults to a MathJax CDN. Keep equation rendering available in local
# and offline builds by shipping the matching MathJax 3 component and fonts.
mathjax_path = "vendor/mathjax/tex-mml-chtml.js"

# sphinxcontrib-mermaid 2.x expects an ES module with a default export.  The
# tiny local adapter wraps the vendored self-contained Mermaid browser bundle,
# keeping diagram rendering, theme switching, and fullscreen support offline.
mermaid_use_local = "vendor/mermaid/mermaid-module.js"
mermaid_output_format = "raw"
mermaid_init_config = {
    "startOnLoad": False,
    "securityLevel": "strict",
    # Keep connectors visible even before the theme-specific CSS override is
    # applied.  This mid-tone has sufficient contrast in both site themes.
    "themeVariables": {"lineColor": "#6B7280"},
}
mermaid_light_theme = "neutral"
mermaid_dark_theme = "dark"
mermaid_fullscreen = True

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
