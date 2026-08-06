# Documentation

PM++ documentation uses plain Sphinx, MyST-NB, and the PyData theme. The
committed `docs/source/conf.py` and explicit toctree hierarchy rooted at
`docs/source/index.md` are the only configuration and navigation sources.

## Local build

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs/source docs/build/html
python -m http.server --directory docs/build/html 8000
```

Treat every warning as a defect. A documentation build must not invoke code
generation that rewrites `conf.py`, API pages, or tracked sources.

The `docs` extra contains only the packages required to build the Sphinx site.
When a documentation change also modifies Python source or docstrings, install
the contributor environment and run the source-formatting check first:

```bash
python -m pip install -e ".[dev,docs]"
python -m yapf --diff --recursive src tests
```

YAPF is intentionally supplied by the `dev` extra rather than the Read the Docs
requirements because the hosted build renders source and does not rewrite it.

## Authoring rules

- Teach one concept once, then link to it from API/notebooks.
- Use explicit, runnable `pmpp.<module>` imports and small deterministic seeds.
- State shapes, units, static/JIT assumptions, hardware, and capacity behavior.
- Mark experimental interfaces and distinguish scientific model choices from
  runtime choices.
- Keep API pages curated; do not expose every private helper with blanket
  `:members:` directives.
- Use NumPy-style public docstrings and fully qualified cross-references.
- Format modified Python source and tests with the YAPF configuration in
  `pyproject.toml`.
- Avoid absolute paths, local account names, external datasets, and mojibake.
- Link the canonical arXiv paper instead of bundling its PDF.

## Notebook policy

Read the Docs never executes notebooks. Commit validated outputs and provenance:
PM++ commit, seed, JAX version, backend, device count/model, configuration,
capacities, and execution date. Reject notebooks with error outputs, stale
kernels, hidden external data, absolute paths, or undocumented optional
dependencies.

All notebooks are re-executed in clean temporary copies with exactly two
selected GPUs, one heavy process at a time. The hardware requirement must be
visible before the first code cell.

## Diagram policy

Use the vendored Mermaid runtime and diagrams with one reading direction,
roughly four to seven nodes, a shared color legend, and a textual equivalent.
Put equations and function/file anchors in surrounding prose, not inside dense
nodes. Verify diagrams in light/dark themes and at narrow widths.
