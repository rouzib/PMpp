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

## Authoring rules

- Teach one concept once, then link to it from API/notebooks.
- Use explicit, runnable `pmpp.<module>` imports and small deterministic seeds.
- State shapes, units, static/JIT assumptions, hardware, and capacity behavior.
- Mark experimental interfaces and distinguish scientific model choices from
  runtime choices.
- Keep API pages curated; do not expose every private helper with blanket
  `:members:` directives.
- Use NumPy-style public docstrings and fully qualified cross-references.
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
