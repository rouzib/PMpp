# Contributing

Use `pmpp` imports for the active implementation. Keep `pmwd/` available as a reference implementation and avoid moving it into the public PM++ API.

For documentation changes, build locally with:

```bash
jupyter-book config sphinx docs/source/
sphinx-apidoc -f -o docs/source/api src/pmpp
sphinx-build -b html docs/source docs/_build/html
```

Do not add GPU-only dependencies to documentation builds.
