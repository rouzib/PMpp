# Documentation

## Repository inventory

The active package is `pmpp` under `src/pmpp`. The repository also keeps `pmwd/` as a validation reference implementation. Documentation is built with Sphinx and MyST from `docs/source`.

## Local build

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

## Authoring rules

- Keep onboarding pages short and link to deeper pages.
- Use valid Python imports (`pmpp`, never `PM++`).
- Mark multi-GPU examples as requiring at least two GPUs.
- Do not put local absolute paths in user docs.
- Keep notebooks or MyST examples deterministic and small.
- Prefer CPU-safe docs builds; do not execute GPU examples on ReadTheDocs.
