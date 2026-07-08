# Contributing

Use `pmpp` imports for the active implementation. Keep `pmwd/` available as a reference implementation and avoid moving it into the public PM++ API.

## Workflow

1. Create a focused branch.
2. Install with `python -m pip install -e ".[dev,docs]"` in an environment with the required build dependencies.
3. Run targeted tests for the module you changed.
4. Build documentation when editing docs or public docstrings.
5. Include validation notes for numerical or gradient changes.

Prefer small, reviewable changes. Document changes to public behavior in tutorials, examples, or API docstrings as appropriate.
