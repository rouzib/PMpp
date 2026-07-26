# Release checklist

## Code and science

- Freeze the intended version in `pyproject.toml` and summarize public changes.
- Run focused tests plus all available CPU/GPU suites on the supported stack.
- Run end-to-end forward and gradient validation in `mesh_halo` with no capacity
  errors.
- Record known experimental interfaces and unsupported behavior (including LPT
  order 3).
- Confirm license and third-party notices.

## Documentation and package

- Build HTML with `-W --keep-going` and run linkcheck.
- Validate all curated API imports and ensure each public object appears once.
- Validate and re-execute every notebook with exactly two selected GPUs, then
  check provenance metadata.
- Inspect desktop/mobile, light/dark, Mermaid, notebook output, search, and
  local/offline assets.
- Build distributions with `python -m build` and verify them with
  `python -m twine check dist/*` in a clean environment.
- Install the built wheel into a fresh environment and run an import/smoke test.

## Publishing

Confirm the TestPyPI/PyPI Trusted Publisher configuration and protected GitHub
environments before tagging. Publish a reviewed version tag only after the
package, source archive, rendered docs, and release notes agree. Never commit
API tokens or package-index credentials.
