# Release checklist

Do not publish a package until TestPyPI and PyPI Trusted Publishers have been configured manually in the package indexes and the matching GitHub environments exist.

## Prepare a local release environment

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

## Validate imports and tests

```bash
python -c "import pmpp; print(pmpp.Configuration, pmpp.MultiGPUConfiguration)"
python -m pytest tests/test_utils.py tests/test_particles_ordered_init.py -q
```

GPU-dependent tests should be run separately on hardware that provides the required devices.

## Build and check distributions

```bash
rm -rf dist build .egg-info
python -m build
python -m twine check dist/*
```

## Configure Trusted Publishing

Before publishing, configure Trusted Publishing manually:

1. Create or claim the `pmpp` project on TestPyPI and PyPI.
2. Add GitHub Trusted Publishers for `rouzib/PMpp` using the workflow `.github/workflows/publish-to-pypi.yml`.
3. Configure the GitHub environments `testpypi` and `pypi`; the `pypi` environment should require manual approval.
4. Verify that no API tokens or PyPI credentials are committed to the repository.

## Publish a release

Pushes to `master` publish to TestPyPI after the Trusted Publisher is configured. Real PyPI publication is restricted to version tags:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Do not create the tag until the release commit has been reviewed and the PyPI Trusted Publisher is ready.
