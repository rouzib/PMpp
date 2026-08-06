# Testing

Run the smallest test that exercises a change, then expand toward the full
scientific path. Use the same JAX backend/device count as the code path under
test and run heavy multi-GPU jobs one process at a time.

## Formatting check

Before running solver tests, verify that the active package and test suite match
the YAPF configuration in `pyproject.toml`:

```bash
python -m yapf --diff --recursive src tests
```

Apply any required formatting with
`python -m yapf --in-place --recursive src tests`, then rerun the check.

## Core focused checks

```bash
python -m pytest tests/test_nested_white_noise.py -q
python -m pytest tests/test_mesh_halo_scatter_gather.py -q
python -m pytest tests/test_grad_gather.py tests/test_grad_gravity.py -q
python -m pytest tests/test_grad_nbody_mesh_halo.py -q
```

Use fresh processes when GPU compilation/memory state could affect a result.
Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when that is part of the tested
environment, and set `PYTHONPATH` to include the repository/test helpers when
running directly from a checkout.

## Test layers

1. **Shape and dtype:** global/logical shapes, sharding, masks, and dtypes.
2. **Invariants:** determinism, periodicity, mean density, mass, unique
   ownership, and zero capacity errors.
3. **Reference numerics:** small local/distributed parity and FFT round trips.
4. **Gradients:** finite leaves and finite-difference/directional-derivative
   checks for the modified operator.
5. **End to end:** initial modes through final density and full custom adjoint in
   `mesh_halo`.

A forward-only passing test is insufficient for a solver operator whose
transpose changed. Likewise, a gradient test on truncated overflow output is
invalid.

## Documentation checks

```bash
sphinx-build -W --keep-going -b html docs/source docs/build/html
sphinx-build -W --keep-going -b linkcheck docs/source docs/build/linkcheck
```

Notebook validation runs every notebook in a temporary copy with exactly two
selected GPUs and must not overwrite committed outputs. The regenerated
notebooks must contain no overflow or error output.
