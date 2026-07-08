# PMWD validation

The `pmwd/` directory is retained as a reference implementation for validation against the PMWD behavior that motivated PM++. Validation workflows should compare both forward quantities and gradients when the relevant PM++ adjoint is under test.

## What to compare

- Initial modes and LPT particle fields.
- Scatter/gather outputs.
- Gravity and force fields.
- N-body final particle states for tiny runs.
- Gradients for targeted stages.

## Commands

Run focused tests before scaling to the full suite:

```bash
python -m pytest tests/test_grad_scatter.py -q
python -m pytest tests/test_grad_gather.py -q
python -m pytest tests/test_grad_gravity.py -q
python -m pytest tests/test_grad_nbody.py -q
```

Tolerances should be stated in the test or validation script. Do not infer physical accuracy from passing a single implementation-comparison test.
