# Contributing

The active installable package is `pmpp` under `src/pmpp`. Keep changes focused,
preserve differentiability and distributed invariants, and provide evidence at
the level affected by the change.

## Workflow

1. Create a focused branch and record the starting revision.
2. Install an editable environment with `python -m pip install -e ".[dev,docs]"`.
3. Inspect the current implementation and its tests before editing; historical
   optimization notes are context, not source of truth.
4. Add or update the smallest test that expresses the intended behavior.
5. Run focused tests, then the relevant end-to-end forward and gradient checks.
6. Build the documentation with warnings as errors when code, docstrings,
   examples, or public behavior changes.
7. Report correctness, runtime settings, and any memory/performance trade-off in
   the review description.

Do not treat a microbenchmark improvement as a solver improvement. Changes to
scatter/gather, force, movement, FFT, or the adjoint need end-to-end validation
in the intended runtime mode.

## Scientific changes

State the equation/discretization being changed, units and array layout, zero
mode/Nyquist treatment, and expected invariants. Never hide a new model choice
behind a neutral-sounding default.

## Distributed changes

Preserve authoritative ownership, periodic neighbor routing, stable padded
slots, transposed spectral layout, and the transpose of every communication
operator. Capacity errors must remain visible and must never be recast as an
acceptable approximation.

## Public API and documentation

Use explicit `pmpp.<module>` imports in examples. Add public objects to the
curated API by user task and write NumPy-style docstrings. Link concepts to API
and notebooks; avoid copying the same explanation into multiple pages. PMWD is
an upstream reference implementation, not part of PM++'s public API
or documentation tree.
