# Benchmark methodology

Performance claims must say exactly what callable is timed. PM++ uses two
different benchmark classes and they answer different questions.

## Full-pipeline benchmark

Use one compiled full-forward callable for each implementation when comparing
scientific pipelines. The timed region should include the same stages—normally
initialization inputs already prepared as stated, then LPT/N-body/final density
according to the benchmark's declared boundary—and should not stage one
implementation into per-step calls while fusing the other.

For a cross-implementation comparison, build one fused full-forward callable
per framework. A staged harness that separately times `nbody` and `scatter` is
diagnostic, not a clean full-forward comparison.

## Focused benchmark

Use a focused benchmark to answer a local engineering question such as FFT,
gather backward, one force evaluation, or one adjoint stage. Label it with the
exact function boundary and do not extrapolate its ratio to the full gradient.
The full gradient contains a forward run plus the custom adjoint/recomputation.

## Timing protocol

1. Start a fresh process and run only one heavy multi-GPU job.
2. Construct inputs/configuration before the timed repetitions unless setup is
   intentionally in scope.
3. Warm up the exact static shapes and callable once.
4. Call `jax.block_until_ready` before starting and after ending each timed
   iteration.
5. Report compilation separately from steady-state execution.
6. Use multiple repetitions and report each sample plus a robust summary.
7. Measure peak memory with a method whose baseline/reset semantics are stated.
8. Keep correctness checks outside timing but run them on the timed result.

## Required report fields

- PM++ commit, script/command, date, JAX/jaxlib, CUDA/driver, and device model;
- particle/mesh shapes, box, dtype, LPT order, schedule, and correction model;
- device count/order, `multigpu_mode`, and `mesh_shape`;
- `max_ptcl_per_slice`, its factor over uniform occupancy, `max_share_ptcl`,
  `max_halo_share_ptcl`, and `max_share_gather_ptcl`;
- warm-up/repetition/synchronization protocol and whether compilation/setup is
  included;
- correctness tolerances, mass/density checks, and confirmation of zero
  capacity errors;
- per-device and total peak memory when comparing one versus multiple devices.

Never omit capacities from a multi-GPU result: they affect both correctness and
memory. Never describe a staged or operator-only time as full-forward speed.
