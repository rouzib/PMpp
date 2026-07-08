# Profile JAX compilation

The first call to a JIT-compiled PM++ function includes compilation time. Separate compilation from execution when measuring performance.

## Checklist

1. Run one warmup call with the target static shapes.
2. Block on outputs with `jax.block_until_ready` before recording elapsed time.
3. Keep capacities, mesh shape, dtype, correction settings, and device count fixed during timing.
4. Report whether timing includes FFT plan/setup, compilation, or only steady-state execution.

Do not compare benchmark numbers across machines unless the CUDA/JAX stack, device model, precision, and synchronization method are documented.
