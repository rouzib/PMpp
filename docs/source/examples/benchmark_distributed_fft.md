# Benchmark distributed FFT

Distributed FFT benchmarks must separate correctness from timing.

## Correctness first

Compare a distributed transform against a known local or reference layout for a small mesh. Record dtype, spectral layout, sharding, and tolerance.

## Timing second

```python
import time
import jax

# warmup_result = fft_fn(x)
# jax.block_until_ready(warmup_result)
# t0 = time.perf_counter()
# result = fft_fn(x)
# jax.block_until_ready(result)
# print(time.perf_counter() - t0)
```

Run a warmup before timing, synchronize outputs, and report device model, JAX version, mesh shape, and whether the number includes compilation.
