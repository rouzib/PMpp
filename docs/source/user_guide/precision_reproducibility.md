# Precision, reproducibility, and gradients

PM++ uses floating-point particle, mesh, FFT, and adjoint operations on highly
parallel accelerators. Three related properties must be considered separately:

- **precision**: how closely floating-point arithmetic approximates real
  arithmetic.
- **repeatability**: whether identical executions return identical bits.
- **gradient accuracy**: whether the implemented adjoint agrees with the
  derivative of the implemented forward calculation.

No finite floating-point calculation is exact. For a simulation that is sensitive
to small numerical differences, use `float64` particle and mesh fields and
validate the intended loss with finite differences. For bitwise-repeatable GPU
execution, also enable deterministic XLA operations. Both choices can cost
substantial runtime and memory.

## Recommended policies

| Requirement | Recommended policy | Main cost |
|---|---|---|
| Maximum throughput | `float_dtype=float32` | Small run-to-run differences may occur in atomic reductions. |
| Repeatable float32 debugging | float32 plus deterministic XLA operations | Deterministic scatter/reduction kernels can be much slower. |
| Higher numerical precision | `float_dtype=float64` and `cosmo_dtype=float64` | Approximately twice the storage per floating field, more FFT/scratch pressure, and lower throughput. |
| Higher precision and repeatability | float64 plus deterministic XLA operations | Highest runtime and memory cost, and finite-difference validation is still required. |

If a result will be used as a precision-sensitive simulation reference, prefer
float64. If a regression test must reproduce the same bits, use the
deterministic flag. If both properties matter, use both rather than treating
one as a substitute for the other.

## Why float32 GPU runs can vary

CIC painting and several movement pullbacks accumulate many values into shared
destinations. GPU implementations normally use floating-point atomics for
these scatter-add operations. Thread scheduling can change the addition order,
and floating-point addition is not associative. The same executable can
therefore produce slightly different float32 sums on repeated executions.

Important accumulation sites include:

- CIC mesh painting in `src/pmpp/scatter.py`.
- Particle-route cotangent accumulation in `src/pmpp/halo_moving.py`.
- Force and gravity VJPs reached through `src/pmpp/steps.py`.

This is not a GPU hardware fault. It is a performance trade-off of unordered
parallel accumulation. OpenXLA's
[GPU determinism guide](https://openxla.org/xla/determinism) explains the
atomic-scatter source and the deterministic alternatives.

During a float32 adjoint, errors can arise from machine precision and the
intrinsic nondeterministic behavior of parallel GPU operations. For an
individual cell, the relative difference is usually on the order of 1%, though
the exact value depends on the simulation and hardware.

## Selecting float64

Enable JAX x64 before constructing a configuration, then explicitly select both
floating dtypes. You can enable it with the environment or through JAX itself:

```bash
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from pmpp.configuration import Configuration

conf = Configuration(
    ptcl_spacing=1.0,
    ptcl_grid_shape=(128, 128, 128),
    mesh_shape=1,
    float_dtype=jnp.float64,
    cosmo_dtype=jnp.float64,
    pallas_cic=False,
    # Multi-GPU configuration and capacities omitted here.
)
```

Construct `Configuration` only after the x64 setting is final. Record
`jax.config.jax_enable_x64`, `conf.float_dtype`, and `conf.cosmo_dtype` with
every simulation artifact.

Float64 uses more memory than float32. Plan capacity from measurements on the
target hardware rather than assuming that a float32 simulation will also fit
in float64.

## Selecting deterministic GPU operations

Set the flag before Python imports JAX or initializes the GPU backend:

```bash
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

This asks XLA to avoid supported nondeterministic GPU implementations and to
use deterministic lowerings for operations such as scatter. Compilation may
fail if an operation has no deterministic implementation.

Deterministic GPU operations can make a simulation about two to six times
slower. Measure the impact on the intended accelerator and workload.
