# PM++: Multi-GPU Particle-Mesh Cosmology

[![Documentation Status](https://readthedocs.org/projects/pmpp-v2/badge/?version=latest)](https://pmpp-v2.readthedocs.io/en/latest/?badge=latest)

PM++ is a JAX-based, differentiable particle-mesh cosmology code built on top
of PMWD ideas and extended for multi-GPU simulations. The active implementation
is imported as `pmpp` and lives in `src/pmpp/`; the `pmwd/` directory is kept as a reference implementation for
validation.

## Current Scope

- Multi-GPU PM N-body simulation with JAX.
- Preferred `mesh_halo` multi-GPU mode.
- PMWD comparison tests for forward and gradient correctness.
- Distributed FFT support for sharded meshes.
- LPT, Boltzmann/growth utilities, scatter/gather, and power-spectrum tools.
- Potential-correction models under `src/pmpp/corrections/`.

## Repository Layout

```text
PM++_v2/
|-- src/pmpp/                    # Active importable PM++ package
|   |-- configuration.py         # Simulation configuration
|   |-- multigpu_configuration.py# Multi-GPU mode/configuration object
|   |-- particles.py             # Particle state and ownership
|   |-- scatter.py               # Particle-to-mesh assignment
|   |-- gather.py                # Mesh-to-particle interpolation
|   |-- gravity.py               # PM force solve
|   |-- steps.py                 # Drift, kick, force, adjoint pieces
|   |-- nbody.py                 # Full N-body integration and VJP
|   |-- FFT_distributed.py       # Distributed FFT construction
|   |-- mesh_halo.py             # Mesh halo exchange helpers
|   |-- modes.py                 # White noise and linear modes
|   |-- lpt.py                   # LPT initialization
|   |-- power_spectrum.py        # Density and particle P(k)
|   |-- corrections/             # Potential corrections
|   `-- potential_correction.py  # Backward-compatible correction facade
|-- pmwd/                        # Reference PMWD implementation
|-- tests/                       # Regression and gradient tests
|-- scripts/                     # Benchmarks and diagnostics
|-- notebooks/                   # Examples and exploratory notebooks
`-- docs/                        # Project documentation
```

## Minimal Multi-GPU Setup

New code should use the nested `MultiGPUConfiguration` object. The older
top-level `compute_mesh=` compatibility path still exists, but is not preferred.

```python
import jax
import jax.numpy as jnp

from pmpp.configuration import Configuration
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh

res = 256
box_size = 1000.0  # Mpc/h
ptcl_grid_shape = (res, res, res)
ptcl_spacing = box_size / res

gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This multi-GPU example requires at least 2 GPUs.")
compute_mesh = create_compute_mesh(gpu_devices)
num_devices = len(gpu_devices)

conf = Configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=compute_mesh,
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=int((res**3 / num_devices) * 1.8),
    max_share_ptcl=50_000,
    max_halo_share_ptcl=50_000,
    max_share_gather_ptcl=200_000,
    float_dtype=jnp.float32,
)
```

Capacity overflows are correctness failures. If a run reports overflow in
particle migration, halo rebuild, or gather exchange buffers, increase the
corresponding capacity and rerun.

## Minimal Forward Run

```python
import jax
import jax.numpy as jnp

from pmpp.boltzmann import boltzmann
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.lpt import lpt
from pmpp.modes import linear_modes, white_noise
from pmpp.nbody import nbody
from pmpp.scatter import scatter

res = 32
box_size = 100.0
conf = Configuration(
    box_size / res,
    (res, res, res),
    mesh_shape=1,
    float_dtype=jnp.float32,
)

cosmo = boltzmann(SimpleLCDM(conf), conf)
modes = white_noise(0, conf)
modes = linear_modes(modes, cosmo, conf)
ptcl = lpt(modes, cosmo, conf)

nbody_jit = jax.jit(nbody, static_argnames=("conf", "reverse"))
ptcl_final = nbody_jit(ptcl, cosmo, conf)
density = scatter(ptcl_final, conf)

print(density.shape)
print(float(density.mean()))
```

Expected sanity checks:

- density shape matches the mesh;
- density mean is close to `1.0`;
- no capacity warnings appear.

## Potential Corrections

Correction implementations now live in `pmpp.corrections` (`src/pmpp/corrections/`).

```python
from pmpp.corrections import (
    apply_potential_correction,
    evaluate_potential_transfer,
    init_potential_correction,
)
```

`pmpp.potential_correction` remains as a compatibility facade for old scripts,
but new code and tests should import from `pmpp.corrections`.

Supported correction families:

- `radial`, `radial_spline`, `neural_spline`
- `mesh_cnn`, `cnn`
- `combined`, `hybrid`, `spline_cnn`
- `pm_window`, `cic_compensation`, `cic_window_compensation`

## Multi-GPU Modes

Prefer `mesh_halo` for current serious multi-GPU work:

- particles are stored authoritatively on their owning slab;
- particles migrate between slabs when needed;
- mesh halos are exchanged for local stencil operations;
- it is generally faster than the older particle-halo path in current
  `256^3`, 2-GPU testing.

`particle_halo` remains useful for comparison and legacy validation.

## Testing

Focused correction and gravity checks:

```bash
/home/rouzib/.virtualenvs/PMPP/bin/python -m pytest \
  tests/test_potential_correction.py \
  tests/test_grad_gravity.py \
  tests/test_gravity_particle_nyquist_filter.py \
  -q
```

Mesh-halo scatter/gather:

```bash
/home/rouzib/.virtualenvs/PMPP/bin/python -m pytest tests/test_mesh_halo_scatter_gather.py -q
```

End-to-end gradient:

```bash
/home/rouzib/.virtualenvs/PMPP/bin/python -m pytest tests/test_grad_nbody.py -q
```

## Notebooks

The primary example notebooks are:

- `notebooks/pmpp_showcase.ipynb`
- `notebooks/mGPU_pmwd_local.ipynb`

Restart notebook kernels after code changes. Stale kernels can keep old module
objects, especially around `pmpp.corrections` and multi-GPU configuration.