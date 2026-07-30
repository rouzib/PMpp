# PM++: Multi-GPU Particle-Mesh Cosmology

<p align="center">
  <img src="https://raw.githubusercontent.com/rouzib/PMpp/master/docs/source/_static/pmpp-logo.svg" alt="PM++ logo" width="360">
</p>

[![Documentation Status](https://readthedocs.org/projects/pmpp-docs/badge/?version=latest)](https://pmpp-docs.readthedocs.io/en/latest/?badge=latest)
[![Package build](https://github.com/rouzib/PMpp/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/rouzib/PMpp/actions/workflows/publish-to-pypi.yml)
[![PyPI version](https://img.shields.io/pypi/v/pmpp.svg)](https://pypi.org/project/pmpp/)
[![Python versions](https://img.shields.io/pypi/pyversions/pmpp.svg)](https://pypi.org/project/pmpp/)
[![License](https://img.shields.io/pypi/l/pmpp.svg)](https://github.com/rouzib/PMpp/blob/master/LICENSE)

PM++ is a JAX-based, differentiable particle-mesh cosmology code built on PMWD
ideas and extended for multi-GPU simulations. The active implementation is
imported as `pmpp` and lives in `src/pmpp/`; `tests/pmwd/` retains the PMWD
reference implementation used exclusively for validation.

The documented baseline uses exactly two GPUs so every example exercises the
distributed ownership, mesh-halo, and FFT paths.

## Installation

On a computer with two visible GPUs:

```bash
python -m venv ~/.venvs/pmpp
source ~/.venvs/pmpp/bin/activate
python -m pip install --upgrade pip
python -m pip install pmpp jupyter

# The checkout supplies the notebooks; PM++ itself remains pip-installed.
git clone https://github.com/rouzib/PMpp.git ~/PMpp
cd ~/PMpp
jupyter lab docs/source/notebooks
```

PM++ requests a CUDA 12-capable JAX build without pinning a specific JAX
release on this target. 

## Current Scope

- Multi-GPU PM N-body simulation with JAX.
- Preferred `mesh_halo` multi-GPU mode.
- PMWD comparison tests for forward and gradient correctness.
- Distributed FFT support for sharded meshes.
- LPT, Boltzmann/growth utilities, scatter/gather, and power-spectrum tools.
- Potential-correction models under `src/pmpp/corrections/`.

## Repository Layout

```text
PMpp/
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
|   `-- potential_correction.py  # Backward-compatible correction facade
|-- tests/                       # Regression and gradient tests
|   `-- pmwd/                    # Test-only PMWD reference implementation
|-- docs/source/notebooks/       # Pre-executed documentation notebooks
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
selected_devices = gpu_devices[:2]
compute_mesh = create_compute_mesh(selected_devices)
num_devices = len(selected_devices)

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

## Minimal Two-GPU Forward Run

```python
import jax
import jax.numpy as jnp

from pmpp.boltzmann import boltzmann
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.lpt import lpt
from pmpp.modes import linear_modes, white_noise
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.nbody import nbody
from pmpp.scatter import scatter
from pmpp.utils import create_compute_mesh

res = 32
box_size = 100.0
gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This PM++ simulation requires at least two GPUs")
selected_devices = gpu_devices[:2]

conf = Configuration(
    box_size / res,
    (res, res, res),
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(selected_devices),
        mode="mesh_halo",
    ),
    float_dtype=jnp.float32,
)

@jax.jit
def simulate(seed):
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    noise = white_noise(seed, conf)
    modes = linear_modes(noise, cosmo, conf)
    particles = lpt(modes, cosmo, conf)
    particles = nbody(particles, cosmo, conf)
    return particles, scatter(particles, conf)

ptcl_final, density = simulate(0)
density.block_until_ready()

print(density.shape)
print(float(density.mean()))
```

Expected sanity checks:

- density shape matches the mesh;
- density mean is close to `1.0`;
- no capacity warnings appear.

## Multi-GPU Modes

Prefer `mesh_halo` for current multi-GPU work:

- particles are stored authoritatively on their owning slab;
- particles migrate between slabs when needed;
- mesh halos are exchanged for local stencil operations;
- it is generally faster than the older particle-halo path in current
  `256^3`, 2-GPU testing.

`particle_halo` remains useful for comparison and legacy validation.

## Testing

Focused gravity checks:

```bash
/home/rouzib/.virtualenvs/PMPP/bin/python -m pytest \
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

The documentation gallery contains six pre-executed notebooks:

- first simulation and configuration;
- resolution-consistent initial conditions evolved from $32^3$ through $256^3$;
- a two-GPU `mesh_halo` run;
- observers and analysis;
- differentiation with finite-difference checks.

Read the Docs renders committed outputs and does not execute the notebooks.
Restart kernels after code changes. Re-run every notebook with exactly two
selected GPUs in a clean temporary copy before committing its outputs.

## License

PM++ is distributed under the BSD-3-Clause license; see [LICENSE](LICENSE).
PM++ is based on PMWD and retains the original PMWD BSD 3-Clause notice in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The test-only `tests/pmwd/`
package is kept as a reference implementation for validation.

## Documentation build

Install the documentation extra and build the Sphinx site locally:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs/source docs/build/html
```
