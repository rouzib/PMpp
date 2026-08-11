# PM++: Multi-GPU Particle-Mesh Cosmology

<p align="center">
  <img src="https://raw.githubusercontent.com/rouzib/PMpp/master/docs/source/_static/pmpp-logo.svg" alt="PM++ logo" width="360">
</p>

[![Documentation Status](https://readthedocs.org/projects/pmpp-docs/badge/?version=latest)](https://pmpp-docs.readthedocs.io/en/latest/?badge=latest)
[![Package build](https://github.com/rouzib/PMpp/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/rouzib/PMpp/actions/workflows/publish-to-pypi.yml)
[![PyPI version](https://img.shields.io/pypi/v/pmpp.svg)](https://pypi.org/project/pmpp/)
[![Python versions](https://img.shields.io/pypi/pyversions/pmpp.svg)](https://pypi.org/project/pmpp/)
[![License](https://img.shields.io/pypi/l/pmpp.svg)](https://github.com/rouzib/PMpp/blob/master/LICENSE)

## What is PM++?

PM++ is a JAX-based, differentiable particle-mesh simulator for large-scale
structure cosmology. It distributes a single simulation across multiple GPUs
and covers the path from Gaussian initial modes through Lagrangian perturbation
theory and N-body evolution to density fields and scientific summary
statistics.

PM++ is designed for simulations that need both scale and derivatives. It can
differentiate observables with respect to initial modes, particle states, and
cosmological parameters while keeping the distributed forward model and its
adjoint in one JAX program.

The implementation builds on ideas from
[PMWD](https://github.com/eelregit/pmwd). The maintained validation suite
compares PM++ directly with PMWD for forward evolution and gradients. For the
configurations covered by those tests, the two agree down to machine
precision.

## Capabilities

- **End-to-end cosmological evolution:** transfer and growth calculations,
  Gaussian and nested initial fields, linear modes, LPT, PM N-body evolution,
  density assignment, observers, and power-spectrum analysis.
- **Automatic differentiation:** forward and reverse derivatives through the
  simulation, including a custom N-body adjoint for memory-efficient reverse
  sweeps.
- **Distributed execution:** sharded particle ownership, mesh-halo exchange,
  distributed FFTs, and particle migration across a multi-GPU device mesh.
- **Accelerated particle-mesh operations:** paired Pallas CIC kernels and an
  optional compiled CUDA routing backend, with portable JAX fallbacks.
- **Scientific configuration:** float32 and float64 execution, configurable
  particle and force meshes, integration schedules, correction models, and
  differentiable cosmological parameters.
- **Validation and analysis:** PMWD forward and gradient comparisons, mass and
  ownership checks, finite-difference tests, power spectra, projections, and
  CAMELS and QUIJOTE data adapters.

## What PM++ enables

By combining multi-GPU execution with differentiability, PM++ makes it
possible to simulate larger cosmological volumes or use finer mass resolution
without giving up parameter sensitivities. This supports field-level inference,
initial-condition reconstruction, gradient-based calibration, and sensitivity
studies of how cosmological parameters shape large-scale structure.

## Installation

PM++ requires Python 3.10 or newer and supports
`jax>=0.9.1,<0.11`. Install the JAX build for the accelerator and driver before
installing PM++. For a CUDA 12 environment:

```bash
python -m venv ~/.venvs/pmpp
source ~/.venvs/pmpp/bin/activate
python -m pip install --upgrade pip
python -m pip install "jax[cuda12]>=0.9.1,<0.11"
python -m pip install pmpp
```

Choose the PM++ extra that matches the environment:

| Use | Installation |
|---|---|
| Run simulations | `python -m pip install pmpp` |
| Run the repository tests | `python -m pip install "pmpp[dev]"` |
| Build the documentation | `python -m pip install "pmpp[docs]"` |
| Develop, test, and build documentation | `python -m pip install "pmpp[dev,docs]"` |

When a compatible CUDA development toolkit and CMake are available, build the
optional accelerated routing extension in the same environment:

```bash
pmpp-build-cuda-routing
```

The compiled router is optional. PM++ uses its portable JAX implementation
when the extension or a compatible `nvcc` is unavailable.

See the
[installation guide](https://pmpp-docs.readthedocs.io/en/latest/getting_started/installation.html)
for CUDA 13, HPC cluster, Compute Canada, and offline-wheel instructions.

## Documentation

The complete documentation is available at
[pmpp-docs.readthedocs.io](https://pmpp-docs.readthedocs.io/en/latest/). It
contains the getting-started workflow, scientific configuration guide,
multi-GPU setup, differentiation guidance, solver internals, and API reference.

Useful entry points:

- [Getting started](https://pmpp-docs.readthedocs.io/en/latest/getting_started/index.html)
- [User guide](https://pmpp-docs.readthedocs.io/en/latest/user_guide/index.html)
- [How PM++ works](https://pmpp-docs.readthedocs.io/en/latest/internals/index.html)
- [API reference](https://pmpp-docs.readthedocs.io/en/latest/api/index.html)

## Repository layout

```text
PMpp/
|-- src/pmpp/
|   |-- core/                  # Configuration and shared utilities
|   |-- cosmology/             # Cosmological models, transfer, and growth
|   |-- initial_conditions/    # White noise, linear modes, and LPT
|   |-- nbody/                 # Particles, gravity, integration, and observers
|   |-- cic/                   # Scatter, gather, and Pallas CIC kernels
|   |-- distributed/           # Device meshes, FFTs, halos, and routing
|   |-- numerics/              # Local FFT and ODE primitives
|   |-- corrections/           # Optional force and phase-space corrections
|   |-- analysis/              # Power spectra and plotting
|   `-- extras/                # CAMELS and QUIJOTE adapters
|-- cuda/                      # Optional native CUDA routing sources
|-- tests/
|   `-- pmwd/                  # Test-only PMWD reference implementation
|-- docs/source/
|   |-- getting_started/       # Installation and first-run guidance
|   |-- user_guide/            # Scientific and runtime configuration
|   |-- internals/             # Algorithms and distributed design
|   |-- api/                   # Public API reference
|   `-- notebooks/             # Pre-rendered scientific workflows
|-- pyproject.toml             # Package metadata, dependencies, and tooling
`-- requirements.txt           # Read the Docs environment requirements
```

The importable implementation lives entirely under `src/pmpp`. The copy of
PMWD under `tests/pmwd` is retained only as a numerical reference for
validation.

## Citation

See the [citation guide](https://pmpp-docs.readthedocs.io/en/latest/citation.html)
for PM++, its discrete-adjoint foundation, and PMWD attribution.

## License

PM++ is distributed under the BSD 3-Clause license. See [LICENSE](LICENSE).
PM++ retains the original PMWD notice in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
