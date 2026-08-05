# How PM++ Works

This section describes the algorithm implemented in `src/pmpp`, from the
cosmological parameters and random modes to a differentiable, distributed
particle-mesh simulation. It develops the equations together with the array
layouts and discrete operators that evaluate them.

The chapters follow the forward simulation and then its reverse pass:

1. [System architecture](architecture.md) defines the state, units, JAX
   program structure, and end-to-end dataflow.
2. [Initial modes and LPT](initial_conditions.md) constructs the transfer and
   growth tables, Gaussian density modes, and first- or second-order particle
   initial conditions.
3. [Particle-mesh force](particle_mesh.md) derives CIC assignment, the
   periodic Poisson solve, spectral forces, the Pallas kernels, and their
   transposes.
4. [Distributed runtime](distributed_runtime.md) explains slab ownership,
   mesh halos, particle migration, distributed real FFTs, and static-capacity
   invariants.
5. [Integration and discrete adjoint](integration_and_adjoint.md) derives the
   growth-matched drift and kick factors and the reverse-time custom VJP.
6. [Optional CUDA routing](cuda_routing.md) documents the typed FFI boundary,
   route records, stable merge, and route transpose.

The underlying PM and discrete-adjoint formulation follows
[Li et al., *Differentiable Cosmological Simulation with the Adjoint Method*,
arXiv:2211.09815v2](https://arxiv.org/abs/2211.09815v2). PM++ keeps those
mathematical operators while adding distributed particle ownership, mesh-halo
communication, distributed FFT layouts, and optional local CUDA routing.

## Conventions

The simulation is periodic. Bold lower-case symbols denote vectors, hats
denote Fourier coefficients, and a prime denotes differentiation with respect
to $\ln a$. The discrete Fourier transforms use JAX's default normalization:
the forward transform is unnormalized and the inverse transform carries
$1/N_\mathrm{mesh}$.

PM++ separates three kinds of quantities:

- **physical state**, such as cosmological parameters, particle displacement,
  velocity, and acceleration
- **discrete representation**, such as mesh indices, Fourier layouts, masks,
  and padded particle slots
- **static program structure**, such as shapes, the device mesh, capacities,
  and selected kernels.

Keeping these distinctions explicit is important. A mathematical field may be
unchanged while its sharding changes, and a particle may change owner while
its physical trajectory remains continuous.

```{toctree}
:maxdepth: 1
:hidden:

architecture
initial_conditions
particle_mesh
distributed_runtime
integration_and_adjoint
cuda_routing
```
