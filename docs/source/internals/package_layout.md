# Python package layout

The baseline PM++ solver groups implementation code by physical or runtime
responsibility. User code imports from the feature packages, while each
implementation has one canonical module path.

```text
src/pmpp/
|-- core/                    # Configuration and shared JAX utilities
|-- cosmology/               # Cosmology models, transfer, and growth
|-- initial_conditions/      # White noise, linear modes, and LPT
|-- numerics/                # Local FFT and ODE primitives
|-- distributed/             # Topology, distributed FFT, halos, and routing
|-- cic/                     # CIC geometry, scatter, gather, and Pallas kernels
|-- nbody/                   # Particles, gravity, integration, and observers
`-- analysis/                # Power spectra and diagnostic plotting
```

## Preferred imports

The feature packages expose the task-level API used by simulation scripts:

```python
from pmpp import Configuration, MultiGPUConfiguration
from pmpp.analysis import density_to_pk
from pmpp.cic import scatter
from pmpp.cosmology import SimpleLCDM, boltzmann
from pmpp.distributed import create_compute_mesh
from pmpp.initial_conditions import linear_modes, lpt, white_noise
from pmpp.nbody import nbody
```

## Responsibilities

`core` owns the immutable simulation configuration and low-level utilities
shared by several domains. `numerics` contains mathematical implementation
helpers that do not define cosmological policy. `cosmology` and
`initial_conditions` construct the physical inputs to the particle solver.

`distributed` owns device topology, distributed FFT construction, mesh-halo
exchange, particle routing, and the optional CUDA backend. `cic` owns the
particle-to-mesh and mesh-to-particle operators.

`nbody` owns the particle state, PM gravity, drift/kick integration, the custom
adjoint, and forward observers. `analysis` contains general observables.

`Configuration` assembles the runtime. A user-supplied
`MultiGPUConfiguration` first acts as a seed containing the compute mesh,
runtime mode, and optional CUDA request. `pmpp.distributed.configuration`
derives slab geometry, capacities, ring permutations, FFTs, CIC callables, and
particle-routing callables. The resulting initialized runtime is stored in
`conf.multigpu`. Selected legacy attributes on `Configuration` forward to that
object, but new code should treat `conf.multigpu` as the runtime owner.

```{mermaid}
flowchart LR
  A["core + numerics"] --> B["cosmology"]
  B --> C["initial conditions"]
  A --> D["distributed runtime"]
  D <--> E["CIC operators"]
  C --> F["N-body solver"]
  D --> F
  E --> F
  F --> G["analysis"]
  E --> G
```

Two composition seams are intentionally visible. Distributed configuration
initializes CIC callables, while CIC uses distributed halo and routing
primitives. LPT also reuses the same Poisson and routing-transpose operations
as N-body. Keeping those shared operations identical is more important than
forcing a misleading one-way dependency diagram.

## Package API boundaries

Feature-package initializers are the stable import boundary. Tests and user
code import task-level objects from `pmpp.core`, `pmpp.cosmology`,
`pmpp.initial_conditions`, `pmpp.distributed`, `pmpp.cic`, `pmpp.nbody`,
or `pmpp.analysis`.

Implementation files such as `pmpp.nbody.integrator` and
`pmpp.distributed.routing` are internal organization details. Package
initializers use ordinary explicit imports as their public interface. This lets
Python IDEs and static-analysis tools resolve every exported name, while
implementation files can still move without changing callers. The former flat
module paths are intentionally absent.
