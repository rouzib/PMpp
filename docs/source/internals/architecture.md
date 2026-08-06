# System architecture

PM++ is a JAX program for a periodic, differentiable particle-mesh simulation
{cite:p}`hockney1988particles,li2024adjoint,bradbury2018jax`. Its structure is
easiest to understand as a sequence of mathematical maps, with distributed
layouts attached to the maps that require communication.

```{mermaid}
flowchart LR
  A["Configuration and cosmology"] --> B["Transfer and growth tables"]
  C["White Fourier modes"] --> D["Linear density modes"]
  B --> D
  D --> E["LPT particle state"]
  E --> F["N-body drift, force, kick"]
  F --> G["Particles or mesh observables"]
  G --> H["Objective"]
  H -. "custom VJP" .-> F
  F -. "reverse reconstruction" .-> E
```

The same public pipeline works with one or multiple devices. Multiple devices
change ownership and array sharding, but not the equations being solved.

## Configuration is static program structure

`Configuration` contains the geometry, units, time schedule, numeric dtypes,
and kernel choices. It is a frozen JAX pytree whose fields are static auxiliary
data. This makes array shapes and control flow known while JAX traces a solver.
The cosmological parameters and particle arrays remain dynamic pytree leaves.
This division follows JAX's treatment of [JIT-static data][jax-jit] and
structured [pytree leaves][jax-pytrees].

For particle spacing $\ell_p$, particle-grid shape
$\mathbf N_p=(N_{p,x},N_{p,y},N_{p,z})$, and mesh shape
$\mathbf N_m$, PM++ defines

$$
\mathbf L_\mathrm{box}=\ell_p\mathbf N_p,
\qquad
V_\mathrm{box}=\prod_i L_{\mathrm{box},i},
\qquad
\ell_m=\frac{L_{\mathrm{box},x}}{N_{m,x}}.
$$

The current solver requires a cubic physical cell size, so `cell_size` is the
single value $\ell_m$. `disp_size` is its reciprocal. The particle and mesh
counts are

$$
N_p=\prod_i N_{p,i},
\qquad
N_m=\prod_i N_{m,i}.
$$

The configuration also builds the Fourier wavevectors, particle-Nyquist masks,
the time arrays, and the distributed helper functions. These derived objects
are created once because they depend only on static geometry.

## Internal units

The user supplies base mass, length, and time units $(M,L,T)$. PM++ converts
constants to those units:

$$
V_0=\frac{L}{T},
\qquad
H_0=H_{0,\mathrm{SI}}T,
\qquad
G=G_\mathrm{SI}\frac{M}{LV_0^2},
\qquad
\rho_\mathrm{crit}=\frac{3H_0^2}{8\pi G}.
$$

The default time unit is $H_0^{-1}$, which makes the numerical value of $H_0$
equal to one. The cosmological parameter `h` still enters physical scale
conversions and the transfer function.

The particle mass is derived rather than stored independently:

$$
m_p=\rho_\mathrm{crit}\,\Omega_m\,\ell_p^3.
$$

The main particle fields use these conventions:

| Field | Meaning | Units |
| --- | --- | --- |
| `pmid` | nearest mesh anchor | integer mesh cells |
| `disp` | offset from that anchor | $L$ |
| `vel` | canonical momentum per unit mass, $a^2\dot{\mathbf x}$ | $H_0L$ |
| `acc` | scaled force, $-\nabla(a\phi)$ | $H_0^2L$ |

These canonical particle variables and internal cosmological units follow the
differentiable PM formulation of {cite:t}`li2024adjoint`.

## Particle representation

PM++ does not normally store an absolute floating-point position. It stores a
nearby integer mesh point and a small physical displacement:

$$
\mathbf x_p=
\left(\ell_m\mathbf m_p+\mathbf d_p\right)
\bmod\mathbf L_\mathrm{box}.
$$

Here $\mathbf m_p$ is `pmid` and $\mathbf d_p$ is `disp`. `from_pos` chooses
the nearest mesh point with a round operation, leaving a centered residual.
This representation keeps the floating part local even when the absolute box
coordinate is large. The integer anchor is allowed to be signed. Periodicity
is applied when a position or raveled cell key is needed.

For a distributed simulation, each device owns a fixed-capacity leading array
of particle slots. `unused_index` marks padding. In `mesh_halo` mode,
`halo_mask` is false because particles have one authoritative owner. The
physical particle count is therefore not inferred from the allocated shape.

## Cosmology as differentiable data

`Cosmology` holds the parameters that may receive gradients. For a
Chevallier-Polarski-Linder (CPL) dark-energy model
{cite:p}`chevallier2001accelerating,linder2003expansion`,

$$
E^2(a)=\frac{H^2(a)}{H_0^2}
=\Omega_m a^{-3}+\Omega_k a^{-2}
+\Omega_\mathrm{de}
a^{-3(1+w_0+w_a)}e^{-3w_a(1-a)},
$$

with $\Omega_\mathrm{de}=1-\Omega_m-\Omega_k$. The time-dependent matter
fraction and logarithmic Hubble derivative are

$$
\Omega_m(a)=\frac{\Omega_m}{a^3E^2(a)},
\qquad
\frac{d\ln H}{d\ln a}=\frac{a}{2E^2(a)}\frac{dE^2}{da}.
$$

Parameters represented by dynamic leaves participate in differentiation.
Optional parameters left as `None` use fixed class values and do not acquire
cotangents. Transfer, growth, and variance tables are derived leaves. Helper
functions project N-body cotangents back to the independent cosmological
parameters instead of treating every derived table entry as an independent
parameter.

## JAX execution model

PM++ uses several JAX mechanisms for different responsibilities:

- [`jax.jit`][jax-jit] compiles the numerical pipeline and specializes it to
  the static configuration.
- [pytrees][jax-pytrees] keep cosmology and particle state structured while
  exposing their differentiable array leaves.
- [`shard_map`][jax-shard-map], named shardings, and
  [`lax.ppermute`][jax-ppermute] express local work and ring communication.
- [`custom_partitioning`][jax-custom-partitioning] tells JAX when an FFT stage
  is local in a particular layout.
- [`custom_vjp`][jax-custom-vjp] lets PM++ program reverse rules explicitly.
  The CIC and N-body rules follow the discrete adjoint formulation of
  {cite:t}`li2024adjoint`.
- [Pallas][jax-pallas] provides the JAX-traceable custom-kernel model used by
  the tiled CIC implementation. PM++ keeps its mathematical contract identical
  to the reference JAX implementation.
- [typed JAX FFI][jax-ffi] provides the boundary used to call shard-local CUDA
  particle pack and merge kernels.

The compiled program has [fixed array shapes][jax-dynamic-shapes].
Counts and masks determine which slots are active at runtime. This is why every
particle capacity is part of the correctness contract. An overflow is not an
approximation. It means the requested state cannot be represented by the
compiled program.

## Forward state transitions

The ordinary forward simulation is

$$
(\boldsymbol\theta,\boldsymbol\omega)
\xrightarrow{\text{transfer, growth}}
\widehat\delta_\mathrm{lin}
\xrightarrow{\mathrm{LPT}}
(\mathbf m,\mathbf d,\mathbf p)
\xrightarrow{\mathrm{N\mbox{-}body}}
(\mathbf m,\mathbf d,\mathbf p,\mathbf a).
$$

Each N-body macro-step alternates three maps:

1. **Drift** updates displacement from canonical velocity, then moves particles
   to the slab that owns the new position.
2. **Force** scatters particles to a mesh, solves Poisson's equation, and
   gathers acceleration back to particles.
3. **Kick** updates canonical velocity from acceleration.

This PM drift-force-kick structure and its growth-aware cosmological variant
are standard in particle-mesh evolution
{cite:p}`hockney1988particles,feng2016fastpm,li2024adjoint`.

The force is initialized before the first macro-step. It is refreshed after
each drift and reused by the adjacent kick.

## Reverse state transitions

JAX sees `nbody` as a custom differentiable primitive. Its forward rule saves
the final particle state and compact static information, not every time step.
The backward rule traverses the macro-steps in reverse. It algebraically
reconstructs the preceding particle state and applies the transpose of each
discrete forward operator.

This is a discretize-then-differentiate construction. The reverse pass is the
transpose of the implemented drift, route, force, and kick maps. It is not a
separate numerical integration of a continuous adjoint equation
{cite:p}`griewank2008derivatives,li2024adjoint`.

## Core invariants

The implementation relies on the following invariants:

- physical positions are periodic
- each active particle has exactly one authoritative owner in `mesh_halo`
- valid authoritative particles are packed in nondecreasing raveled-`pmid`
  order
- invalid slots sort after valid particles through a sentinel key
- scatter and gather use the same CIC stencil
- mesh-halo copy and mesh-halo reduction are transpose operations
- the real FFT transpose uses the correct Hermitian half-spectrum weights
- all configured drift coefficients sum to one and all kick coefficients sum
  to one
- particle and communication capacities never overflow.

These are mathematical and representation invariants. They hold independently
of the number or model of devices used to execute the program.

## Implementation map

| Responsibility | Main module |
| --- | --- |
| geometry, units, schedules, static kernel selection | `configuration.py` |
| differentiable cosmological parameters | `cosmo.py` |
| transfer, growth, and linear variance tables | `boltzmann.py`, `growth.py` |
| random and nested Fourier modes | `modes.py` |
| first- and second-order LPT | `lpt.py` |
| particle state and coordinate conversion | `particles.py` |
| CIC and Pallas kernels | `scatter.py`, `gather.py`, `pallas_cic.py` |
| Poisson force | `gravity.py` |
| drift, force, kick, and their adjoints | `steps.py` |
| N-body scan and custom VJP | `nbody.py` |
| slab layout and runtime binding | `multigpu_configuration.py` |
| particle ownership and route transpose | `halo_moving.py` |
| mesh halos | `mesh_halo.py` |
| distributed FFTs | `FFT_distributed.py` |
| optional CUDA FFI | `cuda_routing.py`, `cuda/route_kernels.cu` |

[jax-jit]: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
[jax-pytrees]: https://docs.jax.dev/en/latest/pytrees.html
[jax-shard-map]: https://docs.jax.dev/en/latest/notebooks/shard_map.html
[jax-ppermute]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html
[jax-custom-partitioning]: https://docs.jax.dev/en/latest/jax.experimental.custom_partitioning.html
[jax-custom-vjp]: https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
[jax-pallas]: https://docs.jax.dev/en/latest/pallas/
[jax-ffi]: https://docs.jax.dev/en/latest/ffi.html
[jax-dynamic-shapes]: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes
