# System architecture

## Purpose

PM++ turns a static experiment description plus differentiable inputs into a
particle trajectory, observable, and optionally gradients. The main design goal
is to preserve one scientific pipeline while changing its storage and
communication layout across devices.

## User workflow

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;
  A["Configuration + seed"]:::comm --> B["Linear modes"]:::state --> C["LPT particles"]:::state --> D["N-body evolution"]:::op --> E["Density / snapshots"]:::state --> F["Statistics or scalar loss"]:::check
```

**Text equivalent:** configuration and a seed determine linear modes; LPT turns
the modes into particles; N-body evolves them; scatter/observers produce fields;
analysis returns statistics or a scalar loss.

Function anchors are `Configuration`, `white_noise`/`linear_modes`, `lpt`,
`nbody`, `scatter`, the observer helpers, and `pmpp.power_spectrum`.

## State and equations

The cosmological forward model can be written as a composition,

$$
\omega,\theta \xrightarrow{\text{linear modes}} \delta_\mathrm{lin}
\xrightarrow{\text{LPT}} z_0
\xrightarrow{f_0,\ldots,f_{N-1}} z_N
\xrightarrow{\mathcal O} y,
$$

where $\omega$ is the random realization, $\theta$ the cosmology, and
$z=(\mathbf q_\mathrm{mesh},\mathbf s,\mathbf v,\mathbf a)$ the stored particle
state. A differentiable objective is $J(\omega,\theta)=\ell(y)$.

## Shapes and units

- configuration values define a periodic box of shape `conf.box_size` in the
  chosen length unit;
- white/linear Fourier modes use rFFT shape
  $(N_x,N_y,N_z/2+1)$, with a transposed sharding on a multi-device runtime;
- particle floating fields have logical shape $(N_p,3)$ but physical storage is
  padded per device;
- density has global shape `conf.mesh_shape` and default mean one;
- positions/displacements are in the configured length unit, while time is
  parameterized by the scale factor $a$.

## Implementation anchors

`src/pmpp/configuration.py` defines a frozen all-static configuration PyTree.
`src/pmpp/cosmo.py` keeps cosmological parameters as dynamic PyTree leaves.
`src/pmpp/particles.py` stores integer mesh indices separately from floating
displacement. `src/pmpp/nbody.py` exposes the public evolution and custom VJP.

## Design trade-offs

- **Frozen configuration:** static shapes and topology make JIT/sharding
  predictable, but changing a configuration usually recompiles.
- **Integer anchor plus displacement:** avoids an extra persistent particle-ID
  array and gives stable mesh-relative geometry. A drift advances `disp`; when
  ownership changes, routing must carry the `pmid` anchor and displacement
  together as one particle record.
- **One public pipeline:** the two-GPU documentation baseline and larger runs
  share the same scientific stages; only the device topology and corresponding
  static storage/communication layout change.
- **Explicit adjoints:** reduce time-history memory and encode communication
  transposes, at the cost of more solver-specific backward code.

## Validation

Validate in layers: deterministic modes, LPT displacements, mass-conserving
scatter, force parity, one-step evolution, and finally full forward/gradient
runs. Multi-GPU validation additionally requires zero capacity errors and checks
that boundary-visible projections do not show slab seams.
