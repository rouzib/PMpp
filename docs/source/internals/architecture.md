# System architecture

## Purpose

PM++ turns a static experiment description plus differentiable inputs into a
particle trajectory, observable, and optionally gradients. The main design goal
is to preserve one scientific pipeline while changing its storage and
communication layout across devices.

## Full PM++ workflow

This is a node-for-node Mermaid reconstruction of `PM++_flow_h.png`: all 34
boxes from the reference are present in the same six system regions. Drag to
pan, use the mouse wheel or trackpad to zoom, or use the controls in the
upper-left corner. The same controls are available in fullscreen view.

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;

  CLOUD["cloud / machine<br/>architecture"]:::comm

  subgraph SETUP["SETUP AND LINEAR THEORY"]
    direction TB
    COSMO["cosmo.py"]:::state
    CONF["configuration.py"]:::comm
    UNITS["units + box<br/>geometry"]:::state
    RESOLUTION["particle + mesh<br/>resolution"]:::state
    BOLTZ["boltzmann.py<br/>transfer + growth + P(k)"]:::op
    DEVICES["device mesh +<br/>slab layout"]:::comm
    HALO_LIMITS["halo regions +<br/>share limits"]:::comm
    HELPERS["runtime helpers<br/>FFT / scatter / gather / movement"]:::comm
    SCHEDULE["time stepping +<br/>save schedule"]:::comm

    COSMO --> BOLTZ
    CONF --> UNITS
    CONF --> RESOLUTION
    CONF --> DEVICES
    CONF --> HALO_LIMITS
    CONF --> HELPERS
    CONF --> SCHEDULE
  end

  subgraph INITIAL["INITIAL CONDITIONS"]
    direction TB
    MODES["modes.py<br/>white noise + linear modes"]:::state
    LPT["lpt.py<br/>particle initial conditions"]:::state
    MODES --> LPT
  end

  subgraph RUNTIME["DISTRIBUTED RUNTIME INTERNALS"]
    direction TB
    SLAB["owned slab per<br/>GPU"]:::comm
    HALO_BANDS["left / right halo<br/>bands"]:::comm
    CAPACITIES["neighbor permutations +<br/>capacities"]:::comm
    FFT_PLAN["FFT_distributed.py<br/>sharded FFT plan"]:::comm
  end

  subgraph STEP["MAIN PARTICLE-MESH TIMESTEP"]
    direction LR
    SCATTER["1. scatter.py<br/>deposit particle mass<br/>to local density mesh"]:::op
    FFT_FORWARD["2. FFT_distributed.py<br/>forward distributed FFT"]:::comm
    POISSON["3. gravity.py<br/>k-space Poisson solve"]:::op
    FFT_INVERSE["4. FFT_distributed.py<br/>inverse distributed FFT"]:::comm
    GATHER["5. gather.py<br/>(gather_old.py in reference)<br/>interpolate mesh acceleration<br/>to particle slots"]:::op
    SLOTS["particles.py<br/>owned slots + halo/spare slots<br/>(pmid + disp + vel + acc)"]:::state
    KICK["6. steps.py / kick<br/>vel = vel + acc"]:::op
    DRIFT["7. steps.py / drift<br/>disp = disp + vel"]:::op
    MIGRATE["8. particles.py + configuration.py<br/>cross-device migration"]:::comm
    REFRESH["9. refresh halo / mesh slots<br/>for next force solve"]:::comm

    SCATTER -->|"local density slabs"| FFT_FORWARD
    FFT_FORWARD -->|"global k-space density"| POISSON
    POISSON -->|"k-space acceleration field"| FFT_INVERSE
    FFT_INVERSE -->|"local acceleration mesh"| GATHER
    GATHER -->|"particle accelerations"| SLOTS
    SLOTS --> KICK --> DRIFT --> MIGRATE --> REFRESH
    REFRESH -->|"next PM step input"| SLOTS
    SLOTS -->|"positions / mass"| SCATTER
  end

  subgraph OUTPUTS["NBODY LOOP + OUTPUTS"]
    direction TB
    NBODY["nbody.py<br/>repeat PM timestep<br/>over scale factor a"]:::op
    SNAPSHOTS["particle<br/>snapshots"]:::state
    MAPS["2D maps /<br/>projections"]:::state
    ENSEMBLES["multi-seed /<br/>CAMELS workflows"]:::state

    NBODY --> SNAPSHOTS
    NBODY --> MAPS --> ENSEMBLES
  end

  subgraph DIFF["DIFFERENTIABLE WORKFLOWS"]
    direction TB
    CUSTOM_VJP["scatter / gravity / gather / nbody<br/>custom VJPs"]:::check
    GRAD_TESTS["gradient<br/>comparison tests"]:::check
    RUN_GRAD["scripts /<br/>run_grad.py"]:::check
    INFERENCE["calibration /<br/>optimization / SBI"]:::check

    CUSTOM_VJP --> GRAD_TESTS --> RUN_GRAD --> INFERENCE
  end

  CLOUD --> DEVICES
  UNITS --> MODES
  RESOLUTION --> MODES
  RESOLUTION --> LPT
  BOLTZ --> MODES
  DEVICES --> SLAB
  DEVICES --> FFT_PLAN
  HALO_LIMITS --> HALO_BANDS
  HALO_LIMITS --> CAPACITIES
  HELPERS --> FFT_PLAN
  LPT -->|"initialize particles"| SLOTS

  SLAB --> SLOTS
  HALO_BANDS -.-> SCATTER
  HALO_BANDS -.-> GATHER
  HALO_BANDS -.-> REFRESH
  CAPACITIES --> MIGRATE
  CAPACITIES --> REFRESH
  FFT_PLAN --> FFT_FORWARD
  FFT_PLAN --> FFT_INVERSE

  SCHEDULE --> NBODY
  NBODY -->|"start each PM step"| SCATTER
  REFRESH -->|"continue loop"| NBODY
  SLOTS --> SNAPSHOTS
  SCATTER -->|"local density"| MAPS

  SCATTER -.-> CUSTOM_VJP
  POISSON -.-> CUSTOM_VJP
  GATHER -.-> CUSTOM_VJP
  NBODY -.-> CUSTOM_VJP
```

**Text equivalent:** the cloud or machine topology, cosmology, and configuration
define the units, box, resolutions, transfer functions, device slabs, halo
limits, runtime helpers, and integration schedule. White noise and linear modes
feed LPT particle initial conditions. Runtime internals provide owned GPU slabs,
halo bands, neighbor capacities, and the distributed FFT plan. Every PM step
scatters particle mass, performs the forward FFT, solves Poisson's equation,
performs the inverse FFT, gathers acceleration, kicks and drifts particles,
migrates slab-crossing particles, and refreshes halos. `nbody.py` repeats that
step and emits snapshots, projections, and ensemble workflows. Custom VJPs feed
gradient tests, gradient scripts, and calibration, optimization, or SBI.

:::{note}
The reference image names `gather_old.py` and particle “halo slots.” The active
implementation uses `gather.py`; in recommended `mesh_halo` mode, only
authoritative particle slots are retained and mesh halos replace duplicated
particle halos. Both reference labels remain visible in the corresponding
nodes so the diagram is a complete crosswalk rather than a silent rewrite.
:::

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
