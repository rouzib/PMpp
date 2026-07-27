# System architecture

## Purpose

PM++ turns a static experiment description plus differentiable inputs into a
particle trajectory, observable, and optionally gradients. The main design goal
is to preserve one scientific pipeline while changing its storage and
communication layout across devices.

## Forward and adjoint workflow

This is the primary scientific workflow: inputs and LPT feed an unrolled
kick-drift-kick particle-mesh simulation, and the custom VJP traverses the same
operators in reverse. Solid arrows are forward data flow; dashed arrows are
gradient dependencies. Drag to pan, use the mouse wheel or trackpad to zoom, or
use the controls in the upper-left corner.

```{mermaid}
flowchart TD
  classDef cfg fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef state fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#111;
  classDef op fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#111;
  classDef fft fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  classDef grad fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#111;
  classDef io fill:#ECEFF1,stroke:#546E7A,stroke-width:2px,color:#111;
  classDef note fill:#FAFAFA,stroke:#9E9E9E,stroke-dasharray:4 4,color:#111;
  classDef section fill:#F8FAFC,stroke:#475569,stroke-width:2px,color:#111,font-weight:bold;

  START(["Inputs"]):::cfg

  subgraph INPUT_ROW[" "]
    direction LR
    THETA["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Cosmology</div><div class='pmpp-node-math'>$$\theta = \{\Omega_m,\Omega_b,h,A_s,n_s,\ldots\}$$</div></div>"]:::cfg
    WN["<div class='pmpp-node-stack'><div class='pmpp-node-title'>White-noise modes</div><div class='pmpp-node-math'>$$\omega(\mathbf{k})$$</div></div>"]:::cfg
    CONF["Frozen Configuration<br/>box, mesh, particles, schedule,<br/>device mesh, halos, capacities"]:::cfg
  end
  style INPUT_ROW fill:transparent,stroke:transparent

  START --> THETA
  START --> WN
  START --> CONF

  subgraph IC["Initial conditions and LPT"]
    direction TB
    PS["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Linear power and growth</div><div class='pmpp-node-math'>$$P_{\mathrm{lin}}(\mathbf{k};\theta)$$</div></div>"]:::op
    MODES["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Linear modes</div><div class='pmpp-node-math'>$$\delta_{\mathrm{lin}}(\mathbf{k}) = \sqrt{P_{\mathrm{lin}}(\mathbf{k};\theta)}\,\omega(\mathbf{k})$$</div></div>"]:::op
    LPT["<div class='pmpp-node-stack'><div class='pmpp-node-title'>ZA / 1LPT / 2LPT</div><div class='pmpp-node-math'>$$\mathbf{q}\mapsto \mathbf{x}(a_{\mathrm{init}}),\,\mathbf{p}(a_{\mathrm{init}})$$</div></div>"]:::op
    A0["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Initial PM force cache</div><div class='pmpp-node-math'>$$\mathbf{a}_{\mathrm{PM}}(a_{\mathrm{init}})$$</div></div>"]:::op
  end

  THETA --> PS
  WN --> MODES
  PS --> MODES --> LPT --> A0
  CONF -.->|"initial-condition configuration"| PS

  subgraph FWD["Forward simulation: unrolled kick-drift-kick PM steps"]
    direction TB

    S0["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Particle state at step n</div><div class='pmpp-node-math'>$$z_n=(\mathrm{pmid}_n,\mathbf{s}_n,\mathbf{v}_n,\mathbf{a}_n)$$</div></div>"]:::state
    K1["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Kick 1/2</div><div class='pmpp-node-math'>$$\mathbf{v}_{n+1/2}=\mathbf{v}_n+K_n\mathbf{a}_n$$</div></div>"]:::op
    D1["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Drift</div><div class='pmpp-node-math'>$$\mathbf{s}_{n+1}=\mathbf{s}_n+D_n\mathbf{v}_{n+1/2}$$</div></div>"]:::op
    MOVE["Domain update<br/>authoritative slab migration<br/>and static-slot compaction"]:::io

    subgraph FORCE[" "]
      direction TB
      FORCE_HEAD(["Distributed PM force solve at step n+1"]):::section
      SCATTER["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Scatter CIC + mesh-halo reduction</div><div class='pmpp-node-math'>$$\{\mathrm{pmid},\mathbf{s}\}\longrightarrow\rho(\mathbf{x})$$</div></div>"]:::op
      DELTA["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Density contrast</div><div class='pmpp-node-math'>$$\delta(\mathbf{x})=\rho(\mathbf{x})/\bar{\rho}-1$$</div></div>"]:::op
      FFT["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Distributed forward FFT</div><div class='pmpp-node-math'>$$\delta(\mathbf{x})\longrightarrow\delta(\mathbf{k})$$</div></div>"]:::fft
      POISSON["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Poisson kernel in k-space</div><div class='pmpp-node-math'>$$-k^2\Phi(\mathbf{k})=\delta(\mathbf{k})$$</div></div>"]:::fft
      GRADK["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Spectral gradient</div><div class='pmpp-node-math'>$$g_i(\mathbf{k})=\mathrm{i}k_i\Phi(\mathbf{k})$$</div></div>"]:::fft
      IFFT["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Distributed inverse FFTs</div><div class='pmpp-node-math'>$$g_i(\mathbf{k})\longrightarrow g_i(\mathbf{x})$$</div></div>"]:::fft
      GATHER["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Gather CIC from mesh halos</div><div class='pmpp-node-math'>$$g_i(\mathbf{x})\longrightarrow\mathbf{a}_{n+1}$$</div></div>"]:::op
    end

    K2["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Kick 1/2</div><div class='pmpp-node-math'>$$\mathbf{v}_{n+1}=\mathbf{v}_{n+1/2}+K_{n+1}\mathbf{a}_{n+1}$$</div></div>"]:::op
    OBS["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Optional observer / collector</div><div class='pmpp-node-math'>$$\text{loss contribution at }a_{n+1}$$</div></div>"]:::io
    S1["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Particle state at step n+1</div><div class='pmpp-node-math'>$$z_{n+1}=(\mathrm{pmid}_{n+1},\mathbf{s}_{n+1},\mathbf{v}_{n+1},\mathbf{a}_{n+1})$$</div></div>"]:::state
  end

  A0 --> S0
  CONF -.->|"static runtime"| S0
  S0 --> K1 --> D1 --> MOVE --> FORCE_HEAD --> SCATTER
  SCATTER --> DELTA --> FFT --> POISSON --> GRADK --> IFFT --> GATHER
  GATHER --> K2 --> OBS --> S1

  S1 --> LOOP{"<div class='pmpp-node-stack'><div class='pmpp-node-title'>Repeat</div><div class='pmpp-node-math'>$$n=0,\ldots,N-1$$</div></div>"}:::note
  LOOP -->|"next step"| S0
  LOOP -->|"after final step"| LOSS
  LOSS["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Final scalar objective</div><div class='pmpp-node-math'>$$L$$</div></div>"]:::grad

  subgraph BWD["Backward pass: discrete adjoint and reverse-time reconstruction"]
    direction TB

    LN["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Seed final cotangent</div><div class='pmpp-node-math'>$$\lambda_{\mathbf{s},N}=\frac{\partial L}{\partial\mathbf{s}_N},\quad\lambda_{\mathbf{v},N}=\frac{\partial L}{\partial\mathbf{v}_N}$$</div></div>"]:::grad
    RK2["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Reverse Kick 1/2: reconstruct and pull back through K2</div><div class='pmpp-node-math'>$$\mathbf{v}_{n+1/2}$$</div></div>"]:::grad

    subgraph FORCE_T["Adjoint of the distributed PM force solve"]
      direction TB
      GADJ["Adjoint of Gather<br/>particle force cotangents to mesh"]:::grad
      IFFTA["Adjoint of inverse FFT<br/>weighted distributed forward FFT"]:::grad
      GRADKA["Adjoint of spectral gradient"]:::grad
      POISSA["Adjoint of Poisson kernel"]:::grad
      FFTA["Adjoint of forward FFT<br/>weighted distributed inverse FFT"]:::grad
      SADJ["Adjoint of Scatter<br/>mesh cotangents to particles"]:::grad
      HALORED["Transpose mesh-halo exchange<br/>sum boundary cotangents"]:::io
    end

    RMOVE["Reverse migration / routing<br/>reconstruct pre-drift ownership"]:::io
    RDRIFT["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Reverse Drift: reconstruct state and pull back coefficient</div><div class='pmpp-node-math'>$$\mathbf{s}_n,\;D_n$$</div></div>"]:::grad
    RESTORE["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Restore incoming force cache for the earlier kick</div><div class='pmpp-node-math'>$$\mathbf{a}_n$$</div></div>"]:::grad
    RK1["Reverse Kick 1/2<br/>pull back through K1"]:::grad
    PREV["<div class='pmpp-node-stack'><div class='pmpp-node-title'>State and cotangent</div><div class='pmpp-node-math'>$$z_n,\lambda_n$$</div></div>"]:::grad
  end

  LOSS --> LN --> RK2 --> GADJ --> IFFTA --> GRADKA --> POISSA --> FFTA --> SADJ --> HALORED
  HALORED --> RMOVE --> RDRIFT --> RESTORE --> RK1 --> PREV
  PREV --> BACKLOOP{"$$n\leftarrow n-1$$"}:::note
  BACKLOOP -->|"continue to a_init"| RK2
  BACKLOOP -->|"at a_init"| GTHETA["<div class='pmpp-node-stack'><div class='pmpp-node-title'>Initial-condition and cosmology pullback</div><div class='pmpp-node-math'>$$\nabla_{\theta}L,\quad\nabla_{\omega}L$$</div></div>"]:::grad
```

**Text equivalent:** cosmology, white-noise modes, and a frozen configuration
produce linear modes, LPT particles, and an initial force cache. Each leapfrog
step kicks, drifts, migrates authoritative particles, scatters mass, performs a
distributed Fourier/Poisson force solve, gathers acceleration, applies the
second kick, and optionally observes the state. The discrete adjoint starts at
the final loss and reverses the second kick, gather, inverse FFT, spectral
gradient, Poisson kernel, forward FFT, scatter, mesh-halo exchange, ownership
routing, drift, and first kick. It reconstructs rather than stores the full
trajectory and accumulates gradients only for supported dynamic inputs.

The equations use familiar physical shorthand. In storage, PM++ represents
position with the integer mesh anchor `pmid` plus floating displacement `disp`,
and uses `vel` and `acc` for momentum-like velocity and the cached PM force.

## Implementation map

This code-linked map follows the same phase-oriented structure as the detailed
PM++ implementation diagram: runtime construction, initialization, forward
evolution, readout, the custom adjoint, and gradient destinations. It shows the
current `mesh_halo` path rather than duplicated particle halos. Drag to pan, use
the mouse wheel or trackpad to zoom, or use the controls in the upper-left
corner. The same controls are available in fullscreen view.

```{mermaid}
flowchart LR
  classDef runtime fill:#ECEFF1,stroke:#546E7A,stroke-width:1.5px,color:#111;
  classDef init fill:#E8F5E9,stroke:#2E7D32,stroke-width:1.5px,color:#111;
  classDef forward fill:#E3F2FD,stroke:#1565C0,stroke-width:1.5px,color:#111;
  classDef force fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#111;
  classDef output fill:#F3E5F5,stroke:#7B1FA2,stroke-width:1.5px,color:#111;
  classDef backward fill:#FCE4EC,stroke:#C2185B,stroke-width:1.5px,color:#111;
  classDef gradient fill:#FFF3E0,stroke:#EF6C00,stroke-width:1.5px,color:#111;
  classDef static fill:#FAFAFA,stroke:#9E9E9E,stroke-dasharray:4 4,color:#111;
  classDef phase fill:#FFFFFF,stroke:#334155,stroke-width:2.5px,color:#111,font-weight:bold;

  subgraph RUNTIME_CONFIG[" "]
    direction TB
    RC_HEAD(["RUNTIME + CONFIGURATION"]):::phase
    CONFIG_FILE["configuration.py<br/>frozen Configuration PyTree"]:::runtime
    MGPU_FILE["multigpu_configuration.py<br/>MultiGPUConfiguration + compute mesh"]:::runtime
    PHYSICAL["physical parameters<br/>particle spacing / grids / mesh / box / units"]:::runtime
    DECOMP["slab decomposition<br/>device mesh / owned slice start and end"]:::runtime
    HALO_COMM["mesh-halo communication<br/>halo bounds / left and right permutations"]:::runtime
    BUFFERS["static capacities<br/>particles per slice / migration / gather buffers"]:::runtime
    METHODS["JIT-bound runtime methods<br/>scatter / gather / movement / distributed FFTs"]:::runtime
    METADATA["static execution metadata<br/>layouts / permutations / capacities / schedule"]:::static

    RC_HEAD --> CONFIG_FILE
    RC_HEAD --> MGPU_FILE
    CONFIG_FILE --> PHYSICAL
    CONFIG_FILE --> BUFFERS
    CONFIG_FILE --> METADATA
    MGPU_FILE --> DECOMP
    MGPU_FILE --> HALO_COMM
    MGPU_FILE --> METHODS
    MGPU_FILE --> METADATA
  end
  style RUNTIME_CONFIG fill:#F8FAFC,stroke:#64748B,stroke-width:2px

  subgraph INIT_PHASE[" "]
    direction TB
    INIT_HEAD(["PHASE 0 - INITIALIZATION"]):::phase
    COSMO_FILE["cosmo.py<br/>cosmology PyTree and growth parameters"]:::init
    BOLTZ_FILE["boltzmann.py<br/>transfer and growth tables / linear P(k)"]:::init
    WHITE_NOISE["modes.py<br/>white_noise or white_noise_nested"]:::init
    LINEAR_MODES["linear_modes<br/>noise scaled by the linear spectrum"]:::init
    LPT_POTENTIAL["lpt.py<br/>Fourier-space LPT potential solves"]:::init
    LPT_DISP["LPT displacement<br/>spectral gradients to real space"]:::init
    LPT_VEL["LPT velocity<br/>growth-rate-scaled displacement"]:::init
    INIT_PARTICLES["particles.py<br/>pmid + disp + vel + acc + unused slots"]:::init
    INIT_REDISTRIBUTE["initial ownership correction<br/>move particles to authoritative GPU slabs"]:::init
    STATE_A0["distributed particle state<br/>at a_start"]:::init

    INIT_HEAD --> COSMO_FILE --> BOLTZ_FILE
    INIT_HEAD --> WHITE_NOISE
    WHITE_NOISE --> LINEAR_MODES
    BOLTZ_FILE --> LINEAR_MODES
    LINEAR_MODES --> LPT_POTENTIAL
    LPT_POTENTIAL --> LPT_DISP
    LPT_POTENTIAL --> LPT_VEL
    LPT_DISP --> INIT_PARTICLES
    LPT_VEL --> INIT_PARTICLES
    INIT_PARTICLES --> INIT_REDISTRIBUTE --> STATE_A0
  end
  style INIT_PHASE fill:#F1F8F4,stroke:#2E7D32,stroke-width:2px

  PHYSICAL ==> COSMO_FILE
  DECOMP ==> INIT_REDISTRIBUTE
  HALO_COMM -.-> INIT_REDISTRIBUTE

  subgraph FORWARD_PASS[" "]
    direction TB
    FWD_HEAD(["PHASE 1 - FORWARD PASS: N-BODY EVOLUTION"]):::phase
    NBODY_FWD["nbody.py<br/>custom-VJP forward entry"]:::forward
    FWD_SCAN["JIT-compiled lax.scan<br/>over the scale-factor schedule"]:::forward
    FWD_RUNTIME["compiled runtime hooks<br/>static routing, capacities, FFT and halo methods"]:::static
    STEP_I["state at step i<br/>authoritative particles in padded GPU slots"]:::forward
    KICK_1["steps.py - first half kick<br/>update velocity from cached acceleration"]:::forward
    DRIFT["steps.py - drift_for_force<br/>update displacement"]:::forward
    MOVE_DETECT["detect slab crossings<br/>pack left and right migration buffers"]:::forward
    PPERMUTE["jax.lax.ppermute<br/>neighbor particle exchange"]:::forward
    UNPACK["insert received particles<br/>compact authoritative static slots"]:::forward
    SCATTER_CIC["scatter.py<br/>CIC deposit to owned density slab"]:::forward
    MESH_HALO_REDUCE["mesh-halo reduction<br/>sum density contributions at slab boundaries"]:::forward
    RFFT_YZ["FFT_distributed.py<br/>local transforms on slab-resident dimensions"]:::forward
    CORNER_FWD["forward corner turn<br/>all_to_all transposed layout"]:::forward
    RFFT_X["complete distributed RFFT<br/>global k-space density"]:::forward
    POISSON["gravity.py<br/>Poisson kernel + spectral gradient"]:::force
    IRFFT_X["start distributed inverse FFT<br/>from transposed k-space fields"]:::forward
    CORNER_BWD["inverse corner turn<br/>restore slab layout"]:::forward
    IRFFT_YZ["finish inverse FFTs<br/>real-space acceleration meshes"]:::forward
    MESH_HALO_REFRESH["exchange acceleration mesh halos<br/>make boundary cells locally available"]:::forward
    GATHER_LOCAL["gather.py<br/>local CIC interpolation to particle slots"]:::forward
    GATHER_EXCHANGE["boundary gather exchange<br/>return edge values to authoritative owners"]:::forward
    KICK_2["steps.py - second half kick<br/>complete the macro-step velocity update"]:::forward
    STEP_I1["state at step i+1"]:::forward
    OBSERVE["optional observers / collectors<br/>snapshots and differentiable summaries"]:::forward
    FWD_REPEAT["repeat scan<br/>for remaining timesteps"]:::static
    FINAL_STATE["final distributed particle state<br/>at a_stop"]:::forward
    SAVED_CONTEXT["custom-VJP residual context<br/>schedule / sparse state / static communication metadata"]:::static

    FWD_HEAD --> NBODY_FWD --> FWD_SCAN --> STEP_I
    FWD_HEAD -.-> FWD_RUNTIME
    STEP_I --> KICK_1 --> DRIFT --> MOVE_DETECT --> PPERMUTE --> UNPACK
    UNPACK --> SCATTER_CIC --> MESH_HALO_REDUCE --> RFFT_YZ --> CORNER_FWD --> RFFT_X
    RFFT_X --> POISSON --> IRFFT_X --> CORNER_BWD --> IRFFT_YZ
    IRFFT_YZ --> MESH_HALO_REFRESH --> GATHER_LOCAL --> GATHER_EXCHANGE --> KICK_2
    KICK_2 --> STEP_I1 --> OBSERVE --> FWD_REPEAT --> FINAL_STATE
    FWD_SCAN --> SAVED_CONTEXT
    FWD_RUNTIME -.-> MOVE_DETECT
    FWD_RUNTIME -.-> MESH_HALO_REDUCE
    FWD_RUNTIME -.-> CORNER_FWD
    FWD_RUNTIME -.-> GATHER_EXCHANGE
  end
  style FORWARD_PASS fill:#F4F8FD,stroke:#1565C0,stroke-width:2px

  STATE_A0 ==> STEP_I
  METADATA -.-> FWD_RUNTIME
  METHODS -.-> FWD_RUNTIME
  HALO_COMM -.-> FWD_RUNTIME

  subgraph OUTPUTS_LOSS[" "]
    direction TB
    OUT_HEAD(["PHASE 2 - RESULTS / READOUT / LOSS"]):::phase
    SNAPSHOTS["particle snapshots"]:::output
    MAPS["density maps and projections"]:::output
    SUMMARY["observable readout<br/>summary statistics / map features"]:::output
    TARGET["target summaries<br/>observations or reference outputs"]:::output
    LOSS["scalar loss / objective"]:::output

    OUT_HEAD --> SNAPSHOTS
    OUT_HEAD --> MAPS
    OUT_HEAD --> SUMMARY
    MAPS --> SUMMARY
    SUMMARY --> LOSS
    TARGET --> LOSS
  end
  style OUTPUTS_LOSS fill:#FAF5FC,stroke:#7B1FA2,stroke-width:2px

  FINAL_STATE ==> OUT_HEAD

  subgraph BACKWARD_PASS[" "]
    direction TB
    BWD_HEAD(["PHASE 3 - BACKWARD PASS: DISCRETE ADJOINT"]):::phase
    LOSS_SEED["incoming output cotangent<br/>dL / d(output)"]:::backward
    READOUT_ADJ["adjoint of readout and loss<br/>map outputs to final-state cotangents"]:::backward
    FINAL_BAR["cotangent of final state<br/>position / velocity / acceleration"]:::backward
    NBODY_BWD["nbody.py<br/>custom-VJP backward entry"]:::backward
    REVERSE_SCAN["reverse timestep traversal<br/>N-1 down to 0"]:::backward
    BWD_RUNTIME["saved schedule + runtime routing<br/>reconstruct the exact discrete path"]:::static
    RECOMPUTE["reverse-time state reconstruction<br/>recover the required forward state"]:::backward
    KICK_2_ADJ["reverse second half kick"]:::backward
    GATHER_EXCHANGE_ADJ["transpose boundary gather exchange<br/>route cotangents back across slabs"]:::backward
    GATHER_LOCAL_ADJ["adjoint of gather<br/>particle-force cotangents to mesh"]:::backward
    IRFFT_ADJ["adjoint of inverse FFT<br/>forward transforms + reverse transpose"]:::backward
    POISSON_ADJ["adjoint of Poisson and spectral kernels<br/>same linear operator structure"]:::force
    RFFT_ADJ["adjoint of forward FFT<br/>inverse transforms + reverse transpose"]:::backward
    SCATTER_ADJ["adjoint of scatter<br/>mesh-density cotangents to particles"]:::backward
    HALO_GRAD_REDUCE["transpose mesh-halo exchange<br/>sum boundary cotangents"]:::backward
    REVERSE_UNPACK["reverse slot compaction<br/>restore pre-migration padded layout"]:::backward
    PPERMUTE_ADJ["inverse ppermute<br/>return cotangents to prior owners"]:::backward
    REVERSE_MOVE["reverse ownership migration<br/>reconstruct the pre-drift state"]:::backward
    DRIFT_ADJ["drift_adj_from_output<br/>reverse drift and accumulate factor gradients"]:::backward
    FORCE_RESTORE["restore incoming force cache<br/>recompute acceleration for the earlier kick"]:::backward
    KICK_1_ADJ["reverse first half kick"]:::backward
    STATE_I_BAR["state and cotangent<br/>at step i"]:::backward
    BWD_REPEAT["repeat reverse scan<br/>to the initial step"]:::static
    STATE_A0_BAR["cotangent of initial particle state"]:::backward

    BWD_HEAD --> LOSS_SEED --> READOUT_ADJ --> FINAL_BAR
    BWD_HEAD --> NBODY_BWD --> REVERSE_SCAN
    FINAL_BAR --> REVERSE_SCAN
    BWD_RUNTIME -.-> REVERSE_SCAN
    REVERSE_SCAN --> RECOMPUTE --> KICK_2_ADJ
    KICK_2_ADJ --> GATHER_EXCHANGE_ADJ --> GATHER_LOCAL_ADJ --> IRFFT_ADJ
    IRFFT_ADJ --> POISSON_ADJ --> RFFT_ADJ --> SCATTER_ADJ --> HALO_GRAD_REDUCE
    HALO_GRAD_REDUCE --> REVERSE_UNPACK --> PPERMUTE_ADJ --> REVERSE_MOVE --> DRIFT_ADJ
    DRIFT_ADJ --> FORCE_RESTORE --> KICK_1_ADJ --> STATE_I_BAR --> BWD_REPEAT --> STATE_A0_BAR
  end
  style BACKWARD_PASS fill:#FDF5F8,stroke:#C2185B,stroke-width:2px

  LOSS ==> LOSS_SEED
  SAVED_CONTEXT ==> BWD_RUNTIME

  subgraph GRADIENT_ENDPOINTS[" "]
    direction TB
    GRAD_HEAD(["PHASE 4 - GRADIENT DESTINATIONS"]):::phase
    INIT_PARTICLE_GRAD["gradient with respect to<br/>initial particle state"]:::gradient
    LPT_ADJ["backpropagate through LPT<br/>potential / displacement / velocity"]:::gradient
    LINEAR_ADJ["backpropagate through linear modes<br/>and power-spectrum scaling"]:::gradient
    WHITE_NOISE_GRAD["gradient with respect to<br/>initial white-noise modes"]:::gradient
    COSMO_GRAD["gradient with respect to selected<br/>cosmological parameters"]:::gradient
    CALIBRATION_GRAD["gradient with respect to<br/>differentiable calibration parameters"]:::gradient
    NO_RUNTIME_GRAD["static runtime metadata<br/>layouts / permutations / capacities<br/>not optimization targets"]:::static

    GRAD_HEAD --> INIT_PARTICLE_GRAD --> LPT_ADJ --> LINEAR_ADJ
    LINEAR_ADJ --> WHITE_NOISE_GRAD
    LINEAR_ADJ --> COSMO_GRAD
    GRAD_HEAD --> CALIBRATION_GRAD
    GRAD_HEAD -.-> NO_RUNTIME_GRAD
  end
  style GRADIENT_ENDPOINTS fill:#FFF8F0,stroke:#EF6C00,stroke-width:2px

  STATE_A0_BAR ==> INIT_PARTICLE_GRAD
```

**Text equivalent:** configuration freezes physical grids, the two-device slab
decomposition, mesh-halo routing, static capacities, and JIT-bound runtime
methods. Cosmology, transfer functions, white noise, linear modes, and LPT
produce an authoritative distributed particle state. The forward custom VJP
scans the integration schedule, migrates slab-crossing particles, deposits CIC
mass, reduces mesh halos, performs transposed distributed FFTs, applies the
Poisson and spectral-gradient kernels, refreshes acceleration halos, gathers
forces, completes the leapfrog step, and records optional readouts. The reverse
pass reconstructs the discrete trajectory and applies the transposes in reverse
order, including gather routing, distributed FFTs, mesh-halo reduction,
particle migration, drift, force-cache restoration, and kicks. Initial-state
cotangents then propagate through LPT and linear modes to supported white-noise
and cosmology targets; layouts, permutations, and capacities remain static.

:::{note}
The supplied reference uses the earlier `gather_old.py` and duplicated particle
halo model. This map intentionally follows the active `gather.py` and
recommended `mesh_halo` implementation: particle storage remains authoritative,
while density and acceleration boundary values move through mesh halos.
:::

Function anchors are `Configuration`, `MultiGPUConfiguration`,
`white_noise`/`linear_modes`, `lpt`, `nbody`, `scatter`, `gather`, `gravity`,
the observer helpers, and `pmpp.power_spectrum`.

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
