# PM++ system diagrams

This page presents PM++ as an explicit unrolled particle-mesh pipeline, then zooms into the blocks that are easiest to lose in a compact overview: initial conditions, the PM force solve, the one-step discrete integrator, the reverse-mode adjoint, the distributed particle representation, and the operator-level blocks that tie the equations directly to the implementation.

The equations use the pmwd/FastPM-style particle-mesh convention: Gaussian initial modes, LPT initial conditions, CIC scatter and gather, a Fourier-space Poisson solve for a scaled potential, and a second-order symplectic leapfrog step whose adjoint reverses the same discrete operators.

## Full unrolled simulation

The full simulation is shown top to bottom so the forward path, the force recomputation inside each step, and the reverse-time adjoint can be followed in one reading direction. Later sections zoom into each major block.

```{mermaid}
flowchart TD
  classDef cfg fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef state fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#111;
  classDef op fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#111;
  classDef fft fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  classDef comm fill:#FDECEA,stroke:#C62828,stroke-width:2px,color:#111;
  classDef grad fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#111;
  classDef io fill:#ECEFF1,stroke:#546E7A,stroke-width:2px,color:#111;
  classDef nte fill:#FAFAFA,stroke:#9E9E9E,stroke-width:1px,color:#111;
  Start([Inputs]):::cfg --> Theta["Cosmology theta"]:::cfg
  Start --> WN["Hermitian white noise modes omega(k)"]:::cfg
  Start --> Conf["Configuration: box, particles, mesh, schedule, slab layout"]:::cfg
  Theta --> IC1["IC1: transfer and growth tables, P_lin(k,theta)"]:::op
  WN --> IC2["IC2: delta_lin(k) = sqrt(P_lin) omega(k)"]:::op
  IC1 --> IC2 --> IC3["IC3: 2LPT or ZA displacement and momentum solve"]:::fft
  IC3 --> Init["Initial particle state: x_init and p_init"]:::state
  Conf --> Init
  Init --> InitMove["Initial owner and halo update"]:::comm
  InitMove --> InitForce["Initial PM force a_init = F(x_init)"]:::op
  InitForce --> S0["State s_n = (x_n, p_n, a_n)"]:::state
  S0 --> K1["K1: first half kick, p_half"]:::op
  K1 --> D1["D: drift, x_next"]:::op
  D1 --> Move["Move: migrate particles and refresh halos"]:::comm
  Move --> Scatter["F1: CIC scatter to mesh density"]:::op
  Scatter --> Delta["F2: overdensity delta"]:::op
  Delta --> FFT["F3: forward FFT"]:::fft
  FFT --> Pois["F4: Poisson solve for scaled potential"]:::fft
  Pois --> Grad["F5: spectral gradient"]:::fft
  Grad --> IFFT["F6: inverse FFTs to mesh force"]:::fft
  IFFT --> Gather["F7: CIC gather to particle acceleration"]:::op
  Gather --> K2["K2: second half kick, p_next"]:::op
  K2 --> Obs["Observation or loss contribution"]:::io
  Obs --> S1["State s_next = (x_next, p_next, a_next)"]:::state
  S1 --> Loop{"repeat until final step"}:::nte
  Loop -->|next forward step| K1
  Loop -->|final state| Loss["Final objective L"]:::grad
  Loss --> Seed["Adjoint seed at final state"]:::grad
  Seed --> RK2["Reverse K2"]:::grad
  RK2 --> AG["Adjoint gather"]:::grad
  AG --> AI["Adjoint inverse FFT"]:::fft
  AI --> AS["Adjoint spectral gradient"]:::fft
  AS --> AP["Adjoint Poisson kernel"]:::fft
  AP --> AF["Adjoint forward FFT"]:::fft
  AF --> AC["Adjoint scatter"]:::grad
  AC --> HR["Halo gradient reduction"]:::comm
  HR --> RD["Reverse drift"]:::grad
  RD --> RM["Reverse routing and ownership"]:::comm
  RM --> RK1["Reverse K1"]:::grad
  RK1 --> Prev["Adjoint at previous step"]:::grad
  Prev --> Back{"repeat backward to a_init"}:::nte
  Back -->|next reverse step| RK2
  Prev --> Out["Gradients with respect to theta, omega, and observation parameters"]:::grad
```

## Initial conditions and LPT

White-noise modes are scaled into a linear density field, then converted into Lagrangian displacements and canonical momenta. This block determines how gradients flow back to cosmological parameters and initial random modes.

```{mermaid}
flowchart TD
  classDef cfg fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef op fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#111;
  classDef fft fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  classDef state fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#111;
  classDef nte fill:#FAFAFA,stroke:#9E9E9E,stroke-width:1px,color:#111;
  Theta["theta"]:::cfg --> Pk["P_lin(k,theta)"]:::op
  WN["omega(k)"]:::cfg --> Modes["delta_lin(k) = sqrt(P_lin) omega(k)"]:::op
  Pk --> Modes --> Phi1["Solve Laplacian phi_1 = delta_lin"]:::fft --> Psi1["Psi_1 = -grad phi_1"]:::fft
  Psi1 --> LPT2["Combine growth factors to build displacement and momentum"]:::fft
  LPT2 --> InitState["Initial state: x = q + s, and p"]:::state
  InitState --> HaloInit["Assign owners and build halo copies"]:::nte
```

The linear Gaussian modes are

$$
\delta_{\mathrm{lin}}(\mathbf{k}) = \sqrt{P_{\mathrm{lin}}(k;\theta)}\,\omega(\mathbf{k}).
$$

Particles start at Lagrangian coordinates $\mathbf{q}$ and are displaced to comoving positions,

$$
\mathbf{x}(a) = \mathbf{q} + \mathbf{s}(\mathbf{q},a).
$$

In 2LPT form,

$$
\mathbf{s} = D_1\,\mathbf{s}^{(1)} + D_2\,\mathbf{s}^{(2)}, \qquad
\mathbf{p} = a^2 H\Big(D_1'\,\mathbf{s}^{(1)} + D_2'\,\mathbf{s}^{(2)}\Big),
$$

where

$$
\nabla^2 \phi^{(1)}(\mathbf{q}) = \delta_{\mathrm{lin}}(\mathbf{q}), \qquad
\mathbf{s}^{(1)} = -\nabla \phi^{(1)}.
$$

## PM force evaluation

The PM force kernel follows the implementation order: scatter, overdensity, FFT, Poisson solve, spectral differentiation, inverse FFT, and gather.

```{mermaid}
flowchart TD
  classDef ptcl fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#111;
  classDef op fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#111;
  classDef fft fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  classDef comm fill:#FDECEA,stroke:#C62828,stroke-width:2px,color:#111;
  P0["Particle positions x_p with owned and halo copies"]:::ptcl --> S1["Scatter: rho_g = sum_p m_p W_CIC"]:::op
  S1 --> S2["Density contrast: delta_g = rho_g / rho_bar - 1"]:::op
  S2 --> F1["Forward FFT: delta(x) to delta(k)"]:::fft
  F1 --> P1["Poisson solve: -k^2 Phi_bar = delta"]:::fft
  P1 --> G1["Spectral gradient: g_i(k) = i k_i Phi_bar"]:::fft
  G1 --> F2["Inverse FFTs: g_i(k) to g_i(x)"]:::fft
  F2 --> G2["Gather: a_p = sum_g W_CIC g_g"]:::op
  G2 --> A0["Particle accelerations a_p"]:::ptcl
  H1["Halo copies make local edge scatter correct"]:::comm -.-> S1
  H2["Distributed slab FFT transpose and all-to-all"]:::comm -.-> F1
  H3["Halo values support edge gather near boundaries"]:::comm -.-> G2
```

The physical potential satisfies

$$
\nabla_{\mathbf{x}}^2 \Phi(\mathbf{x},a) = \frac{3}{2}\Omega_{m0} H_0^2 a^{-1}\delta(\mathbf{x},a).
$$

PM++ factors out time dependence through a scaled potential,

$$
\nabla_{\mathbf{x}}^2 \bar\Phi(\mathbf{x},a) = \delta(\mathbf{x},a), \qquad
-k^2\bar\Phi(\mathbf{k}) = \delta(\mathbf{k}).
$$

With CIC assignment and gathering,

$$
\rho(\mathbf{x}_g) = \sum_p m_p W_{\mathrm{CIC}}(\mathbf{x}_g-\mathbf{x}_p), \qquad
\delta(\mathbf{x}_g) = \frac{\rho(\mathbf{x}_g)}{\bar\rho} - 1,
$$

$$
g_i(\mathbf{k}) = i k_i\bar\Phi(\mathbf{k}), \qquad
\mathbf{a}_p = \sum_g W_{\mathrm{CIC}}(\mathbf{x}_g-\mathbf{x}_p)\,\mathbf{g}(\mathbf{x}_g).
$$

## One discrete integration step

This is the discrete graph that the adjoint reverses: two half-kicks around a drift, with a full PM force recomputation after the drift.

```{mermaid}
flowchart TD
  classDef state fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#111;
  classDef op fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#111;
  classDef force fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  classDef comm fill:#FDECEA,stroke:#C62828,stroke-width:2px,color:#111;
  classDef obs fill:#ECEFF1,stroke:#546E7A,stroke-width:2px,color:#111;
  SIn["Input state: s_n = (x_n, p_n, a_n)"]:::state --> K1s["K1: p_half = p_n + K_n a_n"]:::op --> Ds["D: x_next = x_n + D_n p_half"]:::op --> Ms["Move: pack, exchange, unpack, update masks"]:::comm --> Fs["F: recompute force a_next = F(x_next)"]:::force --> K2s["K2: p_next = p_half + K_next a_next"]:::op --> Os["Obs: optional observation or loss term"]:::obs --> SOut["Output state: s_next = (x_next, p_next, a_next)"]:::state
```

The leapfrog step is

$$
\mathbf{p}_{n+\frac12} = \mathbf{p}_n + K_n\mathbf{a}_n, \qquad
\mathbf{x}_{n+1} = \mathbf{x}_n + D_n\mathbf{p}_{n+\frac12},
$$

$$
\mathbf{a}_{n+1} = F\!\left(\mathbf{x}_{n+1}\right), \qquad
\mathbf{p}_{n+1} = \mathbf{p}_{n+\frac12} + K_{n+1}\mathbf{a}_{n+1}.
$$

Here $D_n$ and $K_n$ are FastPM drift and kick factors. PM++ follows the pmwd convention of choosing them from the ZA growth history so the finite-step update better tracks linear evolution.

$$
s_{n+1} = f_n(s_n,\theta), \qquad
L = \phi(s_N,\theta) + \sum_{n=0}^{N-1}\ell_n(s_n,\theta).
$$

## Backward pass through one PM step

The backward pass applies the transpose of each discrete operator in exact reverse order: reverse K2, reverse the force block, reverse drift and ownership updates, and finally reverse K1.

```{mermaid}
flowchart TD
  classDef grad fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#111;
  classDef fft fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  classDef comm fill:#FDECEA,stroke:#C62828,stroke-width:2px,color:#111;
  classDef step fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#111;
  L0["Incoming adjoint at step n+1"]:::grad --> R2["Reverse K2: gradients to p_half and a_next"]:::grad --> AG["Adjoint gather: particle force gradients to mesh"]:::step --> AI["Adjoint inverse FFT = forward FFT"]:::fft --> AS["Adjoint spectral gradient"]:::fft --> AP["Adjoint Poisson kernel"]:::fft --> AF["Adjoint forward FFT = inverse FFT"]:::fft --> AC["Adjoint scatter: mesh density gradients to particle positions"]:::step --> HR["Reduce halo copy gradients to owners"]:::comm --> RD["Reverse drift: gradients to x_n and p_half"]:::grad --> RM["Reverse routing and ownership transfer"]:::comm --> R1["Reverse K1: gradients to p_n and a_n"]:::grad --> L1["Outgoing adjoint at step n"]:::grad
```

For $s_{n+1} = f_n(s_n,\theta)$, the adjoint recurrence is

$$
\lambda_n = \left(\frac{\partial f_n}{\partial s_n}\right)^\top\lambda_{n+1} + \frac{\partial \ell_n}{\partial s_n}, \qquad
\lambda_N = \frac{\partial \phi}{\partial s_N}.
$$

The parameter gradient accumulates as

$$
\frac{\partial L}{\partial \theta} = \frac{\partial \phi}{\partial \theta} + \sum_{n=0}^{N-1}\left[\lambda_{n+1}^\top\frac{\partial f_n}{\partial \theta} + \frac{\partial \ell_n}{\partial \theta}\right].
$$

With

$$
\rho_g = \sum_p m_p W_{gp}(\mathbf{x}_p), \qquad
\mathbf{a}_p = \sum_g W_{gp}(\mathbf{x}_p)\mathbf{g}_g,
$$

the gather adjoint is scatter-like,

$$
\bar{\mathbf{g}}_g \;{+}{=}\; \sum_p W_{gp}(\mathbf{x}_p)\bar{\mathbf{a}}_p,
$$

while the scatter adjoint contributes position gradients through the stencil derivative,

$$
\bar{\mathbf{x}}_p \;{+}{=}\; \sum_g \bar{\rho}_g\,\frac{\partial}{\partial \mathbf{x}_p}\Big(m_p W_{gp}(\mathbf{x}_p)\Big).
$$

## Distributed ownership and static particle layout

PM++ uses slab decomposition. Each device owns one slab of particles and keeps halo copies so edge-adjacent scatter and gather can run locally. Particle arrays remain static in size for JAX compilation, so ownership updates use masks and fixed-capacity exchange buffers.

```{mermaid}
flowchart TD
  classDef gpu fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef owned fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px,color:#111;
  classDef ghost fill:#FFCCBC,stroke:#D84315,stroke-width:2px,color:#111;
  classDef comm fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  classDef nte fill:#FAFAFA,stroke:#9E9E9E,stroke-width:1px,color:#111;
  G0["Global slab order: GPU zero, GPU one, GPU two"]:::gpu --> G1["Local storage on one GPU"]:::gpu --> O0["Owned particles"]:::owned --> H0["Halo copies near slab edges"]:::ghost --> U0["Unused padded slots"]:::nte --> D0["Drift updates positions"]:::comm --> C0{"Crosses slab boundary"}:::nte
  C0 -->|yes| P0["Pack fixed send buffers"]:::comm
  P0 --> X0["Neighbor exchange by ppermute"]:::comm
  X0 --> N0["Unpack into unused slots"]:::comm
  N0 --> M0["Update owner mask and unused index"]:::comm
  C0 -->|no| M0
  M0 --> H1["Refresh halos for next scatter and gather"]:::comm
  H1 --> S0["Static shaped particle state on device"]:::gpu
```

A stable distributed representation separates an integer mesh index from a floating displacement,

$$
\mathbf{x}_p = \mathbf{i}_p\Delta x + \mathbf{d}_p,
$$

with static-capacity arrays for positions, momenta, accelerations, and masks. This layout supports JAX compilation, fixed communication buffers, and owner and halo bookkeeping during drift and reverse drift.

## CIC operators and adjoint pairing

The same CIC stencil appears twice: once to place mass on the mesh, once to read the mesh force back at particle locations.

```{mermaid}
flowchart TD
  classDef ptcl fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#111;
  classDef mesh fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef op fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#111;
  classDef grad fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#111;
  P["Particle positions and masses"]:::ptcl --> S["Scatter with CIC stencil W_gp"]:::op --> Rho["Mesh density rho_g"]:::mesh
  G["Mesh force g_g"]:::mesh --> T["Gather with same CIC stencil W_gp"]:::op --> A["Particle acceleration a_p"]:::ptcl
  A --> TG["Adjoint gather sends gradients back to mesh"]:::grad --> G
  Rho --> TS["Adjoint scatter sends gradients back to particle positions"]:::grad --> P
```

$$
\rho_g = \sum_p m_p W_{gp}(\mathbf{x}_p), \qquad
\mathbf{a}_p = \sum_g W_{gp}(\mathbf{x}_p)\mathbf{g}_g.
$$

Reverse mode uses the transpose action of the same stencil on the gather side and the stencil derivative on the scatter side,

$$
\bar{\mathbf{g}}_g \;{+}{=}\; \sum_p W_{gp}(\mathbf{x}_p)\bar{\mathbf{a}}_p, \qquad
\bar{\mathbf{x}}_p \;{+}{=}\; \sum_g \bar{\rho}_g\,\partial_{\mathbf{x}_p}\!\left(m_p W_{gp}(\mathbf{x}_p)\right).
$$

## Spectral force operator chain

The Fourier-space part of the PM solve connects the spectral equations to the exact code-path order.

```{mermaid}
flowchart TD
  classDef mesh fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef fft fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  D0["delta on the mesh"]:::mesh --> D1["Forward FFT"]:::fft
  D1 --> D2["delta(k)"]:::fft
  D2 --> D3["Apply Poisson kernel"]:::fft
  D3 --> D4["Phi_bar(k)"]:::fft
  D4 --> D5["Apply i k_x, i k_y, i k_z"]:::fft
  D5 --> D6["g_x(k), g_y(k), g_z(k)"]:::fft
  D6 --> D7["Inverse FFTs"]:::fft
  D7 --> D8["g_x(x), g_y(x), g_z(x)"]:::mesh
```

$$
-k^2\bar{\Phi}(\mathbf{k}) = \delta(\mathbf{k}), \qquad
g_i(\mathbf{k}) = i k_i \bar{\Phi}(\mathbf{k}).
$$

The force block is therefore a sequence of one forward FFT, a Poisson kernel application, three spectral derivatives, and three inverse FFTs.

## Reverse of the spectral force chain

The reverse Fourier-space force operator applies the transpose of each discrete linear map in reverse order.

```{mermaid}
flowchart TD
  classDef mesh fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef fft fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#111;
  classDef grad fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#111;
  A0["Particle force gradient"]:::grad --> A1["Adjoint gather to mesh"]:::grad
  A1 --> A2["Adjoint inverse FFT"]:::fft
  A2 --> A3["Adjoint spectral gradient"]:::fft
  A3 --> A4["Adjoint Poisson kernel"]:::fft
  A4 --> A5["Adjoint forward FFT"]:::fft
  A5 --> A6["Mesh density gradient"]:::mesh
  A6 --> A7["Adjoint scatter to particle positions"]:::grad
```

If the forward operator is

$$
\delta(\mathbf{x}) \rightarrow \delta(\mathbf{k}) \rightarrow \bar{\Phi}(\mathbf{k}) \rightarrow g_i(\mathbf{k}) \rightarrow g_i(\mathbf{x}),
$$

then the reverse operator is

$$
\bar{g}_i(\mathbf{x}) \rightarrow \bar{g}_i(\mathbf{k}) \rightarrow \bar{\Phi}(\mathbf{k}) \rightarrow \bar{\delta}(\mathbf{k}) \rightarrow \bar{\delta}(\mathbf{x}).
$$

## Time-step coefficients and schedule

The leapfrog uses step-dependent drift and kick coefficients derived from the selected scale-factor grid.

```{mermaid}
flowchart TD
  classDef cfg fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef op fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#111;
  classDef state fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#111;
  A0["Scale factor grid a_0 to a_N"]:::cfg --> A1["Midpoints and intervals"]:::op
  A1 --> A2["Drift coefficients D_n"]:::op
  A1 --> A3["Kick coefficients K_n"]:::op
  A2 --> A4["Step n uses D_n in the drift"]:::state
  A3 --> A5["Step n uses K_n and K_n+1 in the kicks"]:::state
```

The simulation uses a prescribed scale-factor grid,

$$
a_0 < a_1 < \cdots < a_N,
$$

and each step uses coefficients derived from that grid,

$$
\mathbf{p}_{n+\frac12} = \mathbf{p}_n + K_n\mathbf{a}_n, \qquad
\mathbf{x}_{n+1} = \mathbf{x}_n + D_n\mathbf{p}_{n+\frac12}, \qquad
\mathbf{p}_{n+1} = \mathbf{p}_{n+\frac12} + K_{n+1}\mathbf{a}_{n+1}.
$$

## Gradient targets

The same adjoint sweep can differentiate with respect to cosmological parameters, initial conditions, or observation-model parameters.

```{mermaid}
flowchart TD
  classDef grad fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#111;
  classDef cfg fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#111;
  classDef state fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#111;
  classDef io fill:#ECEFF1,stroke:#546E7A,stroke-width:2px,color:#111;
  L["Objective L"]:::grad --> A1["Adjoint of final state"]:::grad
  A1 --> A2["Reverse through all PM steps"]:::grad
  A2 --> Th["Gradient with respect to cosmology theta"]:::cfg
  A2 --> Om["Gradient with respect to initial modes omega(k)"]:::cfg
  A2 --> X0["Gradient with respect to initial particle state"]:::state
  L --> ObsP["Gradient with respect to observation parameters"]:::io
```

$$
\frac{\partial L}{\partial \theta}, \qquad
\frac{\partial L}{\partial \omega}, \qquad
\frac{\partial L}{\partial s_0}, \qquad
\frac{\partial L}{\partial \psi_{\mathrm{obs}}}.
$$
