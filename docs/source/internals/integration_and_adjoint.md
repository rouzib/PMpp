# Integration and discrete adjoint

## Purpose

PM++ advances the particle state with a symplectic splitting and differentiates
the exact discrete update in reverse. This discretize-then-optimize construction
is the memory-efficient adjoint strategy developed for PMWD in
[Li et al. (2024)](https://arxiv.org/abs/2211.09815v2), extended here to PM++'s
distributed movement and force operators.

## Forward and reverse timelines

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;
  A["State at a_n"]:::state --> B["Half kick"]:::op --> C["Drift + ownership"]:::comm --> D["PM force"]:::op --> E["Half kick / state a_n+1"]:::state --> F["Reverse exact operators"]:::check --> G["Adjoint at a_n"]:::state
```

**Text equivalent:** the forward step kicks velocity, drifts positions and
updates ownership, recomputes force, and completes the kick. Starting from a
cotangent at the output, the adjoint applies the transposes of the same discrete
operators in reverse order to obtain the input cotangent.

The forward implementation executes the two half-kicks separately. The
hand-written adjoint groups neighboring kick contributions algebraically when
constructing its reverse schedule, while preserving the same logical
kick-drift-force-kick scheme.

## Equations

For one logical leapfrog interval,

$$
\mathbf v_{n+1/2}=\mathbf v_n+K_n\mathbf a_n,
$$

$$
\mathbf x_{n+1}=\mathbf x_n+D_n\mathbf v_{n+1/2},\qquad
\mathbf a_{n+1}=F(\mathbf x_{n+1};\theta),
$$

$$
\mathbf v_{n+1}=\mathbf v_{n+1/2}+K_{n+1}\mathbf a_{n+1}.
$$

The drift/kick factors are built from growth functions and the configured
scale-factor boundaries rather than a constant physical time step. For
$z_{n+1}=f_n(z_n,\theta)$ and terminal loss $J$, the discrete recurrence is

$$
\lambda_n=\left(\frac{\partial f_n}{\partial z_n}\right)^T\lambda_{n+1},
\qquad \lambda_N=\frac{\partial J}{\partial z_N},
$$

with parameter contributions accumulated from each reversed stage.

## Shapes and units

`disp`, `vel`, and `acc` share the padded particle shape. Drift factors have the
inverse-Hubble scaling needed to map canonical velocity to displacement; kick
factors map the scaled acceleration to canonical velocity. Scale factors and
cosmology tables use `cosmo_dtype`; particle updates are cast to `float_dtype`.

Observers return arbitrary fixed-shape PyTrees stacked over the leading schedule
axis. They are deliberately separate from the custom-adjoint `nbody` path.

## Implementation anchors

`src/pmpp/steps.py` defines drift/kick factors, forward substeps, force, and their
adjoints. `src/pmpp/nbody.py` scans the schedule and wraps the flattened state in
a custom VJP. The forward rule saves the final particle state and options rather
than the full trajectory; `nbody_adj` reconstructs earlier states while sweeping
the adjoints backward. `src/pmpp/nbody_observers.py` contains forward-only
projection/collector helpers.

## Design trade-offs

- Saving only the terminal state avoids a tape proportional to the number of
  time steps, but the backward sweep recomputes force/state information.
- Symplectic reversibility supports reconstruction, while finite precision and
  distributed reordering require the adjoint to follow the implemented discrete
  path exactly.
- `nbody_cosmo_grad=False` skips N-body cosmology cotangents and can reduce work,
  but intentionally changes which gradients are computed.
- Observers are simple and memory-visible; making them forward-only avoids
  silently expanding the custom-adjoint contract.

## Validation

Test each drift, kick, force, and movement VJP before the full recurrence. Use
centered finite differences or directional derivatives on tiny schedules, then
run end-to-end N-body gradient regressions in `mesh_halo`. Confirm that the
paired primal has no capacity error; an adjoint cannot repair truncated forward
state.
