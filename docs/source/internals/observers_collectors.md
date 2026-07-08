# Observers and collectors

`nbody` is the main evolution function. `nbody_observe` and `nbody_collect` are forward-only diagnostic interfaces for saving maps or other observables during a run. Legacy observer paths such as `nbody_kappa` should be documented as compatibility helpers when present.

Observers are useful for density projections, snapshots, and monitoring runs without rewriting the integrator. Keep them separate from adjoint workflows unless a specific observer is documented as differentiable.
