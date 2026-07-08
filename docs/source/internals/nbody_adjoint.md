# N-body adjoint

Naive reverse-mode differentiation through every time step of a long N-body run can be memory-prohibitive. PM++ therefore uses a hand-written adjoint for selected N-body workflows.

The forward pass advances particles through drift, force, and kick stages. The backward sweep propagates cotangents through the corresponding drift adjoint, kick adjoint, force adjoint, and halo-movement transpose when multi-GPU particle migration is active. Depending on configuration, states may be recomputed or checkpointed to balance memory and time.

Cosmology cotangents are controlled by the gradient-facing API and settings such as `nbody_cosmo_grad`. Users should start gradient experiments with tiny meshes and finite-value checks before attempting production resolutions.
