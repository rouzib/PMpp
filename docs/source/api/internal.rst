Internal interfaces
===================

These interfaces expose implementation details, experimental backends, and
validation utilities. They are included for development and extension work and
may change without the compatibility guarantees of the task-oriented API.

Derived configuration properties
--------------------------------

.. autoattribute:: pmpp.configuration.Configuration.use_mGPU

.. autoattribute:: pmpp.configuration.Configuration.dim

.. autoattribute:: pmpp.configuration.Configuration.ptcl_cell_vol

.. autoattribute:: pmpp.configuration.Configuration.ptcl_num

.. autoattribute:: pmpp.configuration.Configuration.box_size

.. autoattribute:: pmpp.configuration.Configuration.box_vol

.. autoattribute:: pmpp.configuration.Configuration.cell_size

.. autoattribute:: pmpp.configuration.Configuration.disp_size

.. autoattribute:: pmpp.configuration.Configuration.mesh_size

.. autoattribute:: pmpp.configuration.Configuration.local_mesh_size

.. autoattribute:: pmpp.configuration.Configuration.V

.. autoattribute:: pmpp.configuration.Configuration.H_0

.. autoattribute:: pmpp.configuration.Configuration.c

.. autoattribute:: pmpp.configuration.Configuration.G

.. autoattribute:: pmpp.configuration.Configuration.rho_crit

.. autoattribute:: pmpp.configuration.Configuration.transfer_k_num

.. autoattribute:: pmpp.configuration.Configuration.transfer_lgk_step

.. autoattribute:: pmpp.configuration.Configuration.transfer_k

.. autoattribute:: pmpp.configuration.Configuration.a_lpt_num

.. autoattribute:: pmpp.configuration.Configuration.a_lpt_step

.. autoattribute:: pmpp.configuration.Configuration.a_nbody_num

.. autoattribute:: pmpp.configuration.Configuration.a_nbody_step

.. autoattribute:: pmpp.configuration.Configuration.a_lpt

.. autoattribute:: pmpp.configuration.Configuration.a_nbody

.. autoattribute:: pmpp.configuration.Configuration.growth_a

.. autoattribute:: pmpp.configuration.Configuration.var_tophat

.. autoattribute:: pmpp.configuration.Configuration.varlin_R

Cosmology properties and cotangents
------------------------------------

.. autoattribute:: pmpp.cosmo.Cosmology.k_pivot

.. autoattribute:: pmpp.cosmo.Cosmology.A_s

.. autoattribute:: pmpp.cosmo.Cosmology.Omega_c

.. autoattribute:: pmpp.cosmo.Cosmology.Omega_k

.. autoattribute:: pmpp.cosmo.Cosmology.Omega_de

.. autoattribute:: pmpp.cosmo.Cosmology.w_0

.. autoattribute:: pmpp.cosmo.Cosmology.w_a

.. autoattribute:: pmpp.cosmo.Cosmology.sigma8

.. autoattribute:: pmpp.cosmo.Cosmology.ptcl_mass

.. autofunction:: pmpp.cosmo.zero_cosmology_param_cotangent

.. autofunction:: pmpp.cosmo.cosmology_param_cotangent

.. autofunction:: pmpp.cosmo.project_cosmology_param_cotangent

.. autofunction:: pmpp.cosmo.add_cosmology_cotangents

.. autofunction:: pmpp.cosmo.sub_cosmology_cotangents

.. autofunction:: pmpp.cosmo.scale_cosmology_cotangent

Transfer and mode construction
------------------------------

.. autofunction:: pmpp.boltzmann.transfer_integ

.. autofunction:: pmpp.boltzmann.transfer_fit

.. autofunction:: pmpp.boltzmann.growth_integ

.. autofunction:: pmpp.boltzmann.growth

.. autofunction:: pmpp.boltzmann.varlin_integ

.. autofunction:: pmpp.modes.get_k_magnitude

.. autofunction:: pmpp.modes.get_k_magnitude_transposed

Particle-state helpers
----------------------

.. automethod:: pmpp.particles.Particles.particles_in_slice_mask

.. automethod:: pmpp.particles.Particles.compute_halo_mask

.. automethod:: pmpp.particles.Particles.distribute_ptcl_pos

.. automethod:: pmpp.particles.Particles.pmid_to_pos

.. automethod:: pmpp.particles.Particles.pos_to_pmid

.. automethod:: pmpp.particles.Particles.values_on_device

.. automethod:: pmpp.particles.Particles.remove_particles

.. automethod:: pmpp.particles.Particles.add_particles

.. autofunction:: pmpp.enmesh.enmesh

Gravity and particle--mesh primitives
-------------------------------------

.. autofunction:: pmpp.gravity.get_k_squared

.. autofunction:: pmpp.gravity.get_k_squared_transposed

.. autofunction:: pmpp.gravity.get_discrete_k_squared_transposed

.. autofunction:: pmpp.gravity.apply_particle_nyquist_filter

.. autofunction:: pmpp.gravity.laplace

.. autofunction:: pmpp.gravity.laplace_fwd

.. autofunction:: pmpp.gravity.laplace_bwd

.. autofunction:: pmpp.gravity.laplace_transposed

.. autofunction:: pmpp.gravity.laplace_transposed_fwd

.. autofunction:: pmpp.gravity.laplace_transposed_bwd

.. autofunction:: pmpp.gravity.laplace_transposed_with_kernel

.. autofunction:: pmpp.gravity.neg_grad

.. autofunction:: pmpp.gravity.reduce_duplicate_slot_cot

.. autofunction:: pmpp.gravity.duplicate_slot_counts

.. autofunction:: pmpp.gather.gather_stacked_mesh_halo

.. autofunction:: pmpp.scatter.reduce_grad_across_gpus

Integration and adjoint internals
---------------------------------

.. autofunction:: pmpp.nbody.nbody_kappa

.. autofunction:: pmpp.nbody.nbody_adjoint_fwd

.. autofunction:: pmpp.nbody.nbody_adjoint_bwd

.. autofunction:: pmpp.steps.partition_duplicate_slot_cot

.. autofunction:: pmpp.steps.duplicate_partitioned_slot_cot

.. autofunction:: pmpp.steps.force_acceleration

Halo runtime initializers
-------------------------

.. autofunction:: pmpp.halo_moving.reconstruct_pre_drift_and_pullback_mesh_halo_shard_map

.. autofunction:: pmpp.halo_moving.reconstruct_pre_drift_and_pullback_canonical_shard_map

.. autofunction:: pmpp.halo_moving.initialize_mGPU_halo_movement_canonical

.. autofunction:: pmpp.halo_moving.initialize_mGPU_halo_movement_no_acc

.. autofunction:: pmpp.halo_moving.initialize_mGPU_reconstruct_pre_drift

.. autofunction:: pmpp.halo_moving.initialize_mGPU_reconstruct_pre_drift_pullback

.. autofunction:: pmpp.halo_moving.initialize_mGPU_halo_move_pullback

.. autofunction:: pmpp.halo_moving.initialize_mGPU_compute_halo_mask

CUDA routing
------------

.. autofunction:: pmpp.cuda_routing.extension_status

.. autofunction:: pmpp.cuda_routing.supported_configuration

.. autofunction:: pmpp.cuda_routing.supported_bidir_configuration

.. autofunction:: pmpp.cuda_routing.requested_backend

.. autofunction:: pmpp.cuda_routing.enabled_for_configuration

.. autofunction:: pmpp.cuda_routing.route_pack

.. autofunction:: pmpp.cuda_routing.route_pack_bidir_cuda

.. autofunction:: pmpp.cuda_routing.route_merge

.. autofunction:: pmpp.cuda_routing.route_merge_bidir_cuda

.. autofunction:: pmpp.cuda_routing.route_transpose_split

.. autofunction:: pmpp.cuda_routing.route_transpose_scatter

Pallas CIC kernels
------------------

.. autofunction:: pmpp.pallas_cic.pallas_available

.. autofunction:: pmpp.pallas_cic.pallas_cic_supported

.. autofunction:: pmpp.pallas_cic.pallas_gather

.. autofunction:: pmpp.pallas_cic.pallas_scatter

.. autofunction:: pmpp.pallas_cic.pallas_gather_bwd

.. autofunction:: pmpp.pallas_cic.pallas_scatter_bwd

ODE implementation
------------------

.. autofunction:: pmpp.ode_util.ravel_first_arg

.. autofunction:: pmpp.ode_util.ravel_first_arg_

.. autofunction:: pmpp.ode_util.interp_fit_dopri

.. autofunction:: pmpp.ode_util.fit_4th_order_polynomial

.. autofunction:: pmpp.ode_util.initial_step_size

.. autofunction:: pmpp.ode_util.runge_kutta_step

.. autofunction:: pmpp.ode_util.abs2

.. autofunction:: pmpp.ode_util.mean_error_ratio

.. autofunction:: pmpp.ode_util.optimal_step_size

General utilities
-----------------

.. autofunction:: pmpp.utils.pytree_dataclass

.. autofunction:: pmpp.utils.raise_error

.. autofunction:: pmpp.utils.distribute_array_on_gpus

.. autofunction:: pmpp.utils.is_float0_array

.. autofunction:: pmpp.utils.measure_execution_time

.. autofunction:: pmpp.utils.wraparound_slice

.. autofunction:: pmpp.FFT_distributed.distribute_array_on_gpus_old

.. autofunction:: pmpp.FFT_distributed.test_functions

.. autofunction:: pmpp.plotting_utils.plot_particle_bins_callback

.. autofunction:: pmpp.plotting_utils.resolve_title

QUIJOTE data and validation
---------------------------

.. autoclass:: pmpp.quijote_io.QuijoteCanonicalization
   :members:

.. autofunction:: pmpp.quijote_io.build_quijote_canonicalization

.. autofunction:: pmpp.quijote_io.canonicalize_quijote_arrays

.. autoclass:: pmpp.quijote_metrics.AcceptanceSummary
   :members:

.. autoclass:: pmpp.quijote_metrics.DenseParticleState
   :members:

.. autoclass:: pmpp.quijote_metrics.FixedGridSpectra
   :members:

.. autoclass:: pmpp.quijote_metrics.PeriodicPositionMetrics
   :members:

.. autofunction:: pmpp.quijote_metrics.authoritative_particle_mask

.. autofunction:: pmpp.quijote_metrics.fixed_grid_spectra

.. autofunction:: pmpp.quijote_metrics.gather_authoritative_particles

.. autofunction:: pmpp.quijote_metrics.interlaced_cic_density

.. autofunction:: pmpp.quijote_metrics.periodic_position_metrics

.. autofunction:: pmpp.quijote_metrics.pmid_to_lagrangian_idx

.. autofunction:: pmpp.quijote_metrics.summarize_acceptance

.. autofunction:: pmpp.quijote_metrics.validate_lagrangian_bijection
