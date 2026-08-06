Internal interfaces
===================

These interfaces expose implementation details, experimental backends, and
validation utilities. They are included for development and extension work and
may change without the compatibility guarantees of the task-oriented API.

Derived configuration properties
--------------------------------

.. autoattribute:: pmpp.core.configuration.Configuration.use_mGPU

.. autoattribute:: pmpp.core.configuration.Configuration.dim

.. autoattribute:: pmpp.core.configuration.Configuration.ptcl_cell_vol

.. autoattribute:: pmpp.core.configuration.Configuration.ptcl_num

.. autoattribute:: pmpp.core.configuration.Configuration.box_size

.. autoattribute:: pmpp.core.configuration.Configuration.box_vol

.. autoattribute:: pmpp.core.configuration.Configuration.cell_size

.. autoattribute:: pmpp.core.configuration.Configuration.disp_size

.. autoattribute:: pmpp.core.configuration.Configuration.mesh_size

.. autoattribute:: pmpp.core.configuration.Configuration.local_mesh_size

.. autoattribute:: pmpp.core.configuration.Configuration.V

.. autoattribute:: pmpp.core.configuration.Configuration.H_0

.. autoattribute:: pmpp.core.configuration.Configuration.c

.. autoattribute:: pmpp.core.configuration.Configuration.G

.. autoattribute:: pmpp.core.configuration.Configuration.rho_crit

.. autoattribute:: pmpp.core.configuration.Configuration.transfer_k_num

.. autoattribute:: pmpp.core.configuration.Configuration.transfer_lgk_step

.. autoattribute:: pmpp.core.configuration.Configuration.transfer_k

.. autoattribute:: pmpp.core.configuration.Configuration.a_lpt_num

.. autoattribute:: pmpp.core.configuration.Configuration.a_lpt_step

.. autoattribute:: pmpp.core.configuration.Configuration.a_nbody_num

.. autoattribute:: pmpp.core.configuration.Configuration.a_nbody_step

.. autoattribute:: pmpp.core.configuration.Configuration.a_lpt

.. autoattribute:: pmpp.core.configuration.Configuration.a_nbody

.. autoattribute:: pmpp.core.configuration.Configuration.growth_a

.. autoattribute:: pmpp.core.configuration.Configuration.var_tophat

.. autoattribute:: pmpp.core.configuration.Configuration.varlin_R

Cosmology properties and cotangents
------------------------------------

.. autoattribute:: pmpp.cosmology.models.Cosmology.k_pivot

.. autoattribute:: pmpp.cosmology.models.Cosmology.A_s

.. autoattribute:: pmpp.cosmology.models.Cosmology.Omega_c

.. autoattribute:: pmpp.cosmology.models.Cosmology.Omega_k

.. autoattribute:: pmpp.cosmology.models.Cosmology.Omega_de

.. autoattribute:: pmpp.cosmology.models.Cosmology.w_0

.. autoattribute:: pmpp.cosmology.models.Cosmology.w_a

.. autoattribute:: pmpp.cosmology.models.Cosmology.sigma8

.. autoattribute:: pmpp.cosmology.models.Cosmology.ptcl_mass

.. autofunction:: pmpp.cosmology.models.zero_cosmology_param_cotangent

.. autofunction:: pmpp.cosmology.models.cosmology_param_cotangent

.. autofunction:: pmpp.cosmology.models.project_cosmology_param_cotangent

.. autofunction:: pmpp.cosmology.models.add_cosmology_cotangents

.. autofunction:: pmpp.cosmology.models.sub_cosmology_cotangents

.. autofunction:: pmpp.cosmology.models.scale_cosmology_cotangent

Transfer and mode construction
------------------------------

.. autofunction:: pmpp.cosmology.boltzmann.transfer_integ

.. autofunction:: pmpp.cosmology.boltzmann.transfer_fit

.. autofunction:: pmpp.cosmology.boltzmann.growth_integ

.. autofunction:: pmpp.cosmology.boltzmann.growth

.. autofunction:: pmpp.cosmology.boltzmann.varlin_integ

.. autofunction:: pmpp.initial_conditions.modes.get_k_magnitude

.. autofunction:: pmpp.initial_conditions.modes.get_k_magnitude_transposed

Particle-state helpers
----------------------

.. automethod:: pmpp.nbody.particles.Particles.particles_in_slice_mask

.. automethod:: pmpp.nbody.particles.Particles.compute_halo_mask

.. automethod:: pmpp.nbody.particles.Particles.distribute_ptcl_pos

.. automethod:: pmpp.nbody.particles.Particles.pmid_to_pos

.. automethod:: pmpp.nbody.particles.Particles.pos_to_pmid

.. automethod:: pmpp.nbody.particles.Particles.values_on_device

.. automethod:: pmpp.nbody.particles.Particles.remove_particles

.. automethod:: pmpp.nbody.particles.Particles.add_particles

.. autofunction:: pmpp.cic.enmesh.enmesh

Gravity and particle--mesh primitives
-------------------------------------

.. autofunction:: pmpp.nbody.gravity.get_k_squared

.. autofunction:: pmpp.nbody.gravity.get_k_squared_transposed

.. autofunction:: pmpp.nbody.gravity.get_discrete_k_squared_transposed

.. autofunction:: pmpp.nbody.gravity.apply_particle_nyquist_filter

.. autofunction:: pmpp.nbody.gravity.laplace

.. autofunction:: pmpp.nbody.gravity.laplace_fwd

.. autofunction:: pmpp.nbody.gravity.laplace_bwd

.. autofunction:: pmpp.nbody.gravity.laplace_transposed

.. autofunction:: pmpp.nbody.gravity.laplace_transposed_fwd

.. autofunction:: pmpp.nbody.gravity.laplace_transposed_bwd

.. autofunction:: pmpp.nbody.gravity.laplace_transposed_with_kernel

.. autofunction:: pmpp.nbody.gravity.neg_grad

.. autofunction:: pmpp.nbody.gravity.reduce_duplicate_slot_cot

.. autofunction:: pmpp.nbody.gravity.duplicate_slot_counts

.. autofunction:: pmpp.cic.gather.gather_stacked_mesh_halo

.. autofunction:: pmpp.cic.scatter.reduce_grad_across_gpus

Integration and adjoint internals
---------------------------------

.. autofunction:: pmpp.nbody.solver.nbody_kappa

.. autofunction:: pmpp.nbody.solver.nbody_adjoint_fwd

.. autofunction:: pmpp.nbody.solver.nbody_adjoint_bwd

.. autofunction:: pmpp.nbody.integrator.partition_duplicate_slot_cot

.. autofunction:: pmpp.nbody.integrator.duplicate_partitioned_slot_cot

.. autofunction:: pmpp.nbody.integrator.force_acceleration

Halo runtime initializers
-------------------------

.. autofunction:: pmpp.distributed.routing.reconstruct_pre_drift_and_pullback_mesh_halo_shard_map

.. autofunction:: pmpp.distributed.routing.reconstruct_pre_drift_and_pullback_canonical_shard_map

.. autofunction:: pmpp.distributed.routing.initialize_mGPU_halo_movement_canonical

.. autofunction:: pmpp.distributed.routing.initialize_mGPU_halo_movement_no_acc

.. autofunction:: pmpp.distributed.routing.initialize_mGPU_reconstruct_pre_drift

.. autofunction:: pmpp.distributed.routing.initialize_mGPU_reconstruct_pre_drift_pullback

.. autofunction:: pmpp.distributed.routing.initialize_mGPU_halo_move_pullback

.. autofunction:: pmpp.distributed.routing.initialize_mGPU_compute_halo_mask

CUDA routing
------------

.. autofunction:: pmpp.distributed.cuda.extension_status

.. autofunction:: pmpp.distributed.cuda.supported_configuration

.. autofunction:: pmpp.distributed.cuda.supported_bidir_configuration

.. autofunction:: pmpp.distributed.cuda.requested_backend

.. autofunction:: pmpp.distributed.cuda.enabled_for_configuration

.. autofunction:: pmpp.distributed.cuda.route_pack

.. autofunction:: pmpp.distributed.cuda.route_pack_bidir_cuda

.. autofunction:: pmpp.distributed.cuda.route_merge

.. autofunction:: pmpp.distributed.cuda.route_merge_bidir_cuda

.. autofunction:: pmpp.distributed.cuda.route_transpose_split

.. autofunction:: pmpp.distributed.cuda.route_transpose_scatter

CUDA extension build and discovery
----------------------------------

.. autofunction:: pmpp.distributed.build_cuda.detect_cuda_architectures

.. autofunction:: pmpp.distributed.build_cuda.main

.. autofunction:: pmpp.distributed._cuda_paths.package_cuda_directory

.. autofunction:: pmpp.distributed._cuda_paths.user_cache_cuda_directory

Pallas CIC kernels
------------------

.. autofunction:: pmpp.cic.pallas.pallas_available

.. autofunction:: pmpp.cic.pallas.pallas_cic_supported

.. autofunction:: pmpp.cic.pallas.pallas_gather

.. autofunction:: pmpp.cic.pallas.pallas_scatter

.. autofunction:: pmpp.cic.pallas.pallas_gather_bwd

.. autofunction:: pmpp.cic.pallas.pallas_scatter_bwd

ODE implementation
------------------

.. autofunction:: pmpp.numerics.ode.ravel_first_arg

.. autofunction:: pmpp.numerics.ode.ravel_first_arg_

.. autofunction:: pmpp.numerics.ode.interp_fit_dopri

.. autofunction:: pmpp.numerics.ode.fit_4th_order_polynomial

.. autofunction:: pmpp.numerics.ode.initial_step_size

.. autofunction:: pmpp.numerics.ode.runge_kutta_step

.. autofunction:: pmpp.numerics.ode.abs2

.. autofunction:: pmpp.numerics.ode.mean_error_ratio

.. autofunction:: pmpp.numerics.ode.optimal_step_size

General utilities
-----------------

.. autofunction:: pmpp.core.utils.pytree_dataclass

.. autofunction:: pmpp.core.utils.raise_error

.. autofunction:: pmpp.core.utils.distribute_array_on_gpus

.. autofunction:: pmpp.core.utils.is_float0_array

.. autofunction:: pmpp.core.utils.measure_execution_time

.. autofunction:: pmpp.core.utils.wraparound_slice

.. autofunction:: pmpp.distributed.fft.distribute_array_on_gpus_old

.. autofunction:: pmpp.distributed.fft.test_functions

.. autofunction:: pmpp.analysis.plotting.plot_particle_bins_callback

.. autofunction:: pmpp.analysis.plotting.resolve_title

Package API machinery
---------------------

.. autoclass:: pmpp._api._LazyAPIModule

.. autofunction:: pmpp._api.install_lazy_api
