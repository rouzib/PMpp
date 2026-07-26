Advanced and internal interfaces
================================

.. warning::

   These interfaces expose solver internals and optional data adapters.  They
   are documented for extension, profiling, and validation, but are not held
   to the same compatibility promise as the task-oriented API pages.

For the underlying design, read :doc:`../internals/architecture`,
:doc:`../internals/distributed_runtime`, and
:doc:`../internals/integration_and_adjoint`.  The precomputed distributed
example is :doc:`../notebooks/04_multigpu_mesh_halo`.

Runtime construction
--------------------

.. autofunction:: pmpp.multigpu_configuration.build_multigpu_configuration

.. autofunction:: pmpp.multigpu_configuration.initialize_multigpu_runtime

.. autofunction:: pmpp.scatter.initialize_mGPU_scatter

.. autofunction:: pmpp.gather.initialize_mGPU_gather

Integration and adjoint steps
-----------------------------

.. autofunction:: pmpp.nbody.nbody_init

.. autofunction:: pmpp.nbody.nbody_step

.. autofunction:: pmpp.nbody.nbody_adj

.. autofunction:: pmpp.nbody.nbody_static_halo_scheduled

.. autofunction:: pmpp.steps.drift_factor

.. autofunction:: pmpp.steps.kick_factor

.. autofunction:: pmpp.steps.drift

.. autofunction:: pmpp.steps.drift_for_force

.. autofunction:: pmpp.steps.drift_adj

.. autofunction:: pmpp.steps.drift_adj_from_output

.. autofunction:: pmpp.steps.kick

.. autofunction:: pmpp.steps.kick_adj

.. autofunction:: pmpp.steps.force

.. autofunction:: pmpp.steps.force_adj

.. autofunction:: pmpp.steps.integrate

.. autofunction:: pmpp.steps.integrate_adj

Mesh-halo primitives
--------------------

.. autofunction:: pmpp.mesh_halo.owned_mesh_partition_spec

.. autofunction:: pmpp.mesh_halo.maybe_shard_map_mesh_local_op

.. autofunction:: pmpp.mesh_halo.zero_pad_owned_mesh_halo

.. autofunction:: pmpp.mesh_halo.exchange_owned_mesh_halo_edges

.. autofunction:: pmpp.mesh_halo.extend_owned_mesh_from_halo_edges

.. autofunction:: pmpp.mesh_halo.extend_owned_mesh_with_halo

.. autofunction:: pmpp.mesh_halo.reduce_mesh_halo_to_owned

Distributed FFT construction
----------------------------

.. autofunction:: pmpp.fft.fftfreq

.. autofunction:: pmpp.fft.fftfwd

.. autofunction:: pmpp.fft.fftinv

.. autofunction:: pmpp.FFT_distributed.split_array_for_gpus

.. autofunction:: pmpp.FFT_distributed.distribute_array_on_gpus

.. autofunction:: pmpp.FFT_distributed.create_sharded_fft

.. autofunction:: pmpp.FFT_distributed.create_batched_transposed_real_ffts

.. autofunction:: pmpp.FFT_distributed.create_ffts

Particle routing and halo movement
----------------------------------

.. autofunction:: pmpp.halo_moving.particles_in_slice_mask

.. autofunction:: pmpp.halo_moving.compute_halo_mask

.. autofunction:: pmpp.halo_moving.compute_halo_mask_shard_map

.. autofunction:: pmpp.halo_moving.move_particles_canonical_shard_map

.. autofunction:: pmpp.halo_moving.move_particles_mesh_halo_shard_map

.. autofunction:: pmpp.halo_moving.move_particles_mesh_halo_no_acc_shard_map

.. autofunction:: pmpp.halo_moving.reconstruct_pre_drift_canonical_shard_map

.. autofunction:: pmpp.halo_moving.reconstruct_pre_drift_mesh_halo_shard_map

.. autofunction:: pmpp.halo_moving.halo_move_pullback_from_prestate_shard_map

.. autofunction:: pmpp.halo_moving.halo_move_pullback_mesh_halo_from_prestate_shard_map

Utilities
---------

.. autofunction:: pmpp.utils.build_particle_nyquist_filter

.. autofunction:: pmpp.utils.pmid_to_idx

.. autofunction:: pmpp.utils.build_ring_permutations

.. autofunction:: pmpp.utils.get_a_schedule

.. autofunction:: pmpp.ode_util.odeint

Optional CAMELS adapter
-----------------------

.. autoclass:: pmpp.camels_io.CamelsMetadata

.. autoclass:: pmpp.camels_io.CamelsParticlePair

.. autofunction:: pmpp.camels_io.periodic_wrap

.. autofunction:: pmpp.camels_io.periodic_delta

.. autofunction:: pmpp.camels_io.gadget_velocity_to_pmpp

.. autofunction:: pmpp.camels_io.load_camels_pair

.. autofunction:: pmpp.camels_io.coarsen_camels_pair

.. autofunction:: pmpp.camels_io.velocity_kms_to_canonical
