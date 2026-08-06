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

.. autofunction:: pmpp.distributed.configuration.build_multigpu_configuration

.. autofunction:: pmpp.distributed.configuration.initialize_multigpu_runtime

.. autofunction:: pmpp.cic.scatter.initialize_mGPU_scatter

.. autofunction:: pmpp.cic.gather.initialize_mGPU_gather

Integration and adjoint steps
-----------------------------

.. autofunction:: pmpp.nbody.solver.nbody_init

.. autofunction:: pmpp.nbody.solver.nbody_step

.. autofunction:: pmpp.nbody.solver.nbody_adj

.. autofunction:: pmpp.nbody.solver.nbody_static_halo_scheduled

.. autofunction:: pmpp.nbody.integrator.drift_factor

.. autofunction:: pmpp.nbody.integrator.kick_factor

.. autofunction:: pmpp.nbody.integrator.drift

.. autofunction:: pmpp.nbody.integrator.drift_for_force

.. autofunction:: pmpp.nbody.integrator.drift_adj

.. autofunction:: pmpp.nbody.integrator.drift_adj_from_output

.. autofunction:: pmpp.nbody.integrator.kick

.. autofunction:: pmpp.nbody.integrator.kick_adj

.. autofunction:: pmpp.nbody.integrator.force

.. autofunction:: pmpp.nbody.integrator.force_adj

.. autofunction:: pmpp.nbody.integrator.integrate

.. autofunction:: pmpp.nbody.integrator.integrate_adj

Mesh-halo primitives
--------------------

.. autofunction:: pmpp.distributed.mesh_halo.owned_mesh_partition_spec

.. autofunction:: pmpp.distributed.mesh_halo.maybe_shard_map_mesh_local_op

.. autofunction:: pmpp.distributed.mesh_halo.zero_pad_owned_mesh_halo

.. autofunction:: pmpp.distributed.mesh_halo.exchange_owned_mesh_halo_edges

.. autofunction:: pmpp.distributed.mesh_halo.extend_owned_mesh_from_halo_edges

.. autofunction:: pmpp.distributed.mesh_halo.extend_owned_mesh_with_halo

.. autofunction:: pmpp.distributed.mesh_halo.reduce_mesh_halo_to_owned

Distributed FFT construction
----------------------------

.. autofunction:: pmpp.numerics.fft.fftfreq

.. autofunction:: pmpp.numerics.fft.fftfwd

.. autofunction:: pmpp.numerics.fft.fftinv

.. autofunction:: pmpp.distributed.fft.split_array_for_gpus

.. autofunction:: pmpp.distributed.fft.distribute_array_on_gpus

.. autofunction:: pmpp.distributed.fft.create_sharded_fft

.. autofunction:: pmpp.distributed.fft.create_batched_transposed_real_ffts

.. autofunction:: pmpp.distributed.fft.create_ffts

Particle routing and halo movement
----------------------------------

.. autofunction:: pmpp.distributed.routing.particles_in_slice_mask

.. autofunction:: pmpp.distributed.routing.compute_halo_mask

.. autofunction:: pmpp.distributed.routing.compute_halo_mask_shard_map

.. autofunction:: pmpp.distributed.routing.move_particles_canonical_shard_map

.. autofunction:: pmpp.distributed.routing.move_particles_mesh_halo_shard_map

.. autofunction:: pmpp.distributed.routing.move_particles_mesh_halo_no_acc_shard_map

.. autofunction:: pmpp.distributed.routing.reconstruct_pre_drift_canonical_shard_map

.. autofunction:: pmpp.distributed.routing.reconstruct_pre_drift_mesh_halo_shard_map

.. autofunction:: pmpp.distributed.routing.halo_move_pullback_from_prestate_shard_map

.. autofunction:: pmpp.distributed.routing.halo_move_pullback_mesh_halo_from_prestate_shard_map

Utilities
---------

.. autofunction:: pmpp.core.utils.build_particle_nyquist_filter

.. autofunction:: pmpp.core.utils.pmid_to_idx

.. autofunction:: pmpp.core.utils.build_ring_permutations

.. autofunction:: pmpp.core.utils.get_a_schedule

.. autofunction:: pmpp.numerics.ode.odeint

Optional CAMELS adapter
-----------------------

.. autoclass:: pmpp.extras.camels.io.CamelsMetadata

.. autoclass:: pmpp.extras.camels.io.CamelsParticlePair

.. autofunction:: pmpp.extras.camels.io.periodic_wrap

.. autofunction:: pmpp.extras.camels.io.periodic_delta

.. autofunction:: pmpp.extras.camels.io.gadget_velocity_to_pmpp

.. autofunction:: pmpp.extras.camels.io.load_camels_pair

.. autofunction:: pmpp.extras.camels.io.coarsen_camels_pair

.. autofunction:: pmpp.extras.camels.io.velocity_kms_to_canonical
