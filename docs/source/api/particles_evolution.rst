Particles and evolution
=======================

The particle-state and forward-solver interfaces below form the main PM++
pipeline.  See :doc:`../user_guide/evolution_and_analysis`,
:doc:`../internals/particle_mesh`, and
:doc:`../internals/integration_and_adjoint` for behavior and design choices.
The complete small run is :doc:`../notebooks/01_first_simulation`; observers
and gradients are demonstrated in :doc:`../notebooks/05_observers_and_analysis`
and :doc:`../notebooks/06_differentiation`.

Particle state and constructors
-------------------------------

.. autoclass:: pmpp.nbody.particles.Particles

.. automethod:: pmpp.nbody.particles.Particles.gen_grid

.. automethod:: pmpp.nbody.particles.Particles.from_pos

.. automethod:: pmpp.nbody.particles.Particles.from_pos_sharded

.. automethod:: pmpp.nbody.particles.Particles.from_ordered_pos

.. automethod:: pmpp.nbody.particles.Particles.from_pmid

.. automethod:: pmpp.nbody.particles.Particles.from_ptcl

.. automethod:: pmpp.nbody.particles.Particles.pos

.. automethod:: pmpp.nbody.particles.Particles.raveled_id

Forward evolution
-----------------

.. autofunction:: pmpp.nbody.solver.nbody

.. autofunction:: pmpp.nbody.solver.nbody_observe

.. autofunction:: pmpp.nbody.solver.nbody_collect

.. autofunction:: pmpp.nbody.observers.density_projection_observer

.. autofunction:: pmpp.nbody.observers.nbody_kappa

Particle--mesh operators
------------------------

.. autofunction:: pmpp.cic.scatter.scatter

.. autofunction:: pmpp.cic.gather.gather

.. autofunction:: pmpp.nbody.gravity.gravity
