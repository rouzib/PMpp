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

.. autoclass:: pmpp.particles.Particles

.. automethod:: pmpp.particles.Particles.gen_grid

.. automethod:: pmpp.particles.Particles.from_pos

.. automethod:: pmpp.particles.Particles.from_pos_sharded

.. automethod:: pmpp.particles.Particles.from_ordered_pos

.. automethod:: pmpp.particles.Particles.from_pmid

.. automethod:: pmpp.particles.Particles.from_ptcl

.. automethod:: pmpp.particles.Particles.pos

.. automethod:: pmpp.particles.Particles.raveled_id

Forward evolution
-----------------

.. autofunction:: pmpp.nbody.nbody

.. autofunction:: pmpp.nbody.nbody_observe

.. autofunction:: pmpp.nbody.nbody_collect

.. autofunction:: pmpp.nbody_observers.density_projection_observer

Particle--mesh operators
------------------------

.. autofunction:: pmpp.scatter.scatter

.. autofunction:: pmpp.gather.gather

.. autofunction:: pmpp.gravity.gravity
