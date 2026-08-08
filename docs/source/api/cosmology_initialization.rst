Cosmology and initial conditions
================================

These interfaces build the cosmology tables and initial particle state.  Their
scientific context is covered in :doc:`../user_guide/initial_conditions` and
:doc:`../internals/initial_conditions`.  A worked multi-resolution comparison
is available in :doc:`../notebooks/03_nested_white_noise`.

Cosmology objects
-----------------

.. autoclass:: pmpp.cosmology.models.Cosmology

.. automethod:: pmpp.cosmology.models.Cosmology.from_sigma8

.. automethod:: pmpp.cosmology.models.Cosmology.astype

.. py:data:: pmpp.cosmology.models.SimpleLCDM

   Convenience partial constructor for a flat :class:`pmpp.cosmology.models.Cosmology`
   with representative LCDM defaults.  Pass the active configuration as its
   first argument and override any cosmological parameter by keyword.

Differentiable parameter selection
----------------------------------

.. autofunction:: pmpp.cosmology.models.cosmology_param_names

.. autofunction:: pmpp.cosmology.models.cosmology_param_values

.. autofunction:: pmpp.cosmology.models.replace_cosmology_params

Background expansion
--------------------

.. autofunction:: pmpp.cosmology.models.E2

.. autofunction:: pmpp.cosmology.models.H_deriv

.. autofunction:: pmpp.cosmology.models.Omega_m_a

Transfer, growth, and variance
------------------------------

.. autofunction:: pmpp.cosmology.boltzmann.boltzmann

.. autofunction:: pmpp.cosmology.boltzmann.transfer

.. autofunction:: pmpp.cosmology.growth.growth

.. autofunction:: pmpp.cosmology.boltzmann.linear_power

.. autofunction:: pmpp.cosmology.boltzmann.varlin

Random and linear modes
-----------------------

.. autofunction:: pmpp.initial_conditions.modes.white_noise

.. autofunction:: pmpp.initial_conditions.modes.white_noise_nested

.. autofunction:: pmpp.initial_conditions.modes.linear_modes

Lagrangian perturbation theory
------------------------------

.. autofunction:: pmpp.initial_conditions.lpt.lpt

.. autofunction:: pmpp.initial_conditions.lpt.lpt_low_memory

.. autofunction:: pmpp.initial_conditions.lpt.lpt_low_memory_with_telemetry
