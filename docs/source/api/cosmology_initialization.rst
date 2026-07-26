Cosmology and initial conditions
================================

These interfaces build the cosmology tables and initial particle state.  Their
scientific context is covered in :doc:`../user_guide/initial_conditions` and
:doc:`../internals/initial_conditions`.  A worked multi-resolution comparison
is available in :doc:`../notebooks/03_nested_white_noise`.

Cosmology objects
-----------------

.. autoclass:: pmpp.cosmo.Cosmology

.. automethod:: pmpp.cosmo.Cosmology.from_sigma8

.. automethod:: pmpp.cosmo.Cosmology.astype

.. py:data:: pmpp.cosmo.SimpleLCDM

   Convenience partial constructor for a flat :class:`pmpp.cosmo.Cosmology`
   with representative LCDM defaults.  Pass the active configuration as its
   first argument and override any cosmological parameter by keyword.

Differentiable parameter selection
----------------------------------

.. autofunction:: pmpp.cosmo.cosmology_param_names

.. autofunction:: pmpp.cosmo.cosmology_param_values

.. autofunction:: pmpp.cosmo.replace_cosmology_params

Background expansion
--------------------

.. autofunction:: pmpp.cosmo.E2

.. autofunction:: pmpp.cosmo.H_deriv

.. autofunction:: pmpp.cosmo.Omega_m_a

Transfer, growth, and variance
------------------------------

.. autofunction:: pmpp.boltzmann.boltzmann

.. autofunction:: pmpp.boltzmann.transfer

.. autofunction:: pmpp.growth.growth

.. autofunction:: pmpp.boltzmann.linear_power

.. autofunction:: pmpp.boltzmann.varlin

Random and linear modes
-----------------------

.. autofunction:: pmpp.modes.white_noise

.. autofunction:: pmpp.modes.white_noise_nested

.. autofunction:: pmpp.modes.linear_modes

Lagrangian perturbation theory
------------------------------

.. autofunction:: pmpp.lpt.lpt
