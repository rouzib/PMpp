Configuration and runtime selection
===================================

Use these objects to define a simulation and select its JAX devices.  See
:doc:`../user_guide/configuration` for field semantics,
:doc:`../user_guide/multigpu` for the preferred ``mesh_halo`` setup, and
:doc:`../notebooks/02_configuration` for an executable tour.

Configuration
-------------

.. autoclass:: pmpp.configuration.Configuration

Multi-GPU configuration
-----------------------

.. autoclass:: pmpp.multigpu_configuration.MultiGPUConfiguration

Device mesh construction
------------------------

.. autofunction:: pmpp.utils.create_compute_mesh
