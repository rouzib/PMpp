Analysis and plotting
=====================

These helpers turn density meshes and particle states into differentiable
summary statistics or diagnostic figures.  See
:doc:`../user_guide/evolution_and_analysis` and the pre-executed
:doc:`../notebooks/05_observers_and_analysis`.

Power spectra
-------------

.. autofunction:: pmpp.power_spectrum.delta_to_pk

.. autofunction:: pmpp.power_spectrum.density_to_pk

.. autofunction:: pmpp.power_spectrum.particles_to_pk

Cross spectra and correlation
-----------------------------

.. autofunction:: pmpp.power_spectrum.delta_to_cross_correlation

.. autofunction:: pmpp.power_spectrum.density_to_cross_correlation

.. autofunction:: pmpp.power_spectrum.particles_to_cross_correlation

.. autofunction:: pmpp.power_spectrum.cross_correlation

Diagnostic plotting
-------------------

.. autofunction:: pmpp.plotting_utils.plot_particle_distribution_on_gpus

.. autofunction:: pmpp.plotting_utils.plot_pos_distribution

.. autofunction:: pmpp.plotting_utils.plot_particle_bins
