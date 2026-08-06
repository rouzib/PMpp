"""Cosmological models, transfer functions, and growth utilities."""

from .._api import install_lazy_api

_EXPORTS = {
    "Cosmology": (".models", "Cosmology"),
    "E2": (".models", "E2"),
    "SimpleLCDM": (".models", "SimpleLCDM"),
    "boltzmann": (".boltzmann", "boltzmann"),
    "growth": (".growth", "growth"),
    "linear_power": (".boltzmann", "linear_power"),
    "cosmology_param_names": (".models", "cosmology_param_names"),
    "cosmology_param_values": (".models", "cosmology_param_values"),
    "replace_cosmology_params": (".models", "replace_cosmology_params"),
    "transfer": (".boltzmann", "transfer"),
    "varlin": (".boltzmann", "varlin"),
}

install_lazy_api(__name__, _EXPORTS)
