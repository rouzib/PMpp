"""Particles, gravity, symplectic integration, and N-body solvers."""

from .._api import install_lazy_api

_EXPORTS = {
    "Particles": (".particles", "Particles"),
    "_assert_halo_move_succeeded": (".integrator", "_assert_halo_move_succeeded"),
    "_halo_move_vjp": (".integrator", "_halo_move_vjp"),
    "drift": (".integrator", "drift"),
    "drift_adj": (".integrator", "drift_adj"),
    "drift_adj_from_output": (".integrator", "drift_adj_from_output"),
    "drift_for_force": (".integrator", "drift_for_force"),
    "force": (".integrator", "force"),
    "force_acceleration": (".integrator", "force_acceleration"),
    "force_adj": (".integrator", "force_adj"),
    "gravity": (".gravity", "gravity"),
    "integrate": (".integrator", "integrate"),
    "integrate_adj": (".integrator", "integrate_adj"),
    "integrator": (".integrator", None),
    "kick": (".integrator", "kick"),
    "kick_adj": (".integrator", "kick_adj"),
    "nbody": (".solver", "nbody"),
    "nbody_adj": (".solver", "nbody_adj"),
    "nbody_collect": (".solver", "nbody_collect"),
    "nbody_init": (".solver", "nbody_init"),
    "nbody_kappa": (".solver", "nbody_kappa"),
    "nbody_observe": (".solver", "nbody_observe"),
    "nbody_static_halo_scheduled": (".solver", "nbody_static_halo_scheduled"),
    "nbody_step": (".solver", "nbody_step"),
    "neg_grad": (".gravity", "neg_grad"),
    "solver": (".solver", None),
}

install_lazy_api(__name__, _EXPORTS)
