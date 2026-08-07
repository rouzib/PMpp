import jax
import jax.numpy as jnp
import numpy as np

from pmpp.core import Configuration
from pmpp.cosmology import SimpleLCDM, boltzmann
from pmpp.nbody import Particles
from pmpp.nbody.observers import density_projection_observer, nbody_kappa
import pmpp.nbody.observers as observers_module
import pmpp.nbody.solver as solver_module


def _short_conf(**changes):
    kwargs = dict(
        ptcl_spacing=1.0, ptcl_grid_shape=(2, 2, 2), mesh_shape=1, pallas_cic=False, a_start=0.5, a_stop=0.51,
        a_custom=(0.5, 0.51),
    )
    kwargs.update(changes)
    return Configuration(**kwargs)


def _cosmo(conf):
    return boltzmann(SimpleLCDM(conf), conf)


def test_density_projection_observer_has_exact_mass_and_normalization_on_uniform_lattice():
    conf = _short_conf()
    ptcl = Particles.gen_grid(conf, vel=True)
    cosmo = _cosmo(conf)
    for axis in range(3):
        raw = density_projection_observer(axis)(0.5, ptcl, cosmo, conf)
        normalized = density_projection_observer(axis, normalize=True)(0.5, ptcl, cosmo, conf)
        assert raw.shape == tuple(size for index, size in enumerate(conf.mesh_shape) if index != axis)
        np.testing.assert_allclose(np.asarray(raw), 2.0, atol=2e-6)
        np.testing.assert_allclose(np.asarray(normalized), 1.0, atol=2e-6)
        np.testing.assert_allclose(float(raw.sum()), conf.mesh_size, atol=2e-6)


def test_nbody_observe_records_start_and_end_pytrees_and_returns_same_final_state_as_nbody():
    conf = _short_conf()
    ptcl = Particles.gen_grid(conf, vel=True)
    cosmo = _cosmo(conf)

    def observer(a, particle, _cosmo, _conf):
        return {
            "a": a,
            "mean_disp": jnp.mean(particle.disp, axis=0),
            "mass": jnp.sum(~particle.unused_index).astype(_conf.float_dtype),
        }

    final, observations = solver_module.nbody_observe(
        ptcl, cosmo, conf, observer, include_start=True, return_final=True,
    )
    direct = solver_module.nbody(ptcl, cosmo, conf)
    np.testing.assert_allclose(np.asarray(observations["a"]), [0.5, 0.51], rtol=1e-7)
    np.testing.assert_allclose(np.asarray(observations["mass"]), conf.ptcl_num)
    np.testing.assert_allclose(np.asarray(observations["mean_disp"]), 0, atol=2e-6)
    np.testing.assert_allclose(np.asarray(final.disp), np.asarray(direct.disp), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(final.vel), np.asarray(direct.vel), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(final.acc), np.asarray(direct.acc), rtol=2e-6, atol=2e-6)

    reverse_observations = solver_module.nbody_observe(ptcl, cosmo, conf, observer, reverse=True)
    np.testing.assert_allclose(np.asarray(reverse_observations["a"]), [0.5], rtol=1e-7)


def test_nbody_collect_updates_arbitrary_state_and_optional_final_particle():
    conf = _short_conf()
    ptcl = Particles.gen_grid(conf, vel=True)
    cosmo = _cosmo(conf)

    def collector(state, a_prev, a_next, particle, _cosmo, _conf):
        increment = jnp.asarray([a_prev, a_next,
                                 jnp.sum(jnp.square(particle.disp)),
                                 jnp.sum(~particle.unused_index), ], dtype=_conf.float_dtype)
        return state + increment

    initial = jnp.zeros((4, ), dtype=conf.float_dtype)
    final, state = solver_module.nbody_collect(ptcl, cosmo, conf, collector, initial, return_final=True)
    state_only = solver_module.nbody_collect(ptcl, cosmo, conf, collector, initial, return_final=False)
    np.testing.assert_allclose(np.asarray(state), [0.5, 0.51, 0.0, conf.ptcl_num], atol=2e-6)
    np.testing.assert_allclose(np.asarray(state_only), np.asarray(state), atol=2e-6)
    assert isinstance(final, Particles)
    assert final.acc is not None


def test_nbody_kappa_saved_maps_have_exact_shape_mass_and_periodic_slice_content():
    conf = _short_conf(to_save_a=[0.51], slice_to_save=[0, 2], max_slice_width=2)
    ptcl = Particles.gen_grid(conf, vel=True)
    cosmo = _cosmo(conf)
    maps = nbody_kappa(ptcl, cosmo, conf)
    assert maps.shape == (1, 3, 2, 2)
    np.testing.assert_allclose(np.asarray(maps), 2.0, atol=3e-6)
    for axis in range(3):
        np.testing.assert_allclose(float(maps[0, axis].sum()), conf.mesh_size, atol=3e-6)

    no_save_conf = _short_conf(to_save_a=None)
    no_save_result = nbody_kappa(Particles.gen_grid(no_save_conf, vel=True), _cosmo(no_save_conf), no_save_conf)
    assert isinstance(no_save_result, Particles)


def test_segment_order_and_legacy_wrapper_dispatch_are_explicit(monkeypatch):
    calls = []

    def fake_nbody(state, cosmo, conf, reverse=False, correction=None):
        del cosmo, correction
        calls.append((conf, reverse))
        return state * 10 + conf

    monkeypatch.setattr(solver_module, "nbody", fake_nbody)
    assert solver_module.nbody_static_halo_scheduled(0, object(), [1, 2, 3]) == 123
    assert calls == [(1, False), (2, False), (3, False)]
    calls.clear()
    assert solver_module.nbody_static_halo_scheduled(0, object(), [1, 2, 3], reverse=True) == 321
    assert calls == [(3, True), (2, True), (1, True)]

    sentinel = object()
    monkeypatch.setattr(observers_module, "nbody_kappa", lambda ptcl, cosmo, conf, reverse=False: sentinel)
    assert solver_module.nbody_kappa(object(), object(), object(), reverse=True) is sentinel
