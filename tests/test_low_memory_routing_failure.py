"""Failure propagation for the forward-only low-memory N-body solver."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from pmpp.nbody import integrator, solver
from pmpp.core.utils import AXIS_NAME


def test_low_memory_uniform_grid_completes_without_failure():
    from pmpp.core import Configuration
    from pmpp.cosmology import SimpleLCDM, boltzmann
    from pmpp.nbody import Particles

    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1,
                         a_start=0.1, a_stop=0.2, a_nbody_maxstep=0.1)
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    particles = Particles.gen_grid(conf, vel=True, acc=True)
    compiled = solver.lower_nbody_low_memory(particles, cosmo, conf).compile()
    result, occupancy, moved, invalid = solver.nbody_low_memory_with_telemetry(
        particles, cosmo, conf, compiled=compiled,
    )
    assert int(occupancy) == 64
    assert int(moved) == 0
    assert int(invalid) == 0
    assert np.max(np.abs(np.asarray(result.disp))) < 1e-4


@pytest.mark.parametrize("route_failed", [False, True])
def test_failed_drift_skips_force_and_kick(monkeypatch, route_failed):
    conf = SimpleNamespace(symp_splits=((1, 1),))

    def route(_a_vel, _a_prev, _a_next, state, _cosmo, _conf):
        return state + 1, jnp.int32(7), jnp.int32(3), jnp.bool_(route_failed)

    monkeypatch.setattr(integrator, "_drift_for_force_low_memory_status", route)
    monkeypatch.setattr(integrator, "force", lambda _a, state, _c, _conf, **_kw: state + 10)
    monkeypatch.setattr(integrator, "kick", lambda _a, _b, _c, state, _co, _conf: state + 100)
    compiled = jax.jit(lambda state: integrator.integrate_low_memory(
        jnp.float32(0.1), jnp.float32(0.2), state, None, conf,
    ))
    state, moved, invalid, failed = jax.device_get(compiled(jnp.int32(0)))
    assert int(state) == (1 if route_failed else 111)
    assert int(moved) == 7
    assert int(invalid) == 3
    assert bool(failed) is route_failed


def test_one_shard_failure_skips_collective_force_on_all_devices(monkeypatch):
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    conf = SimpleNamespace(symp_splits=((1, 1),))

    global_failure = shard_map(
        lambda local: lax.pmax(jnp.any(local == 2).astype(jnp.int32), AXIS_NAME) != 0,
        mesh=mesh, in_specs=P(AXIS_NAME), out_specs=P(), check_vma=False,
    )

    def route(_a_vel, _a_prev, _a_next, state, _cosmo, _conf):
        return state + 1, jnp.int32(7), jnp.int32(3), global_failure(state)

    def forbidden_force(_a, state, _cosmo, _conf, **_kw):
        jax.debug.callback(lambda: (_ for _ in ()).throw(AssertionError("force ran after failure")))
        return state + 10

    monkeypatch.setattr(integrator, "_drift_for_force_low_memory_status", route)
    monkeypatch.setattr(integrator, "force", forbidden_force)
    monkeypatch.setattr(integrator, "kick", lambda _a, _b, _c, state, _co, _conf: state + 100)
    state = jax.device_put(jnp.arange(4, dtype=jnp.int32), NamedSharding(mesh, P(AXIS_NAME)))
    compiled = jax.jit(lambda value: integrator.integrate_low_memory(
        jnp.float32(0.1), jnp.float32(0.2), value, None, conf,
    ))
    result, _, _, failed = compiled(state)
    np.testing.assert_array_equal(np.asarray(result), np.arange(4) + 1)
    assert bool(failed)


def test_failed_scan_skips_remaining_steps(monkeypatch):
    monkeypatch.setattr(solver, "_nbody_scale_factors", lambda _conf, _reverse: jnp.arange(4, dtype=jnp.float32))
    monkeypatch.setattr(solver, "_max_authoritative_occupancy", lambda state, _conf: state)
    monkeypatch.setattr(solver, "force", lambda _a, state, _c, _conf, **_kw: state + 10)

    def integrate(_a_prev, a_next, state, _cosmo, _conf):
        return state + 1, jnp.int32(7), jnp.int32(3), a_next == jnp.float32(2)

    monkeypatch.setattr(solver, "integrate_low_memory", integrate)
    compiled = jax.jit(lambda state: solver._nbody_low_memory_impl(state, None, None))
    state, occupancy, moved, invalid, failed, failed_step = jax.device_get(compiled(jnp.int32(0)))
    assert (int(state), int(occupancy), int(moved), int(invalid)) == (12, 11, 7, 3)
    assert bool(failed)
    assert int(failed_step) == 1


def test_host_reports_failure_after_compiled_status_returns(monkeypatch):
    monkeypatch.setattr(solver, "_validate_low_memory_nbody", lambda *_args: None)
    monkeypatch.setattr(solver, "_cosmo_state", lambda _cosmo: None)
    monkeypatch.setattr(solver, "_nbody_low_memory_state", lambda *_args: (
        None, jnp.int32(28), jnp.int32(74541), jnp.int32(4), jnp.bool_(True), jnp.int32(6),
    ))
    particle = SimpleNamespace(pmid=None, unused_index=None, halo_mask=None,
                               attr=None, disp=None, vel=None, acc=None)
    conf = SimpleNamespace(max_ptcl_per_slice=100, max_share_ptcl=50,
                           multigpu=SimpleNamespace(far_send_capacity=20, far_recv_capacity=30))
    with pytest.raises(solver.ParticleRoutingFailure, match="synchronized N-body step 7") as exc:
        solver.nbody_low_memory_with_telemetry(particle, None, conf)
    assert "max_particles_moved=74541" in str(exc.value)
    assert "far_recv_capacity=30" in str(exc.value)
