"""Four-GPU native hybrid routing acceptance tests.

Set PMPP_REQUIRE_HYBRID_NATIVE=1 on the target H100 node so a missing or stale
library fails the suite instead of turning this hardware test into a skip.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from src.pmpp.core.configuration import Configuration
from src.pmpp.core.utils import AXIS_NAME
from src.pmpp.distributed.configuration import MultiGPUConfiguration
from src.pmpp.distributed.cuda import extension_status
from src.pmpp.nbody.integrator import _fused_drift_route


def _available_devices():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if len(devices) < 4:
        if os.environ.get("PMPP_REQUIRE_HYBRID_NATIVE") == "1":
            pytest.fail("four GPU devices are required for hybrid native qualification")
        pytest.skip("four GPU devices are unavailable")
    return devices[:4]


def _configuration(*, hybrid, far_send_capacity=8, far_recv_capacity=8):
    devices = _available_devices()
    status = extension_status()
    if hybrid and not status["hybrid_feature"]:
        if os.environ.get("PMPP_REQUIRE_HYBRID_NATIVE") == "1":
            pytest.fail("loaded CUDA routing manifest does not advertise the hybrid ABI")
        pytest.skip("hybrid CUDA library has not been built")
    mesh = Mesh(np.asarray(devices), (AXIS_NAME,))
    return Configuration(
        1.0, (16, 16, 16), mesh_shape=1, float_dtype=jnp.float32,
        multigpu=MultiGPUConfiguration(
            compute_mesh=mesh, mode="mesh_halo", cuda_routing=True,
            cuda_routing_backend="bidir_mergepath",
            migration_policy="hybrid" if hybrid else "neighbor_only",
            far_send_capacity=far_send_capacity if hybrid else None,
            far_recv_capacity=far_recv_capacity if hybrid else None,
            far_chunk_size=2 if hybrid else None,
        ),
        max_ptcl_per_slice=8, max_share_ptcl=4,
        max_halo_share_ptcl=4, max_share_gather_ptcl=4,
    )


def _input_state(*, far, velocity_driven=False):
    pmid = np.zeros((4, 8, 3), np.int16)
    disp = np.zeros((4, 8, 3), np.float32)
    vel = np.zeros_like(disp)
    unused = np.ones((4, 8), bool)
    for source in range(4):
        for row in range(4):
            pmid[source, row] = (source * 4 + row, source, row)
            unused[source, row] = False
            vel[source, row, 1] = source * 4 + row + 0.25
        drifts = vel if velocity_driven else disp
        if far:
            drifts[source, 0, 0] = 8
            drifts[source, 1, 0] = -8
        drifts[source, 2, 0] = 4
    return (jnp.asarray(pmid.reshape(-1, 3)), jnp.asarray(disp.reshape(-1, 3)),
            jnp.asarray(vel.reshape(-1, 3)), jnp.asarray(unused.reshape(-1)))


def _oracle(pmid, disp, vel, unused, factor):
    pmid = np.asarray(pmid).reshape(4, 8, 3)
    disp = np.asarray(disp).reshape(4, 8, 3)
    vel = np.asarray(vel).reshape(4, 8, 3)
    unused = np.asarray(unused).reshape(4, 8)
    expected = [[] for _ in range(4)]
    for source in range(4):
        for row in range(8):
            if unused[source, row]:
                continue
            drifted = disp[source, row] + factor * vel[source, row]
            destination = int(((float(pmid[source, row, 0]) + float(drifted[0])) % 16) // 4)
            expected[destination].append((tuple(int(x) for x in pmid[source, row]), drifted, vel[source, row]))
    for entries in expected:
        entries.sort(key=lambda item: item[0])
    return expected


@pytest.mark.parametrize("far", [False, True])
def test_native_fused_route_matches_global_oracle(far):
    hybrid = _configuration(hybrid=True)
    strict = _configuration(hybrid=False)
    pmid, disp, vel, unused = _input_state(far=far)
    result = hybrid.mGPU_halo_moving_low_memory(pmid, disp, vel, jnp.float32(0), unused)
    assert not bool(result[5])
    assert int(result[7]) == 0
    expected = _oracle(pmid, disp, vel, unused, 0)
    out_pmid = np.asarray(result[0]).reshape(4, 8, 3)
    out_disp = np.asarray(result[1]).reshape(4, 8, 3)
    out_vel = np.asarray(result[2]).reshape(4, 8, 3)
    out_unused = np.asarray(result[4]).reshape(4, 8)
    for destination in range(4):
        entries = expected[destination]
        np.testing.assert_array_equal(out_unused[destination],
                                      [False] * len(entries) + [True] * (8 - len(entries)))
        for row, (key, expected_disp, expected_vel) in enumerate(entries):
            np.testing.assert_array_equal(out_pmid[destination, row], key)
            np.testing.assert_array_equal(out_disp[destination, row], expected_disp)
            np.testing.assert_array_equal(out_vel[destination, row], expected_vel)
    strict_result = strict.mGPU_halo_moving_low_memory(pmid, disp, vel, jnp.float32(0), unused)
    if far:
        assert bool(strict_result[5])
    else:
        for actual, reference in zip(result[:5], strict_result[:5]):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


def test_native_fused_saved_input_vjp_preserves_particle_identity():
    conf = _configuration(hybrid=True)
    pmid, disp, vel, unused = _input_state(far=True, velocity_driven=True)

    def route(d, v, f):
        result = _fused_drift_route(pmid, d, v, f, unused, conf)
        return result[1], result[2]

    (out_disp, out_vel), pullback = jax.vjp(route, disp, vel, jnp.float32(1))
    assert out_disp.shape == disp.shape and out_vel.shape == vel.shape
    out_pmid = conf.mGPU_halo_moving_low_memory(pmid, disp, vel, jnp.float32(1), unused)[0]
    cot_disp = out_pmid.astype(jnp.float32) + jnp.float32(1)
    cot_vel = cot_disp * jnp.float32(0.25)
    grad_disp, grad_vel, grad_factor = pullback((cot_disp, cot_vel))
    expected_disp = pmid.astype(jnp.float32) + jnp.float32(1)
    expected_disp = jnp.where(unused[:, None], 0, expected_disp)
    np.testing.assert_allclose(np.asarray(grad_disp), np.asarray(expected_disp), atol=1e-5)
    np.testing.assert_allclose(np.asarray(grad_vel), np.asarray(1.25 * expected_disp), atol=1e-5)
    np.testing.assert_allclose(np.asarray(grad_factor),
                               np.asarray(jnp.sum(expected_disp * vel)), rtol=1e-5)


@pytest.mark.parametrize("capacity_kind", ["send", "receive", "final"])
def test_native_hybrid_capacity_failures_are_global_and_clear_output(capacity_kind):
    conf = _configuration(
        hybrid=True,
        far_send_capacity=1 if capacity_kind == "send" else 8,
        far_recv_capacity=1 if capacity_kind == "receive" else 8,
    )
    pmid, disp, vel, unused = _input_state(far=True)
    if capacity_kind == "final":
        # Eight residents on rank 0 plus four direct arrivals from rank 2.
        pmid = np.asarray(pmid).reshape(4, 8, 3).copy()
        disp = np.zeros((4, 8, 3), np.float32)
        unused = np.asarray(unused).reshape(4, 8).copy()
        pmid[0, 4:, :] = np.array([[row, 1, 4 + row] for row in range(4)], np.int16)
        unused[0, 4:] = False
        disp[2, :4, 0] = -8
        pmid = jnp.asarray(pmid.reshape(-1, 3))
        disp = jnp.asarray(disp.reshape(-1, 3))
        unused = jnp.asarray(unused.reshape(-1))
    result = conf.mGPU_halo_moving_low_memory(pmid, disp, vel, jnp.float32(0), unused)
    assert bool(result[5])
    assert np.all(np.asarray(result[4]))
    assert not np.any(np.asarray(result[0]))
    assert not np.any(np.asarray(result[1]))


def test_native_hybrid_multiple_wraps_and_invalid_coordinate():
    conf = _configuration(hybrid=True)
    pmid, disp, vel, unused = _input_state(far=False)
    disp = np.asarray(disp).reshape(4, 8, 3).copy()
    disp[0, 0, 0] = 3 * 16 + 8
    disp[1, 0, 0] = -(3 * 16 + 8)
    disp = jnp.asarray(disp.reshape(-1, 3))
    result = conf.mGPU_halo_moving_low_memory(pmid, disp, vel, jnp.float32(0), unused)
    assert not bool(result[5])
    expected = _oracle(pmid, disp, vel, unused, 0)
    actual_pmid = np.asarray(result[0]).reshape(4, 8, 3)
    actual_unused = np.asarray(result[4]).reshape(4, 8)
    for destination, entries in enumerate(expected):
        np.testing.assert_array_equal(
            actual_pmid[destination, ~actual_unused[destination]],
            np.asarray([entry[0] for entry in entries], dtype=np.int16).reshape(-1, 3),
        )

    disp = np.asarray(disp).reshape(4, 8, 3).copy()
    disp[0, 0, 0] = np.inf
    bad = conf.mGPU_halo_moving_low_memory(
        pmid, jnp.asarray(disp.reshape(-1, 3)), vel, jnp.float32(0), unused,
    )
    assert bool(bad[5])
    assert np.all(np.asarray(bad[4]))


def test_native_hybrid_exact_slab_edge_and_adjacent_float32_values():
    conf = _configuration(hybrid=True)
    pmid, disp, vel, unused = _input_state(far=False)
    pmid = np.asarray(pmid).reshape(4, 8, 3).copy()
    disp = np.asarray(disp).reshape(4, 8, 3).copy()
    pmid[0, :3, 0] = 0
    disp[0, :3, 0] = (
        np.nextafter(np.float32(4), np.float32(0)),
        np.float32(4),
        np.nextafter(np.float32(4), np.float32(np.inf)),
    )
    pmid = jnp.asarray(pmid.reshape(-1, 3))
    disp = jnp.asarray(disp.reshape(-1, 3))
    result = conf.mGPU_halo_moving_low_memory(pmid, disp, vel, jnp.float32(0), unused)
    assert not bool(result[5])
    expected = _oracle(pmid, disp, vel, unused, 0)
    actual_pmid = np.asarray(result[0]).reshape(4, 8, 3)
    actual_unused = np.asarray(result[4]).reshape(4, 8)
    for destination, entries in enumerate(expected):
        np.testing.assert_array_equal(
            actual_pmid[destination, ~actual_unused[destination]],
            np.asarray([entry[0] for entry in entries], dtype=np.int16).reshape(-1, 3),
        )


def test_native_hybrid_full_nbody_gradient_matches_strict_without_far():
    """Exercise hybrid LPT, N-body, scatter, and the ordinary custom adjoint."""
    from src.pmpp.cic import scatter
    from src.pmpp.cosmology import SimpleLCDM, boltzmann
    from src.pmpp.initial_conditions import linear_modes, lpt, white_noise
    from src.pmpp.nbody import nbody

    mesh = Mesh(np.asarray(_available_devices()), (AXIS_NAME,))

    def make_conf(hybrid):
        return Configuration(
            12.5, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float32,
            cosmo_dtype=jnp.float32,
            multigpu=MultiGPUConfiguration(
                compute_mesh=mesh, mode="mesh_halo", cuda_routing=True,
                cuda_routing_backend="bidir_mergepath",
                migration_policy="hybrid" if hybrid else "neighbor_only",
                far_send_capacity=32 if hybrid else None,
                far_recv_capacity=32 if hybrid else None,
                far_chunk_size=8 if hybrid else None,
            ),
            max_ptcl_per_slice=256, max_share_ptcl=128,
            max_halo_share_ptcl=128, max_share_gather_ptcl=128,
            a_start=0.1, a_stop=0.15, a_nbody_maxstep=0.05,
            pallas_cic=False,
        )

    def evaluate(conf):
        cosmo = boltzmann(SimpleLCDM(conf), conf)
        modes = linear_modes(white_noise(3, conf, real=True), cosmo, conf)
        particles = lpt(modes, cosmo, conf)

        def loss(initial_disp):
            evolved = nbody(particles.replace(disp=initial_disp), cosmo, conf)
            density = scatter(evolved, conf)
            return jnp.mean((density - jnp.float32(1)) ** 2)

        return jax.jit(jax.value_and_grad(loss))(particles.disp)

    strict_loss, strict_grad = evaluate(make_conf(False))
    hybrid_loss, hybrid_grad = evaluate(make_conf(True))
    np.testing.assert_allclose(np.asarray(hybrid_loss), np.asarray(strict_loss), rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(hybrid_grad), np.asarray(strict_grad), rtol=5e-3, atol=5e-5)
    assert np.all(np.isfinite(np.asarray(hybrid_grad)))
