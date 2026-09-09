"""Numerical regressions for the communication/memory optimizations.

Run with PMPP_TEST_DEVICES=4/8 and JAX_PLATFORMS=cpu plus
XLA_FLAGS=--xla_force_host_platform_device_count=4/8 to check wider rings.
"""
import os
import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh, create_ffts
from pmpp.distributed.fft import create_shared_gradient_fft
from pmpp.nbody.gravity import neg_grad


def mesh():
    count = int(os.environ.get("PMPP_TEST_DEVICES", "2"))
    if len(jax.devices()) < count:
        pytest.skip(f"requires {count} devices")
    return create_compute_mesh(jax.devices()[:count])


@pytest.mark.parametrize("nz", [1, 7, 8])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_half_spectrum_transpose_arbitrary_cotangents(nz, dtype):
    with jax.enable_x64(dtype == jnp.float64):
        m = mesh()
        shape = (m.size * 2, m.size * 3, nz)
        x = jax.device_put(np.random.default_rng(2).normal(size=shape).astype(dtype), NamedSharding(m, P("gpus")))
        forward, _, _, _, transposed, _ = create_ffts(m)
        rng = np.random.default_rng(9)
        g = (rng.normal(size=shape[:-1] + (nz // 2 + 1, )) +
             1j * rng.normal(size=shape[:-1] +
                             (nz // 2 + 1, ))).astype(np.complex128 if dtype == jnp.float64 else np.complex64)
        expected = jax.vjp(jnp.fft.rfftn, jnp.asarray(np.asarray(x)))[1](jnp.asarray(g))[0]
        for fn, spec in [(forward, P("gpus")), (transposed, P(None, "gpus"))]:
            cot = jax.device_put(g, NamedSharding(m, spec))
            actual = jax.jit(lambda x, g: jax.vjp(fn, x)[1](g)[0])(x, cot)
            assert actual.sharding.is_equivalent_to(NamedSharding(m, P("gpus", None, None)), 3)
            captured = jax.jit(lambda g: jax.vjp(fn, x)[1](g)[0])(cot)
            assert captured.sharding.is_equivalent_to(NamedSharding(m, P("gpus", None, None)), 3)
            np.testing.assert_allclose(actual, expected, atol=2e-5 if dtype == jnp.float32 else 1e-12, rtol=2e-6)


@pytest.mark.parametrize("nz", [7, 8])
def test_shared_force_fft_and_all_pullbacks_match_reference(nz):
    m = mesh()
    shape = (m.size * 2, m.size * 3, nz)
    rng = np.random.default_rng(8)
    pot = jax.device_put(
        (rng.normal(size=shape[:-1] + (nz // 2 + 1, )) + 1j * rng.normal(size=shape[:-1] +
                                                                         (nz // 2 + 1, ))).astype(np.complex64),
        NamedSharding(m, P(None, "gpus"))
    )
    k = [
        jnp.fft.fftfreq(shape[0])[:, None, None] * 2 * jnp.pi,
        jnp.fft.fftfreq(shape[1])[None, :, None] * 2 * jnp.pi,
        jnp.fft.rfftfreq(nz)[None, None, :] * 2 * jnp.pi
    ]
    factors = tuple(neg_grad(v.astype(jnp.float32), jnp.ones_like(v, dtype=jnp.float32), 1.) for v in k)
    fn = create_shared_gradient_fft(m, shape)
    reference = lambda p, *f: jnp.stack([jnp.fft.irfftn(p * v, s=shape) for v in f])
    g = jax.device_put(rng.normal(size=(3, ) + shape).astype(np.float32), NamedSharding(m, P(None, "gpus")))
    actual, pullback = jax.vjp(fn, pot, *factors)
    expected, ref_pullback = jax.vjp(reference, jnp.asarray(np.asarray(pot)), *factors)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    for a, b in zip(jax.jit(pullback)(g), ref_pullback(jnp.asarray(np.asarray(g)))):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


def test_streaming_lpt_source_and_gradient_match_six_strain_reference():
    module = importlib.import_module("pmpp.initial_conditions.lpt")
    m = mesh()
    n = max(8, m.size * 2)
    conf = Configuration(
        1., (n, ) * 3, multigpu=MultiGPUConfiguration(compute_mesh=m, cuda_routing=False), pallas_cic=False
    )
    pot = conf.mGPU_rfftn_transposed(jnp.sin(jnp.arange(n**3, dtype=jnp.float32).reshape((n, ) * 3)))

    def ref(p):
        d = [module._strain(conf.kvec, i, i, p, conf) for i in range(3)]
        return d[0] * d[1] + d[0] * d[2] + d[1] * d[2] - sum(
            module._strain(conf.kvec, i, j, p, conf)**2 for i, j in [(0, 1), (0, 2), (1, 2)]
        )

    fn = lambda p: module._L_streaming_2lpt(conf.kvec, p, conf)
    np.testing.assert_allclose(jax.jit(fn)(pot), jax.jit(ref)(pot), rtol=2e-5, atol=2e-5)
    for fun in (fn, ref):
        g = jax.jit(jax.grad(lambda p: jnp.mean(fun(p)**2)))(pot)
        if fun is fn:
            actual = g
        else:
            np.testing.assert_allclose(actual, g, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Pallas requires GPU")
@pytest.mark.parametrize("channels", [(), (3, )])
def test_separate_halo_gather_values_and_gradients(channels):
    from pmpp.cic.pallas import pallas_gather_halos, pallas_gather, pallas_gather_bwd
    rng = np.random.default_rng(3)
    owned, left, right = [jnp.asarray(rng.normal(size=(n, 8, 8) + channels), dtype=jnp.float32) for n in (4, 1, 1)]
    pmid = jnp.asarray(rng.integers(0, 8, size=(259, 3)),
                       dtype=jnp.int16).at[:, 0].set(jnp.arange(259, dtype=jnp.int16) % 4)
    disp = jnp.asarray(rng.uniform(-.4, .4, size=(259, 3)), dtype=jnp.float32)
    valid = jnp.arange(259) % 7 != 0
    kw = dict(offset=jnp.array([-1., 0., 0.]), particle_cell_size=1., global_shape=(8, 8, 8), valid_mask=valid)
    fn = lambda d, o, l, r: pallas_gather_halos(pmid, d, o, l, r, **kw)
    val, vjp = jax.vjp(fn, disp, owned, left, right)
    full = jnp.concatenate((left, owned, right))
    expected = pallas_gather(pmid, disp, full, **kw)
    np.testing.assert_allclose(val, expected, atol=2e-6, rtol=3e-6)
    cot = jnp.asarray(rng.normal(size=val.shape), dtype=jnp.float32)
    a = jax.jit(vjp)(cot)
    d, g = pallas_gather_bwd(pmid, disp, full, cot, **kw)
    for got, want in zip(a, (d, g[1:-1], g[:1], g[-1:])):
        np.testing.assert_allclose(got, want, atol=4e-6, rtol=3e-5)


@pytest.mark.parametrize("native", [False, True])
def test_drift_routing_crossings_and_pullback_by_particle_identity(native):
    from pmpp.nbody import Particles
    from pmpp.nbody.integrator import _fused_drift_route
    m = mesh()
    n = max(8, 2 * m.size)
    conf = Configuration(
        1., (n, ) * 3, pmid_dtype=jnp.int16, pallas_cic=False, max_ptcl_per_slice=n**3 // m.size + 4 * n * n,
        max_share_ptcl=2 * n * n, multigpu=MultiGPUConfiguration(compute_mesh=m, cuda_routing=native)
    )
    if native and conf.mGPU_halo_moving_low_memory is None:
        pytest.skip("requires fused CUDA routing library")
    ptcl = Particles.gen_grid(conf)
    disp = jnp.where(ptcl.unused_index[:, None], 0., jnp.full_like(ptcl.disp, .1))
    vel = jnp.zeros_like(disp).at[:, 0].set(jnp.where(ptcl.pmid[:, 0] % 2 == 0, -.4, 1.2))
    vel = jnp.where(ptcl.unused_index[:, None], 0., vel)
    factor = jnp.float32(1.)

    def route(d, v, f):
        if native:
            return _fused_drift_route(ptcl.pmid, d, v, f, ptcl.unused_index, conf)
        return conf.mGPU_halo_moving_no_acc(
            ptcl.pmid, d, d + v * f, v, conf.halo_start, conf.halo_end, ptcl.unused_index
        )

    out = jax.jit(route)(disp, vel, factor)
    ids = np.asarray(ptcl.pmid)
    out_ids = np.asarray(out[0])
    used = ~np.asarray(ptcl.unused_index)
    out_used = ~np.asarray(out[4])
    assert used.sum() == out_used.sum() == n**3
    assert not np.asarray(out[5]).any()
    output_slots = {tuple(p): i for i, p in enumerate(out_ids) if out_used[i]}
    mapped = np.array([output_slots[tuple(p)] if u else 0 for p, u in zip(ids, used)])
    np.testing.assert_allclose(np.asarray(out[1])[mapped[used]], np.asarray(disp + vel * factor)[used], atol=2e-7)
    capacity = conf.max_ptcl_per_slice
    owner = (np.floor((out_ids[:, 0] + np.asarray(out[1])[:, 0]) % n).astype(int) // (n // m.size))
    np.testing.assert_array_equal(owner[out_used], (np.arange(len(owner)) // capacity)[out_used])
    for dev in range(m.size):
        idx = np.flatnonzero(out_used[dev * capacity:(dev + 1) * capacity]) + dev * capacity
        keys = np.ravel_multi_index(out_ids[idx].T, (n, ) * 3)
        assert np.all(np.diff(keys) > 0)
    rng = np.random.default_rng(41)
    dc = jax.device_put(rng.normal(size=disp.shape).astype(np.float32), disp.sharding)
    vc = jax.device_put(rng.normal(size=vel.shape).astype(np.float32), vel.sharding)
    pullback = jax.jit(lambda d, v, f: jax.vjp(lambda d, v, f: route(d, v, f)[1:3], d, v, f)[1]((dc, vc)))
    gd, gv, gf = pullback(disp, vel, factor)
    expected_d = np.where(used[:, None], np.asarray(dc)[mapped], 0)
    expected_v = np.where(used[:, None], np.asarray(vc)[mapped] + expected_d, 0)
    np.testing.assert_allclose(gd, expected_d, atol=2e-6)
    np.testing.assert_allclose(gv, expected_v, atol=2e-6)
    np.testing.assert_allclose(gf, np.sum(expected_d * np.asarray(vel)), rtol=2e-5, atol=3e-5)


def test_capacity_calibration_preserves_measured_peaks_and_rejects_truncation():
    from pmpp.distributed import routing_capacity_report
    report = routing_capacity_report([[90, 100], [110, 80]], [[2, 5], [7, 0]], headroom=.1)
    assert report["max_ptcl_per_slice"] >= 121
    assert report["max_share_ptcl"] == 8
    with pytest.raises(ValueError, match="truncated"):
        routing_capacity_report([10], [5], overflowed=True)
    with pytest.raises(ValueError, match="observations"):
        routing_capacity_report([], [1])
