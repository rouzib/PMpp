"""Finite-difference gradient verification for all PM++ differentiable components.

Each test computes a scalar loss function of the component, obtains the JAX
autodiff (AD) gradient, and compares the directional derivative against a
centered finite-difference (FD) approximation along a random perturbation.

Multi-GPU note: for tests that perturb particle displacements, the loss
function includes a halo exchange so that halo copies stay consistent with
the perturbed owned values. Float64 accumulation is used in loss reductions
to avoid cancellation errors in the FD subtraction.
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import sys
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cosmo import SimpleLCDM, E2
from src.boltzmann import boltzmann, linear_power as boltzmann_linear_power
from src.growth import growth as growth_fn
from src.modes import white_noise, linear_modes
from src.lpt import lpt
from src.scatter import scatter
from src.gather import gather
from src.gravity import gravity
from src.steps import drift, kick, force, integrate
from src.nbody import nbody
from src.particles import Particles
from src.configuration import Configuration
from src.utils import create_compute_mesh
from test_utils import init_conf

try:
    import pytest
except ImportError:
    pytest = None

GPU_COUNT = len([d for d in jax.devices() if d.platform == "gpu"])

REQUIRE_2GPU = (
    pytest.mark.skipif(GPU_COUNT < 1, reason="requires at least 1 GPU")
    if pytest else lambda f: f
)


# ═══════════════════════════ helpers ═══════════════════════════

def _sumsq(arr):
    """0.5 * sum(arr**2) accumulated in float64 for FD precision."""
    return jnp.float64(0.5) * jnp.sum(arr.astype(jnp.float64) ** 2)


def _owned_mask_2d(ptcl):
    """(N, 3) bool mask: True at non-halo, non-unused slots."""
    m = ~ptcl.unused_index
    if ptcl.halo_mask is not None:
        m = m & ~ptcl.halo_mask
    return m[:, None] & jnp.ones((1, 3), dtype=jnp.bool_)


def _owned_mask_1d(ptcl):
    """(N,) bool mask: True at non-halo, non-unused slots."""
    m = ~ptcl.unused_index
    if ptcl.halo_mask is not None:
        m = m & ~ptcl.halo_mask
    return m


def _sync_halos(ptcl, conf):
    """Run halo exchange so that halo copies reflect owned-slot values."""
    runtime = conf.multigpu
    if runtime is None:
        raise RuntimeError("Halo synchronization requires an initialized multi-GPU runtime.")

    pmid, disp, vel, acc, hm, ui, _, _ = runtime.halo_moving(
        ptcl.pmid, ptcl.disp, ptcl.disp, ptcl.vel, ptcl.acc,
        runtime.halo_start, runtime.halo_end,
        ptcl.unused_index,
    )
    return ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=acc,
                        halo_mask=hm, unused_index=ui)


def _check_fd(loss_fn, x, mask=None, eps=5e-4, rtol=1e-4, atol=1e-5, seed=42, fd_order=2):
    """Assert that the AD directional derivative matches centered FD.

    Parameters
    ----------
    loss_fn : x -> scalar (should return float64 via _sumsq for precision)
    x       : input array
    mask    : optional bool array; only True elements participate in v
    eps     : FD step size
    rtol, atol : comparison tolerances
    seed    : PRNG seed for random direction
    """
    loss_fn = jax.jit(loss_fn)
    v = jax.random.normal(jax.random.PRNGKey(seed), x.shape, x.dtype)
    if mask is not None:
        v = v * mask.astype(v.dtype)
    norm = float(jnp.linalg.norm(v))
    if norm == 0:
        return  # degenerate — nothing to test
    v = v / norm

    ad = float(jnp.sum(jax.jit(jax.grad(loss_fn))(x) * v))
    fp = float(loss_fn(x + eps * v))
    fm = float(loss_fn(x - eps * v))
    if fd_order == 4:
        fp2 = float(loss_fn(x + 2 * eps * v))
        fm2 = float(loss_fn(x - 2 * eps * v))
        fd = (-fp2 + 8 * fp - 8 * fm + fm2) / (12 * eps)
    else:
        fd = (fp - fm) / (2 * eps)

    err = abs(ad - fd)
    scale = max(abs(ad), abs(fd), 1e-8)
    assert err < atol or err / scale < rtol, (
        f"FD mismatch: AD={ad:.6e}  FD={fd:.6e}  "
        f"abs_err={err:.4e}  rel_err={err / scale:.4e}"
    )


def _base_setup(num_ptcl=8, seed=42):
    """2-GPU conf + cosmo + particles with small random perturbations."""
    conf = init_conf(
        num_ptcl=num_ptcl, mesh_shape=1, box_size=100.0,
        num_devices=jax.device_count(), max_ptcl_per_slice=1.8,
        max_share_ptcl=20000, max_share_gather_ptcl=50000,
    )
    cosmo = SimpleLCDM(conf)
    cosmo = boltzmann(cosmo, conf)

    ptcl = Particles.gen_grid(conf, vel=True, acc=True)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    valid = ~ptcl.unused_index

    disp = jnp.where(
        valid[:, None],
        jax.random.uniform(
            k1, ptcl.disp.shape,
            minval=-0.3 * conf.cell_size,
            maxval=0.3 * conf.cell_size,
            dtype=conf.float_dtype,
        ),
        ptcl.disp,
    )
    vel = jnp.where(
        valid[:, None],
        jax.random.normal(k2, ptcl.disp.shape, dtype=conf.float_dtype) * 0.01,
        ptcl.vel,
    )
    acc = jnp.where(
        valid[:, None],
        jax.random.normal(k3, ptcl.disp.shape, dtype=conf.float_dtype) * 0.001,
        ptcl.acc,
    )
    return conf, cosmo, ptcl.replace(disp=disp, vel=vel, acc=acc)


@lru_cache(maxsize=None)
def _base_setup_x64(num_ptcl=8):
    """Small float64 2-GPU setup for multi-step nbody FD checks."""
    with jax.experimental.enable_x64():
        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"][:2]
        compute_mesh = create_compute_mesh(gpu_devices)
        conf = Configuration(
            ptcl_spacing=100.0 / num_ptcl,
            ptcl_grid_shape=(num_ptcl,) * 3,
            mesh_shape=1,
            compute_mesh=compute_mesh,
            max_ptcl_per_slice=int(num_ptcl ** 3 / len(gpu_devices) * 1.8),
            max_share_ptcl=20000,
            max_share_gather_ptcl=50000,
            to_save_z=[1, 2 / 3, 1 / 3, 0],
            a_start=1 / 60,
            a_stop=1,
            a_nbody_maxstep=1 / 60,
            float_dtype=jnp.float64,
            cosmo_dtype=jnp.float64,
        )
        cosmo = boltzmann(SimpleLCDM(conf), conf)
        ptcl = lpt(linear_modes(white_noise(0, conf), cosmo, conf), cosmo, conf)
        return conf, cosmo, ptcl


@lru_cache(maxsize=None)
def _particle_fd_setup_x64(num_ptcl=4, seed=42):
    """Small float64 2-GPU setup for particle/mesh FD checks."""
    with jax.experimental.enable_x64():
        devices = [d for d in jax.devices() if d.platform == "gpu"]
        compute_mesh = create_compute_mesh(devices)
        conf = Configuration(
            ptcl_spacing=100.0 / num_ptcl,
            ptcl_grid_shape=(num_ptcl,) * 3,
            mesh_shape=1,
            compute_mesh=compute_mesh,
            max_ptcl_per_slice=int(num_ptcl ** 3 / len(devices) * 3.0),
            max_share_ptcl=4000,
            max_share_gather_ptcl=4000,
            to_save_z=[1, 2 / 3, 1 / 3, 0],
            a_start=1 / 60,
            a_nbody_maxstep=1 / 60,
            a_stop=1 / 30,
            float_dtype=jnp.float64,
            cosmo_dtype=jnp.float64,
        )
        cosmo = boltzmann(SimpleLCDM(conf), conf)

        ptcl = Particles.gen_grid(conf, vel=True, acc=True)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
        valid = ~ptcl.unused_index

        disp = jnp.where(
            valid[:, None],
            jax.random.uniform(
                k1,
                ptcl.disp.shape,
                minval=-0.3 * conf.cell_size,
                maxval=0.3 * conf.cell_size,
                dtype=conf.float_dtype,
            ),
            ptcl.disp,
        )
        vel = jnp.where(
            valid[:, None],
            jax.random.normal(k2, ptcl.disp.shape, dtype=conf.float_dtype) * 0.01,
            ptcl.vel,
        )
        acc = jnp.where(
            valid[:, None],
            jax.random.normal(k3, ptcl.disp.shape, dtype=conf.float_dtype) * 0.001,
            ptcl.acc,
        )
        return conf, cosmo, ptcl.replace(disp=disp, vel=vel, acc=acc)


# ═══════════════════════════ Scatter ═══════════════════════════

@REQUIRE_2GPU
def test_fd_scatter_disp():
    """Scatter gradient w.r.t. particle displacement."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()

    def loss(disp):
        p = _sync_halos(ptcl.replace(disp=disp), conf)
        return _sumsq(scatter(p, conf))

    _check_fd(loss, ptcl.disp, mask=_owned_mask_2d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


@REQUIRE_2GPU
def test_fd_scatter_val():
    """Scatter gradient w.r.t. explicit particle values."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()
    val = (~ptcl.unused_index).astype(conf.float_dtype) * (conf.mesh_size / conf.ptcl_num)

    def loss(v):
        return _sumsq(scatter(ptcl, conf, val=v))

    _check_fd(loss, val, mask=_owned_mask_1d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ Gather ═══════════════════════════

@REQUIRE_2GPU
def test_fd_gather_disp():
    """Gather gradient w.r.t. particle displacement."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()
    dens = scatter(ptcl, conf)

    def loss(disp):
        p = _sync_halos(ptcl.replace(disp=disp), conf)
        return _sumsq(gather(p, conf, dens))

    _check_fd(loss, ptcl.disp, mask=_owned_mask_2d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


@REQUIRE_2GPU
def test_fd_gather_mesh():
    """Gather gradient w.r.t. mesh values."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()
    dens = scatter(ptcl, conf)

    def loss(mesh):
        return _sumsq(gather(ptcl, conf, mesh))

    _check_fd(loss, dens, eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ Gravity ═══════════════════════════

@REQUIRE_2GPU
def test_fd_gravity_disp():
    """Gravity gradient w.r.t. particle displacement."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()

    def loss(disp):
        p = _sync_halos(ptcl.replace(disp=disp), conf)
        acc = gravity(conf.a_start, p, cosmo, conf)
        return _sumsq(acc)

    _check_fd(loss, ptcl.disp, mask=_owned_mask_2d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


@REQUIRE_2GPU
def test_fd_gravity_omega_m():
    """Gravity gradient w.r.t. Omega_m."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()

    def loss(om):
        acc = gravity(conf.a_start, ptcl, cosmo.replace(Omega_m=om), conf)
        return _sumsq(acc)

    _check_fd(loss, cosmo.Omega_m, eps=1e-5, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ Drift ═══════════════════════════

@REQUIRE_2GPU
def test_fd_drift_disp():
    """Drift gradient w.r.t. displacement."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()
    a0, a1 = conf.a_nbody[0], conf.a_nbody[1]

    def loss(disp):
        p = _sync_halos(ptcl.replace(disp=disp), conf)
        p_out = drift(a0, a0, a1, p, cosmo, conf)
        return _sumsq(p_out.disp)

    _check_fd(loss, ptcl.disp, mask=_owned_mask_2d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


@REQUIRE_2GPU
def test_fd_drift_vel():
    """Drift gradient w.r.t. velocity."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()
    a0, a1 = conf.a_nbody[0], conf.a_nbody[1]

    def loss(vel):
        p = _sync_halos(ptcl.replace(vel=vel), conf)
        p_out = drift(a0, a0, a1, p, cosmo, conf)
        return _sumsq(p_out.disp)

    _check_fd(loss, ptcl.vel, mask=_owned_mask_2d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ Kick ═══════════════════════════

@REQUIRE_2GPU
def test_fd_kick_vel():
    """Kick gradient w.r.t. velocity."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()
    a0, a1 = conf.a_nbody[0], conf.a_nbody[1]

    def loss(vel):
        return _sumsq(kick(a0, a0, a1, ptcl.replace(vel=vel), cosmo, conf).vel)

    _check_fd(loss, ptcl.vel, eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


@REQUIRE_2GPU
def test_fd_kick_acc():
    """Kick gradient w.r.t. acceleration."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()
    a0, a1 = conf.a_nbody[0], conf.a_nbody[1]

    def loss(acc):
        return _sumsq(kick(a0, a0, a1, ptcl.replace(acc=acc), cosmo, conf).vel)

    _check_fd(loss, ptcl.acc, eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ Force ═══════════════════════════

@REQUIRE_2GPU
def test_fd_force_disp():
    """Force (gravity) gradient w.r.t. displacement."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()

    def loss(disp):
        p = _sync_halos(ptcl.replace(disp=disp), conf)
        return _sumsq(force(conf.a_start, p, cosmo, conf).acc)

    _check_fd(loss, ptcl.disp, mask=_owned_mask_2d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ Integrate (one symplectic step) ═══════════════════════════

@REQUIRE_2GPU
def test_fd_integrate_disp():
    """Single integration step gradient w.r.t. displacement."""
    conf, cosmo, ptcl = _particle_fd_setup_x64()
    ptcl = force(conf.a_start, ptcl, cosmo, conf)
    a0, a1 = conf.a_nbody[0], conf.a_nbody[1]

    def loss(disp):
        p = _sync_halos(ptcl.replace(disp=disp), conf)
        p_out = integrate(a0, a1, p, cosmo, conf)
        return _sumsq(p_out.disp) + _sumsq(p_out.vel)

    _check_fd(loss, ptcl.disp, mask=_owned_mask_2d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ Linear Modes ═══════════════════════════

@REQUIRE_2GPU
def test_fd_linear_modes():
    """Linear modes gradient w.r.t. real-space white noise input."""
    conf, cosmo, _ = _particle_fd_setup_x64()
    wn = white_noise(0, conf, real=True)

    def loss(m):
        return _sumsq(jnp.abs(linear_modes(m, cosmo, conf)))

    _check_fd(loss, wn, eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ LPT ═══════════════════════════

@REQUIRE_2GPU
def test_fd_lpt_modes():
    """LPT gradient w.r.t. real-space white noise (through linear_modes + lpt)."""
    conf, cosmo, _ = _particle_fd_setup_x64()
    wn = white_noise(0, conf, real=True)

    def loss(m):
        p = lpt(linear_modes(m, cosmo, conf), cosmo, conf)
        return _sumsq(p.disp)

    _check_fd(loss, wn, eps=1e-4, rtol=1e-7, atol=1e-9, fd_order=4)


# ═══════════════════════════ N-body (full manual adjoint) ═══════════════════════════

@REQUIRE_2GPU
def test_fd_nbody_disp():
    """N-body gradient w.r.t. initial displacement (tests nbody_adj)."""
    conf, cosmo, ptcl = _base_setup_x64()

    def loss(disp):
        return _sumsq(nbody(ptcl.replace(disp=disp), cosmo, conf).disp)

    _check_fd(loss, ptcl.disp, mask=_owned_mask_2d(ptcl), eps=1e-4, rtol=1e-7, atol=1e-9)


@REQUIRE_2GPU
def test_fd_nbody_omega_m():
    """N-body gradient w.r.t. Omega_m (tests adjoint cosmology gradient)."""
    conf, cosmo, ptcl = _base_setup_x64()

    def loss(om):
        return _sumsq(nbody(ptcl, cosmo.replace(Omega_m=om), conf).disp)

    _check_fd(loss, cosmo.Omega_m, eps=1e-5, rtol=1e-7, atol=1e-9)


# ═══════════════════════════ Cosmology / Boltzmann ═══════════════════════════

def test_fd_E2():
    """E2(a) gradient w.r.t. scale factor."""
    conf = init_conf(num_ptcl=4, mesh_shape=1, box_size=100.0)
    cosmo = SimpleLCDM(conf)
    a = jnp.asarray(0.5, dtype=conf.cosmo_dtype)

    def loss(a_val):
        return 0.5 * E2(a_val, cosmo) ** 2

    _check_fd(loss, a, eps=1e-6)


def test_fd_growth():
    """Growth function gradient w.r.t. scale factor."""
    conf = init_conf(num_ptcl=4, mesh_shape=1, box_size=100.0)
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    a = jnp.asarray(0.5, dtype=conf.cosmo_dtype)

    def loss(a_val):
        return 0.5 * growth_fn(a_val, cosmo, conf) ** 2

    _check_fd(loss, a, eps=1e-6)


def test_fd_linear_power_As():
    """Linear power spectrum gradient w.r.t. A_s_1e9."""
    conf = init_conf(num_ptcl=4, mesh_shape=1, box_size=100.0)
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    k = jnp.asarray(0.1, dtype=conf.cosmo_dtype)

    def loss(As):
        return 0.5 * boltzmann_linear_power(k, None, cosmo.replace(A_s_1e9=As), conf) ** 2

    _check_fd(loss, cosmo.A_s_1e9, eps=1e-5)


def test_fd_linear_power_ns():
    """Linear power spectrum gradient w.r.t. n_s."""
    conf = init_conf(num_ptcl=4, mesh_shape=1, box_size=100.0)
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    k = jnp.asarray(0.1, dtype=conf.cosmo_dtype)

    def loss(ns):
        return 0.5 * boltzmann_linear_power(k, None, cosmo.replace(n_s=ns), conf) ** 2

    _check_fd(loss, cosmo.n_s, eps=1e-5)


def test_fd_linear_power_omega_m():
    """Linear power spectrum gradient w.r.t. Omega_m."""
    conf = init_conf(num_ptcl=4, mesh_shape=1, box_size=100.0)
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    k = jnp.asarray(0.1, dtype=conf.cosmo_dtype)

    def loss(om):
        return 0.5 * boltzmann_linear_power(k, None, cosmo.replace(Omega_m=om), conf) ** 2

    _check_fd(loss, cosmo.Omega_m, eps=1e-5)


# ═══════════════════════════ CLI runner ═══════════════════════════

if __name__ == "__main__":
    passed = failed = skipped = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_fd_") and callable(fn)):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except SystemExit:
            print(f"  SKIP  {name}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
