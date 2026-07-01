import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import Cosmology as CosmologyPMWD
from pmwd.nbody import drift as drift_pmwd
from pmwd.nbody import nbody as nbody_pmwd
from pmwd.nbody import nbody_init as nbody_init_pmwd
from pmwd.particles import Particles as ParticlesPMWD
from pmwd.scatter import scatter as scatter_pmwd

from pmpp.boltzmann import boltzmann as boltzmann_pmpp
from pmpp.camels_io import load_camels_pair
from pmpp.configuration import Configuration
from pmpp.cosmo import Cosmology
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.nbody import nbody as nbody_pmpp
from pmpp.nbody import nbody_init as nbody_init_pmpp
from pmpp.particles import Particles
from pmpp.scatter import scatter as scatter_pmpp
from pmpp.steps import drift as drift_pmpp
from pmpp.utils import create_compute_mesh

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])
CAMELS_DIR = REPO_ROOT / "CAMELS"
MESH_SHAPES = (1, 2)
A_NBODY_MAXSTEP = 1 / 64
MAX_PTCL_FACTOR = 1.12
MAX_SHARE_PTCL = 50_000
FULL_FORWARD_RMSE_MAX = 3e-2
FULL_FORWARD_MEAN_ABS_MAX = 1e-3
FULL_FORWARD_COSINE_MIN = 0.999999


def _skip(reason):
    if pytest is not None:
        pytest.skip(reason)
    raise SystemExit(reason)


def _require_two_gpus():
    if GPU_COUNT >= 2:
        return
    _skip("CAMELS PMWD/PM++ agreement test requires at least 2 GPUs")


def _load_camels_or_skip():
    if not CAMELS_DIR.exists():
        _skip(f"CAMELS directory not found at {CAMELS_DIR}")
    try:
        return load_camels_pair(CAMELS_DIR)
    except (FileNotFoundError, OSError) as exc:
        _skip(f"CAMELS ICs are not available: {exc}")


def _build_confs(pair, mesh_shape):
    meta = pair.metadata
    grid_size = meta.grid_size
    ptcl_spacing = meta.box_size / grid_size
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)
    avg_per_gpu = (grid_size ** 3) / len(gpu_devices)
    max_ptcl_per_slice = int(np.ceil(avg_per_gpu * MAX_PTCL_FACTOR))

    conf_pmpp = Configuration(
        ptcl_spacing,
        (grid_size,) * 3,
        mesh_shape=mesh_shape,
        multigpu=MultiGPUConfiguration(compute_mesh=compute_mesh, mode="mesh_halo"),
        max_ptcl_per_slice=max_ptcl_per_slice,
        max_share_ptcl=MAX_SHARE_PTCL,
        max_halo_share_ptcl=MAX_SHARE_PTCL,
        max_share_gather_ptcl=MAX_SHARE_PTCL,
        a_start=meta.a_start,
        a_stop=1.0,
        a_nbody_maxstep=A_NBODY_MAXSTEP,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=ptcl_spacing,
        ptcl_grid_shape=(grid_size,) * 3,
        mesh_shape=mesh_shape,
        a_start=meta.a_start,
        a_stop=1.0,
        a_nbody_maxstep=A_NBODY_MAXSTEP,
    )
    return conf_pmpp, conf_pmwd


def _build_cosmologies(pair, conf_pmpp, conf_pmwd):
    meta = pair.metadata
    cosmo_pmpp = Cosmology.from_sigma8(
        conf_pmpp,
        sigma8=meta.sigma8,
        n_s=meta.n_s,
        Omega_m=meta.omega_m,
        Omega_b=meta.omega_b,
        h=meta.h,
    )
    cosmo_pmpp = boltzmann_pmpp(cosmo_pmpp, conf_pmpp)

    cosmo_pmwd = CosmologyPMWD.from_sigma8(
        conf_pmwd,
        sigma8=meta.sigma8,
        n_s=meta.n_s,
        Omega_m=meta.omega_m,
        Omega_b=meta.omega_b,
        h=meta.h,
    )
    cosmo_pmwd = boltzmann_pmwd(cosmo_pmwd, conf_pmwd)
    return cosmo_pmpp, cosmo_pmwd


def _ordered_camels_ic(pair):
    order = np.argsort(np.asarray(pair.ids, dtype=np.int64))
    ids = np.asarray(pair.ids, dtype=np.int64)[order]
    expected = np.arange(ids.size, dtype=np.int64)
    assert np.array_equal(ids, expected), "CAMELS IC ids do not map to the canonical particle-grid order"
    ic_pos = jnp.asarray(np.asarray(pair.ic_pos, dtype=np.float32)[order])
    ic_vel = jnp.asarray(np.asarray(pair.ic_vel, dtype=np.float32)[order])
    return ic_pos, ic_vel


def _pmwd_particles_from_ordered_pos(conf_pmwd, pos, vel):
    grid = ParticlesPMWD.gen_grid(conf_pmwd)
    dtype = conf_pmwd.float_dtype
    box_size = jnp.asarray(conf_pmwd.box_size, dtype=dtype)
    anchor = grid.pmid.astype(dtype) * conf_pmwd.cell_size
    disp = jnp.mod(pos.astype(dtype) - anchor + 0.5 * box_size, box_size) - 0.5 * box_size
    return ParticlesPMWD(conf_pmwd, grid.pmid, disp, vel=vel.astype(dtype))


def _first_drift_args(conf):
    a_prev = float(conf.a_nbody[0])
    a_next = float(conf.a_nbody[1])
    D = 0.0
    for d, _k in np.asarray(jax.device_get(conf.symp_splits), dtype=np.float64):
        if float(d) == 0.0:
            continue
        D += float(d)
        a_disp_next = a_prev * (1.0 - D) + a_next * D
        return a_prev, a_prev, a_disp_next
    raise AssertionError("No non-zero drift split found in symplectic integrator")


def _density_stats(label, dens_pmpp, dens_pmwd):
    diff = dens_pmpp - dens_pmwd
    return (
        f"{label}: rmse={np.sqrt(np.mean(diff ** 2)):.3e}, "
        f"max_abs={np.max(np.abs(diff)):.3e}, "
        f"pmpp_mean={dens_pmpp.mean():.9f}, pmwd_mean={dens_pmwd.mean():.9f}"
    )


def _assert_density_match(label, dens_pmpp, dens_pmwd):
    assert dens_pmpp.shape == dens_pmwd.shape, f"{label}: shape mismatch {dens_pmpp.shape} vs {dens_pmwd.shape}"
    assert np.isclose(dens_pmpp.mean(), 1.0, atol=5e-6), f"{label}: PM++ mean drifted from mass conservation"
    assert np.isclose(dens_pmwd.mean(), 1.0, atol=5e-6), f"{label}: PMWD mean drifted from mass conservation"
    assert np.allclose(dens_pmpp, dens_pmwd, atol=2e-6, rtol=2e-6), _density_stats(label, dens_pmpp, dens_pmwd)


def _assert_particle_count_preserved(ptcl_pmpp, conf_pmpp, label):
    unused = np.asarray(jax.device_get(ptcl_pmpp.unused_index))
    valid_count = int((~unused).sum())
    assert valid_count == conf_pmpp.ptcl_num, (
        f"{label}: PM++ valid particle count changed after CAMELS load/migration "
        f"({valid_count} vs expected {conf_pmpp.ptcl_num})"
    )


def _assert_full_forward_agreement(label, dens_pmpp, dens_pmwd):
    diff = dens_pmpp - dens_pmwd
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_abs = float(np.mean(np.abs(diff)))
    cosine = float(np.dot(dens_pmpp.ravel(), dens_pmwd.ravel()) / (np.linalg.norm(dens_pmpp.ravel()) * np.linalg.norm(dens_pmwd.ravel())))
    assert np.isclose(dens_pmpp.mean(), 1.0, atol=5e-6), f"{label}: PM++ mean drifted from mass conservation"
    assert np.isclose(dens_pmwd.mean(), 1.0, atol=5e-6), f"{label}: PMWD mean drifted from mass conservation"
    assert rmse < FULL_FORWARD_RMSE_MAX, f"{label}: rmse={rmse:.6e} exceeds {FULL_FORWARD_RMSE_MAX:.6e}"
    assert mean_abs < FULL_FORWARD_MEAN_ABS_MAX, f"{label}: mean_abs={mean_abs:.6e} exceeds {FULL_FORWARD_MEAN_ABS_MAX:.6e}"
    assert cosine > FULL_FORWARD_COSINE_MIN, f"{label}: cosine={cosine:.12f} below {FULL_FORWARD_COSINE_MIN:.12f}"


def test_camels_ic_path_matches_pmwd(mesh_shape):
    _require_two_gpus()
    pair = _load_camels_or_skip()

    conf_pmpp, conf_pmwd = _build_confs(pair, mesh_shape)
    cosmo_pmpp, cosmo_pmwd = _build_cosmologies(pair, conf_pmpp, conf_pmwd)

    ic_pos, ic_vel = _ordered_camels_ic(pair)

    ptcl_pmpp = Particles.from_ordered_pos(conf_pmpp, ic_pos, vel=ic_vel)
    ptcl_pmwd = _pmwd_particles_from_ordered_pos(conf_pmwd, ic_pos, ic_vel)

    jit_scatter_pmpp = jax.jit(scatter_pmpp, static_argnames=("conf",))
    jit_scatter_pmwd = jax.jit(scatter_pmwd, static_argnames=("conf",))
    jit_nbody_init_pmpp = jax.jit(nbody_init_pmpp, static_argnames=("conf",))
    jit_nbody_init_pmwd = jax.jit(nbody_init_pmwd, static_argnames=("conf",))
    jit_drift_pmpp = jax.jit(drift_pmpp, static_argnames=("conf",))
    jit_drift_pmwd = jax.jit(drift_pmwd, static_argnames=("conf",))

    dens_load_pmpp = np.asarray(jax.device_get(jit_scatter_pmpp(ptcl_pmpp, conf_pmpp)))
    dens_load_pmwd = np.asarray(jax.device_get(jit_scatter_pmwd(ptcl_pmwd, conf_pmwd)))
    _assert_particle_count_preserved(ptcl_pmpp, conf_pmpp, f"mesh_shape={mesh_shape} load")
    _assert_density_match(f"mesh_shape={mesh_shape} load", dens_load_pmpp, dens_load_pmwd)

    ptcl_pmpp = jit_nbody_init_pmpp(conf_pmpp.a_nbody[0], ptcl_pmpp, cosmo_pmpp, conf_pmpp)
    ptcl_pmwd, _ = jit_nbody_init_pmwd(conf_pmwd.a_nbody[0], ptcl_pmwd, None, cosmo_pmwd, conf_pmwd)
    dens_init_pmpp = np.asarray(jax.device_get(jit_scatter_pmpp(ptcl_pmpp, conf_pmpp)))
    dens_init_pmwd = np.asarray(jax.device_get(jit_scatter_pmwd(ptcl_pmwd, conf_pmwd)))
    _assert_particle_count_preserved(ptcl_pmpp, conf_pmpp, f"mesh_shape={mesh_shape} nbody_init")
    _assert_density_match(f"mesh_shape={mesh_shape} nbody_init", dens_init_pmpp, dens_init_pmwd)

    a_vel, a_prev, a_next = _first_drift_args(conf_pmpp)
    ptcl_pmpp = jit_drift_pmpp(a_vel, a_prev, a_next, ptcl_pmpp, cosmo_pmpp, conf_pmpp)
    ptcl_pmwd = jit_drift_pmwd(a_vel, a_prev, a_next, ptcl_pmwd, cosmo_pmwd, conf_pmwd)
    dens_drift_pmpp = np.asarray(jax.device_get(jit_scatter_pmpp(ptcl_pmpp, conf_pmpp)))
    dens_drift_pmwd = np.asarray(jax.device_get(jit_scatter_pmwd(ptcl_pmwd, conf_pmwd)))
    _assert_particle_count_preserved(ptcl_pmpp, conf_pmpp, f"mesh_shape={mesh_shape} first_drift")
    _assert_density_match(f"mesh_shape={mesh_shape} first_drift", dens_drift_pmpp, dens_drift_pmwd)


def test_camels_full_forward_matches_pmwd(mesh_shape):
    _require_two_gpus()
    pair = _load_camels_or_skip()

    conf_pmpp, conf_pmwd = _build_confs(pair, mesh_shape)
    cosmo_pmpp, cosmo_pmwd = _build_cosmologies(pair, conf_pmpp, conf_pmwd)

    ic_pos, ic_vel = _ordered_camels_ic(pair)

    ptcl_pmpp = Particles.from_ordered_pos(conf_pmpp, ic_pos, vel=ic_vel)
    ptcl_pmwd = _pmwd_particles_from_ordered_pos(conf_pmwd, ic_pos, ic_vel)

    jit_nbody_pmpp = jax.jit(nbody_pmpp, static_argnames=("conf", "reverse"))
    jit_nbody_pmwd = jax.jit(nbody_pmwd, static_argnames=("conf", "reverse"))
    jit_scatter_pmpp = jax.jit(scatter_pmpp, static_argnames=("conf",))
    jit_scatter_pmwd = jax.jit(scatter_pmwd, static_argnames=("conf",))

    ptcl_pmpp = jit_nbody_pmpp(ptcl_pmpp, cosmo_pmpp, conf_pmpp)
    ptcl_pmwd, _ = jit_nbody_pmwd(ptcl_pmwd, None, cosmo_pmwd, conf_pmwd)

    dens_pmpp = np.asarray(jax.device_get(jit_scatter_pmpp(ptcl_pmpp, conf_pmpp)))
    dens_pmwd = np.asarray(jax.device_get(jit_scatter_pmwd(ptcl_pmwd, conf_pmwd)))
    _assert_particle_count_preserved(ptcl_pmpp, conf_pmpp, f"mesh_shape={mesh_shape} full_forward")
    _assert_full_forward_agreement(f"mesh_shape={mesh_shape} full_forward", dens_pmpp, dens_pmwd)


if pytest is not None:
    test_camels_ic_path_matches_pmwd = pytest.mark.parametrize(
        "mesh_shape",
        MESH_SHAPES,
        ids=[f"mesh{mesh_shape}" for mesh_shape in MESH_SHAPES],
    )(test_camels_ic_path_matches_pmwd)
    test_camels_full_forward_matches_pmwd = pytest.mark.parametrize(
        "mesh_shape",
        MESH_SHAPES,
        ids=[f"mesh{mesh_shape}" for mesh_shape in MESH_SHAPES],
    )(test_camels_full_forward_matches_pmwd)


if __name__ == "__main__":
    for mesh_shape in MESH_SHAPES:
        test_camels_ic_path_matches_pmwd(mesh_shape)
    for mesh_shape in MESH_SHAPES:
        test_camels_full_forward_matches_pmwd(mesh_shape)
    print("CAMELS IC PMWD/PM++ agreement checks passed")
