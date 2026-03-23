import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import SimpleLCDM as SimpleLCDM_PM
from pmwd.lpt import lpt as lpt_pmwd
from pmwd.modes import linear_modes as linear_modes_pmwd
from pmwd.modes import white_noise as white_noise_pmwd
from pmwd.nbody import nbody as nbody_pmwd
from pmwd.scatter import scatter as scatter_pmwd

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.configuration import Configuration
from src.cosmo import SimpleLCDM as SimpleLCDM_PP
from src.lpt import lpt as lpt_pmpp
from src.modes import linear_modes as linear_modes_pmpp
from src.modes import white_noise as white_noise_pmpp
from src.nbody import nbody as nbody_pmpp
from src.particles import Particles
from src.scatter import scatter as scatter_pmpp
from src.utils import create_compute_mesh, pmid_to_idx

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])
REQUIRES_ONE_GPU = "forward mesh-shape tests require at least 1 GPU"

GEN_GRID_MESH_SHAPES = (1, 2, 5)
SHORT_FORWARD_CASES = (
    (1, 8, 1 / 32),
    (2, 8, 1 / 32),
)
FULL_FORWARD_MASS_CASES = (
    (1, 8, 1.0),
    (2, 8, 1.0),
    (2, 16, 1.0),
    (5, 16, 1.0),
)


def _require_two_gpus():
    if GPU_COUNT >= 2:
        return
    if pytest is not None:
        pytest.skip(REQUIRES_ONE_GPU)
    raise SystemExit(REQUIRES_ONE_GPU)


def _init_confs(
    mesh_shape=2,
    num_ptcl=8,
    a_start=1 / 64,
    a_stop=1 / 32,
    a_nbody_maxstep=None,
    max_ptcl_factor=3.0,
    max_share_ptcl=12000,
    max_share_gather_ptcl=30000,
):
    box_size = 100.0
    ptcl_grid_shape = (num_ptcl,) * 3
    ptcl_spacing = box_size / num_ptcl
    if a_nbody_maxstep is None:
        a_nbody_maxstep = a_start

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)

    conf_pmpp = Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=mesh_shape,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=int(num_ptcl**3 / len(gpu_devices) * max_ptcl_factor),
        max_share_ptcl=max_share_ptcl,
        max_share_gather_ptcl=max_share_gather_ptcl,
        a_start=a_start,
        a_stop=a_stop,
        a_nbody_maxstep=a_nbody_maxstep,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_stop=conf_pmpp.a_stop,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    return conf_pmpp, conf_pmwd


def _run_pmwd_pmpp_forward(conf_pmpp, conf_pmwd):
    cosmo_pmwd = boltzmann_pmwd(SimpleLCDM_PM(conf_pmwd), conf_pmwd)
    cosmo_pmpp = boltzmann_pmpp(SimpleLCDM_PP(conf_pmpp), conf_pmpp)

    modes_pmwd = linear_modes_pmwd(white_noise_pmwd(0, conf_pmwd), cosmo_pmwd, conf_pmwd)
    modes_pmpp = linear_modes_pmpp(white_noise_pmpp(0, conf_pmpp), cosmo_pmpp, conf_pmpp)

    ptcl_pmwd, _ = lpt_pmwd(modes_pmwd, cosmo_pmwd, conf_pmwd)
    ptcl_pmpp = lpt_pmpp(modes_pmpp, cosmo_pmpp, conf_pmpp.replace(max_share_ptcl=conf_pmpp.max_share_ptcl * 4))
    assert ptcl_pmpp.acc is None

    ptcl_pmwd, _ = nbody_pmwd(ptcl_pmwd, None, cosmo_pmwd, conf_pmwd)
    ptcl_pmpp = jax.jit(nbody_pmpp, static_argnames=("conf", "reverse"))(ptcl_pmpp, cosmo_pmpp, conf_pmpp)

    dens_pmwd = np.asarray(jax.device_get(scatter_pmwd(ptcl_pmwd, conf_pmwd)))
    dens_pmpp = np.asarray(jax.device_get(scatter_pmpp(ptcl_pmpp, conf_pmpp)))

    return ptcl_pmwd, ptcl_pmpp, dens_pmwd, dens_pmpp


def _run_pmpp_forward(conf_pmpp):
    cosmo_pmpp = boltzmann_pmpp(SimpleLCDM_PP(conf_pmpp), conf_pmpp)
    modes_pmpp = linear_modes_pmpp(white_noise_pmpp(0, conf_pmpp), cosmo_pmpp, conf_pmpp)
    ptcl_pmpp = lpt_pmpp(modes_pmpp, cosmo_pmpp, conf_pmpp.replace(max_share_ptcl=conf_pmpp.max_share_ptcl * 4))
    assert ptcl_pmpp.acc is None

    ptcl_pmpp = jax.jit(nbody_pmpp, static_argnames=("conf", "reverse"))(ptcl_pmpp, cosmo_pmpp, conf_pmpp)
    dens_pmpp = np.asarray(jax.device_get(scatter_pmpp(ptcl_pmpp, conf_pmpp)))
    return dens_pmpp


def _first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp):
    particle_keys = np.asarray(jax.device_get(pmid_to_idx(ptcl_pmwd.pmid, conf_pmwd)))
    slot_keys = np.asarray(jax.device_get(pmid_to_idx(ptcl_pmpp.pmid, conf_pmpp, ptcl_pmpp.unused_index)))

    first_slot = np.full(particle_keys.shape[0], -1, dtype=np.int32)
    key_to_particle = {int(key): pid for pid, key in enumerate(particle_keys)}

    for slot, key in enumerate(slot_keys):
        if key < 0:
            continue
        pid = key_to_particle.get(int(key))
        if pid is not None and first_slot[pid] < 0:
            first_slot[pid] = slot

    missing = np.flatnonzero(first_slot < 0)
    if missing.size:
        raise AssertionError(f"Missing PMPP slots for particle ids: {missing[:10].tolist()}")

    return first_slot


def _expected_gen_grid_x(conf_pmpp):
    step = int(conf_pmpp.ptcl_halo_width)
    count = conf_pmpp.ptcl_grid_shape[0] // conf_pmpp.num_devices + 1
    nmesh = int(conf_pmpp.nMesh)
    expected = {}
    for gpu_id, offset in zip(np.asarray(conf_pmpp.devices_index, dtype=int), np.asarray(conf_pmpp.offsets, dtype=int)):
        expected[gpu_id] = np.asarray(
            [int((offset - step + i * step) % nmesh) for i in range(count)],
            dtype=np.float64,
        )
    return expected


def test_gen_grid_keeps_one_particle_slice_halo(mesh_shape):
    _require_two_gpus()

    conf_pmpp, _ = _init_confs(mesh_shape=mesh_shape, num_ptcl=8)
    ptcl = Particles.gen_grid(conf_pmpp, vel=True)

    expected_x = _expected_gen_grid_x(conf_pmpp)

    for shard in ptcl.pmid.addressable_shards:
        gpu_id = shard.device.id
        shard_index = next(i for i, item in enumerate(ptcl.disp.addressable_shards) if item.device.id == gpu_id)
        pmid = shard.data
        disp = ptcl.disp.addressable_shards[shard_index].data
        unused = ptcl.unused_index.addressable_shards[shard_index].data
        x_mesh = np.asarray(jax.device_get(((pmid[:, 0] + disp[:, 0] * conf_pmpp.disp_size) % conf_pmpp.nMesh)[~unused]))
        x_unique = x_mesh[:: conf_pmpp.ptcl_grid_shape[1] * conf_pmpp.ptcl_grid_shape[2]]
        assert np.array_equal(x_unique, expected_x[gpu_id])


def test_short_run_forward_matches_pmwd(mesh_shape, num_ptcl, a_stop):
    _require_two_gpus()

    conf_pmpp, conf_pmwd = _init_confs(mesh_shape=mesh_shape, num_ptcl=num_ptcl, a_stop=a_stop)
    ptcl_pmwd, ptcl_pmpp, dens_pmwd, dens_pmpp = _run_pmwd_pmpp_forward(conf_pmpp, conf_pmwd)
    first_slot = _first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp)

    disp_pmwd = np.asarray(jax.device_get(ptcl_pmwd.disp))
    disp_pmpp = np.asarray(jax.device_get(ptcl_pmpp.disp))[first_slot]

    assert dens_pmpp.shape == tuple(int(s) for s in conf_pmpp.mesh_shape)
    assert np.isclose(dens_pmpp.mean(), 1.0)
    assert np.isclose(dens_pmpp.sum(), float(conf_pmpp.mesh_size))
    assert np.allclose(dens_pmpp, dens_pmwd, atol=1e-8, rtol=1e-8)
    assert np.allclose(disp_pmpp, disp_pmwd, atol=1e-8, rtol=1e-8)


def test_forward_conserves_mass_across_mesh_shapes(mesh_shape, num_ptcl, a_stop):
    _require_two_gpus()

    conf_pmpp, _ = _init_confs(mesh_shape=mesh_shape, num_ptcl=num_ptcl, a_stop=a_stop)
    dens_pmpp = _run_pmpp_forward(conf_pmpp)

    assert dens_pmpp.shape == tuple(int(s) for s in conf_pmpp.mesh_shape)
    assert np.isclose(dens_pmpp.mean(), 1.0)
    assert np.isclose(dens_pmpp.sum(), float(conf_pmpp.mesh_size))


if pytest is not None:
    test_gen_grid_keeps_one_particle_slice_halo = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason=REQUIRES_ONE_GPU,
    )(pytest.mark.parametrize("mesh_shape", GEN_GRID_MESH_SHAPES)(test_gen_grid_keeps_one_particle_slice_halo))
    test_short_run_forward_matches_pmwd = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason=REQUIRES_ONE_GPU,
    )(pytest.mark.parametrize(
        ("mesh_shape", "num_ptcl", "a_stop"),
        SHORT_FORWARD_CASES,
        ids=[f"mesh{mesh_shape}_n{num_ptcl}" for mesh_shape, num_ptcl, _ in SHORT_FORWARD_CASES],
    )(test_short_run_forward_matches_pmwd))
    test_forward_conserves_mass_across_mesh_shapes = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason=REQUIRES_ONE_GPU,
    )(pytest.mark.parametrize(
        ("mesh_shape", "num_ptcl", "a_stop"),
        FULL_FORWARD_MASS_CASES,
        ids=[f"mesh{mesh_shape}_n{num_ptcl}_a{a_stop:g}" for mesh_shape, num_ptcl, a_stop in FULL_FORWARD_MASS_CASES],
    )(test_forward_conserves_mass_across_mesh_shapes))


if __name__ == "__main__":
    for mesh_shape in GEN_GRID_MESH_SHAPES:
        test_gen_grid_keeps_one_particle_slice_halo(mesh_shape)
    for mesh_shape, num_ptcl, a_stop in SHORT_FORWARD_CASES:
        test_short_run_forward_matches_pmwd(mesh_shape, num_ptcl, a_stop)
    for mesh_shape, num_ptcl, a_stop in FULL_FORWARD_MASS_CASES:
        test_forward_conserves_mass_across_mesh_shapes(mesh_shape, num_ptcl, a_stop)
    print("forward mesh-shape regressions passed")
