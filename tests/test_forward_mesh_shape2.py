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


def _init_confs():
    box_size = 100.0
    num_ptcl = 8
    ptcl_grid_shape = (num_ptcl,) * 3
    ptcl_spacing = box_size / num_ptcl

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)

    conf_pmpp = Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=2,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=int(num_ptcl**3 / len(gpu_devices) * 2.5),
        max_share_ptcl=4000,
        max_share_gather_ptcl=8000,
        a_start=1 / 64,
        a_stop=1 / 32,
        a_nbody_maxstep=1 / 64,
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


def test_mesh_shape_two_gen_grid_keeps_one_particle_slice_halo():
    if GPU_COUNT < 2:
        if pytest is not None:
            pytest.skip("mesh_shape=2 forward test requires 2 GPUs")
        raise SystemExit("mesh_shape=2 forward test requires 2 GPUs")

    conf_pmpp, _ = _init_confs()
    ptcl = Particles.gen_grid(conf_pmpp, vel=True)

    expected_x = {
        0: np.array([14.0, 0.0, 2.0, 4.0, 6.0]),
        1: np.array([6.0, 8.0, 10.0, 12.0, 14.0]),
    }

    for shard in ptcl.pmid.addressable_shards:
        gpu_id = shard.device.id
        shard_index = next(i for i, item in enumerate(ptcl.disp.addressable_shards) if item.device.id == gpu_id)
        pmid = shard.data
        disp = ptcl.disp.addressable_shards[shard_index].data
        unused = ptcl.unused_index.addressable_shards[shard_index].data
        x_mesh = np.asarray(jax.device_get(((pmid[:, 0] + disp[:, 0] * conf_pmpp.disp_size) % conf_pmpp.nMesh)[~unused]))
        x_unique = x_mesh[:: conf_pmpp.ptcl_grid_shape[1] * conf_pmpp.ptcl_grid_shape[2]]
        assert np.array_equal(x_unique, expected_x[gpu_id])


def test_mesh_shape_two_forward_matches_pmwd():
    if GPU_COUNT < 2:
        if pytest is not None:
            pytest.skip("mesh_shape=2 forward test requires 2 GPUs")
        raise SystemExit("mesh_shape=2 forward test requires 2 GPUs")

    conf_pmpp, conf_pmwd = _init_confs()

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
    first_slot = _first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp)

    disp_pmwd = np.asarray(jax.device_get(ptcl_pmwd.disp))
    disp_pmpp = np.asarray(jax.device_get(ptcl_pmpp.disp))[first_slot]

    assert dens_pmpp.shape == (16, 16, 16)
    assert np.isclose(dens_pmpp.mean(), 1.0)
    assert np.isclose(dens_pmpp.sum(), 4096.0)
    assert np.allclose(dens_pmpp, dens_pmwd, atol=1e-8, rtol=1e-8)
    assert np.allclose(disp_pmpp, disp_pmwd, atol=1e-8, rtol=1e-8)


if pytest is not None:
    test_mesh_shape_two_gen_grid_keeps_one_particle_slice_halo = pytest.mark.skipif(
        GPU_COUNT < 2,
        reason="mesh_shape=2 forward test requires 2 GPUs",
    )(test_mesh_shape_two_gen_grid_keeps_one_particle_slice_halo)
    test_mesh_shape_two_forward_matches_pmwd = pytest.mark.skipif(
        GPU_COUNT < 2,
        reason="mesh_shape=2 forward test requires 2 GPUs",
    )(test_mesh_shape_two_forward_matches_pmwd)


if __name__ == "__main__":
    test_mesh_shape_two_gen_grid_keeps_one_particle_slice_halo()
    test_mesh_shape_two_forward_matches_pmwd()
    print("mesh_shape=2 forward regression passed")
