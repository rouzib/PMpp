import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.gather import gather as gather_pmwd
from pmwd.particles import Particles as ParticlesPMWD
from pmwd.scatter import scatter as scatter_pmwd

from src.gather import gather as gather_pmpp
from src.particles import Particles
from src.scatter import scatter

from test_utils import init_conf

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def _particle_slot_mapping(ptcl_pmwd, conf):
    pid_payload = jnp.repeat(jnp.arange(conf.ptcl_num, dtype=jnp.int32)[:, None], 3, axis=1)

    pid_slots = []
    unused_slots = []
    for i in range(conf.num_devices):
        pmid_sliced, disp_sliced, pid_vel, _, unused_index, _ = Particles.distribute_ptcl_pos(
            ptcl_pmwd.pmid,
            ptcl_pmwd.disp,
            pid_payload,
            None,
            conf,
            i,
        )
        del pmid_sliced, disp_sliced
        pid_slots.append(np.asarray(pid_vel[:, 0]))
        unused_slots.append(np.asarray(unused_index))

    pid_slots = np.concatenate(pid_slots, axis=0)
    unused_slots = np.concatenate(unused_slots, axis=0)
    valid_slots = ~unused_slots

    first_slot = np.full(conf.ptcl_num, -1, dtype=np.int32)
    for slot, pid in enumerate(pid_slots):
        if valid_slots[slot] and first_slot[pid] < 0:
            first_slot[pid] = slot

    if np.any(first_slot < 0):
        missing = np.flatnonzero(first_slot < 0)
        raise AssertionError(f"Missing PM++ slots for particle ids: {missing[:10].tolist()}")

    return pid_slots, valid_slots, first_slot


def _slots_to_unique(grad_slots, pid_slots, valid_slots, ptcl_num):
    grad = np.zeros((ptcl_num, grad_slots.shape[-1]), dtype=grad_slots.dtype)
    for slot, pid in enumerate(pid_slots):
        if valid_slots[slot]:
            grad[pid] += grad_slots[slot]
    return grad


def _build_pair():
    conf = init_conf(
        num_ptcl=6,
        mesh_shape=1,
        box_size=100.0,
        num_devices=2,
        max_ptcl_per_slice=1.6,
        max_share_ptcl=20000,
        max_share_gather_ptcl=50000,
        multigpu_mode="mesh_halo",
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf.ptcl_spacing,
        ptcl_grid_shape=conf.ptcl_grid_shape,
        mesh_shape=conf.mesh_shape,
        a_start=conf.a_start,
        a_nbody_maxstep=conf.a_nbody_maxstep,
    )

    ptcl_pmwd = ParticlesPMWD.gen_grid(conf_pmwd)
    disp = jax.random.uniform(
        jax.random.PRNGKey(42),
        shape=ptcl_pmwd.disp.shape,
        minval=-0.45 * conf.cell_size,
        maxval=0.45 * conf.cell_size,
    )
    ptcl_pmwd = ptcl_pmwd.replace(disp=disp.astype(conf.float_dtype))
    ptcl_pmpp = Particles.from_ptcl(ptcl_pmwd, conf)
    pid_slots, valid_slots, first_slot = _particle_slot_mapping(ptcl_pmwd, conf)
    return conf, conf_pmwd, ptcl_pmwd, ptcl_pmpp, pid_slots, valid_slots, first_slot


def test_mesh_halo_scatter_matches_pmwd():
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("mesh-halo scatter test requires at least 1 GPU")
        raise SystemExit("mesh-halo scatter test requires at least 1 GPU")

    conf, conf_pmwd, ptcl_pmwd, ptcl_pmpp, pid_slots, valid_slots, _ = _build_pair()

    dens_pmwd = np.asarray(jax.device_get(scatter_pmwd(ptcl_pmwd, conf_pmwd)))
    dens_pmpp = np.asarray(jax.device_get(scatter(ptcl_pmpp, conf)))
    assert np.allclose(dens_pmpp, dens_pmwd, atol=1e-6, rtol=1e-6)

    def loss_pmwd(disp):
        return 0.5 * jnp.sum(scatter_pmwd(ptcl_pmwd.replace(disp=disp), conf_pmwd) ** 2)

    def loss_pmpp(disp):
        return 0.5 * jnp.sum(scatter(ptcl_pmpp.replace(disp=disp), conf) ** 2)

    grad_pmwd = np.asarray(jax.device_get(jax.grad(loss_pmwd)(ptcl_pmwd.disp)))
    grad_pmpp_slots = np.asarray(jax.device_get(jax.grad(loss_pmpp)(ptcl_pmpp.disp)))
    grad_pmpp = _slots_to_unique(grad_pmpp_slots, pid_slots, valid_slots, conf.ptcl_num)
    assert np.allclose(grad_pmpp, grad_pmwd, atol=1e-6, rtol=1e-6)


def test_mesh_halo_gather_matches_pmwd():
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("mesh-halo gather test requires at least 1 GPU")
        raise SystemExit("mesh-halo gather test requires at least 1 GPU")

    conf, conf_pmwd, ptcl_pmwd, ptcl_pmpp, pid_slots, valid_slots, first_slot = _build_pair()
    mesh = jax.random.normal(jax.random.PRNGKey(7), shape=conf.mesh_shape, dtype=conf.float_dtype)
    first_slot_j = jnp.array(first_slot)

    values_pmwd = np.asarray(jax.device_get(gather_pmwd(ptcl_pmwd, conf_pmwd, mesh)))
    values_pmpp_slots = np.asarray(jax.device_get(gather_pmpp(ptcl_pmpp, conf, mesh)))
    values_pmpp = values_pmpp_slots[first_slot]
    assert np.allclose(values_pmpp, values_pmwd, atol=1e-6, rtol=1e-6)

    def disp_loss_pmwd(disp_array):
        return 0.5 * jnp.sum(gather_pmwd(ptcl_pmwd.replace(disp=disp_array), conf_pmwd, mesh) ** 2)

    def disp_loss_pmpp(disp_array):
        values = gather_pmpp(ptcl_pmpp.replace(disp=disp_array), conf, mesh)[first_slot_j]
        return 0.5 * jnp.sum(values ** 2)

    grad_disp_pmwd = np.asarray(jax.device_get(jax.grad(disp_loss_pmwd)(ptcl_pmwd.disp)))
    grad_disp_pmpp_slots = np.asarray(jax.device_get(jax.grad(disp_loss_pmpp)(ptcl_pmpp.disp)))
    grad_disp_pmpp = _slots_to_unique(grad_disp_pmpp_slots, pid_slots, valid_slots, conf.ptcl_num)
    assert np.allclose(grad_disp_pmpp, grad_disp_pmwd, atol=1e-6, rtol=1e-6)

    def mesh_loss_pmwd(mesh_array):
        return 0.5 * jnp.sum(gather_pmwd(ptcl_pmwd, conf_pmwd, mesh_array) ** 2)

    def mesh_loss_pmpp(mesh_array):
        values = gather_pmpp(ptcl_pmpp, conf, mesh_array)[first_slot_j]
        return 0.5 * jnp.sum(values ** 2)

    grad_mesh_pmwd = np.asarray(jax.device_get(jax.grad(mesh_loss_pmwd)(mesh)))
    grad_mesh_pmpp = np.asarray(jax.device_get(jax.grad(mesh_loss_pmpp)(mesh)))
    assert np.allclose(grad_mesh_pmpp, grad_mesh_pmwd, atol=1e-6, rtol=1e-6)


if pytest is not None:
    test_mesh_halo_scatter_matches_pmwd = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="mesh-halo scatter test requires at least 1 GPU",
    )(test_mesh_halo_scatter_matches_pmwd)
    test_mesh_halo_gather_matches_pmwd = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="mesh-halo gather test requires at least 1 GPU",
    )(test_mesh_halo_gather_matches_pmwd)


if __name__ == "__main__":
    test_mesh_halo_scatter_matches_pmwd()
    test_mesh_halo_gather_matches_pmwd()
    print("mesh halo scatter/gather regressions passed")
