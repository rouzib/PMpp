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

from src.gather import gather as gather_pmpp
from src.particles import Particles

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
        gpu_id = conf.devices_index[i]
        _, _, pid_vel, _, unused_index, _ = Particles.distribute_ptcl_pos(
            ptcl_pmwd.pmid,
            ptcl_pmwd.disp,
            pid_payload,
            None,
            conf,
            gpu_id,
        )
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
        raise AssertionError(f"Missing PMPP slots for particle ids: {missing[:10].tolist()}")

    return pid_slots, valid_slots, first_slot


def _sum_duplicate_slot_gradients(grad_pmpp_slots, pid_slots, valid_slots, ptcl_num):
    grad = np.zeros((ptcl_num, grad_pmpp_slots.shape[-1]), dtype=grad_pmpp_slots.dtype)
    for slot, pid in enumerate(pid_slots):
        if valid_slots[slot]:
            grad[pid] += grad_pmpp_slots[slot]
    return grad


def _check_gather_gradients_match_pmwd_on_unique_particles(particle_halo_gather_mesh_halo=False):
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("gather gradient test requires at least 1 GPU")
        raise SystemExit("gather gradient test requires at least 1 GPU")

    conf = init_conf(
        num_ptcl=6,
        mesh_shape=1,
        box_size=100.0,
        num_devices=2,
        max_ptcl_per_slice=1.6,
        max_share_ptcl=20000,
        max_share_gather_ptcl=50000,
        multigpu_mode="particle_halo",
        particle_halo_gather_mesh_halo=particle_halo_gather_mesh_halo,
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

    mesh = jax.random.normal(jax.random.PRNGKey(7), shape=conf.mesh_shape, dtype=conf.float_dtype)

    pid_slots, valid_slots, first_slot = _particle_slot_mapping(ptcl_pmwd, conf)
    first_slot_j = jnp.array(first_slot)

    values_pmwd = np.asarray(jax.device_get(gather_pmwd(ptcl_pmwd, conf_pmwd, mesh)))
    values_pmpp_slots = np.asarray(jax.device_get(gather_pmpp(ptcl_pmpp, conf, mesh)))
    values_pmpp = values_pmpp_slots[first_slot]
    # This passes at 1e-8 in float64; float32 GPU accumulation differs at the 1e-7 level.
    assert np.allclose(values_pmpp, values_pmwd, atol=1e-6, rtol=1e-6)

    def disp_loss_pmwd(disp_array):
        ptcl = ptcl_pmwd.replace(disp=disp_array)
        values = gather_pmwd(ptcl, conf_pmwd, mesh)
        return 0.5 * jnp.sum(values**2)

    def disp_loss_pmpp_unique(disp_array):
        ptcl = ptcl_pmpp.replace(disp=disp_array)
        values = gather_pmpp(ptcl, conf, mesh)[first_slot_j]
        return 0.5 * jnp.sum(values**2)

    grad_disp_pmwd = np.asarray(jax.device_get(jax.grad(disp_loss_pmwd)(ptcl_pmwd.disp)))
    grad_disp_pmpp_slots = np.asarray(jax.device_get(jax.grad(disp_loss_pmpp_unique)(ptcl_pmpp.disp)))
    grad_disp_pmpp = _sum_duplicate_slot_gradients(
        grad_disp_pmpp_slots,
        pid_slots,
        valid_slots,
        conf.ptcl_num,
    )
    assert np.allclose(grad_disp_pmpp, grad_disp_pmwd, atol=1e-6, rtol=1e-6)

    def mesh_loss_pmwd(mesh_array):
        values = gather_pmwd(ptcl_pmwd, conf_pmwd, mesh_array)
        return 0.5 * jnp.sum(values**2)

    def mesh_loss_pmpp_unique(mesh_array):
        values = gather_pmpp(ptcl_pmpp, conf, mesh_array)[first_slot_j]
        return 0.5 * jnp.sum(values**2)

    grad_mesh_pmwd = np.asarray(jax.device_get(jax.grad(mesh_loss_pmwd)(mesh)))
    grad_mesh_pmpp = np.asarray(jax.device_get(jax.grad(mesh_loss_pmpp_unique)(mesh)))
    assert np.allclose(grad_mesh_pmpp, grad_mesh_pmwd, atol=1e-6, rtol=1e-6)


def test_gather_gradients_match_pmwd_on_unique_particles():
    _check_gather_gradients_match_pmwd_on_unique_particles(False)


def test_particle_halo_mesh_edge_gather_gradients_match_pmwd_on_unique_particles():
    _check_gather_gradients_match_pmwd_on_unique_particles(True)


if pytest is not None:
    test_gather_gradients_match_pmwd_on_unique_particles = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="gather gradient test requires at least 1 GPU",
    )(test_gather_gradients_match_pmwd_on_unique_particles)
    test_particle_halo_mesh_edge_gather_gradients_match_pmwd_on_unique_particles = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="gather gradient test requires at least 1 GPU",
    )(test_particle_halo_mesh_edge_gather_gradients_match_pmwd_on_unique_particles)


if __name__ == "__main__":
    test_gather_gradients_match_pmwd_on_unique_particles()
    test_particle_halo_mesh_edge_gather_gradients_match_pmwd_on_unique_particles()
    print("gather gradient regression passed")
