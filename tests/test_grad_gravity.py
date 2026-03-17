import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import SimpleLCDM as SimpleLCDM_PMWD
from pmwd.gravity import gravity as gravity_pmwd
from pmwd.particles import Particles as ParticlesPMWD

from src.cosmo import SimpleLCDM as SimpleLCDM_PMPP
from src.gravity import gravity as gravity_pmpp
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


def test_gravity_matches_pmwd_for_forward_and_gradients():
    if GPU_COUNT < 2:
        if pytest is not None:
            pytest.skip("gravity gradient test requires 2 GPUs")
        raise SystemExit("gravity gradient test requires 2 GPUs")

    conf = init_conf(
        num_ptcl=4,
        mesh_shape=1,
        box_size=100.0,
        num_devices=2,
        max_ptcl_per_slice=1.8,
        max_share_ptcl=20000,
        max_share_gather_ptcl=50000,
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

    cosmo_pmwd = SimpleLCDM_PMWD(conf_pmwd)
    cosmo_pmpp = SimpleLCDM_PMPP(conf)

    pid_slots, valid_slots, first_slot = _particle_slot_mapping(ptcl_pmwd, conf)
    first_slot_j = jnp.array(first_slot)

    acc_pmwd_fn = jax.jit(lambda ptcl: gravity_pmwd(conf.a_start, ptcl, cosmo_pmwd, conf_pmwd))
    acc_pmpp_fn = jax.jit(lambda ptcl: gravity_pmpp(conf.a_start, ptcl, cosmo_pmpp, conf))

    acc_pmwd = np.asarray(jax.device_get(acc_pmwd_fn(ptcl_pmwd)))
    acc_pmpp_slots = np.asarray(jax.device_get(acc_pmpp_fn(ptcl_pmpp)))
    acc_pmpp = acc_pmpp_slots[first_slot]
    assert np.allclose(acc_pmpp, acc_pmwd, atol=1e-5, rtol=1e-4)

    def disp_loss_pmwd(disp_array):
        ptcl = ptcl_pmwd.replace(disp=disp_array)
        acc = gravity_pmwd(conf.a_start, ptcl, cosmo_pmwd, conf_pmwd)
        return 0.5 * jnp.sum(acc**2)

    def disp_loss_pmpp_unique(disp_array):
        ptcl = ptcl_pmpp.replace(disp=disp_array)
        acc = gravity_pmpp(conf.a_start, ptcl, cosmo_pmpp, conf)[first_slot_j]
        return 0.5 * jnp.sum(acc**2)

    grad_disp_pmwd_fn = jax.jit(jax.grad(disp_loss_pmwd))
    grad_disp_pmpp_fn = jax.jit(jax.grad(disp_loss_pmpp_unique))
    grad_disp_pmwd = np.asarray(jax.device_get(grad_disp_pmwd_fn(ptcl_pmwd.disp)))
    grad_disp_pmpp_slots = np.asarray(jax.device_get(grad_disp_pmpp_fn(ptcl_pmpp.disp)))
    grad_disp_pmpp = grad_disp_pmpp_slots[first_slot]
    assert np.allclose(grad_disp_pmpp, grad_disp_pmwd, atol=1e-5, rtol=1e-4)

    for pid in np.unique(pid_slots[valid_slots]):
        slots = np.flatnonzero(valid_slots & (pid_slots == pid))
        ref = grad_disp_pmpp_slots[slots[0]]
        assert np.allclose(grad_disp_pmpp_slots[slots], ref, atol=1e-5, rtol=1e-4)


if pytest is not None:
    test_gravity_matches_pmwd_for_forward_and_gradients = pytest.mark.skipif(
        GPU_COUNT < 2,
        reason="gravity gradient test requires 2 GPUs",
    )(test_gravity_matches_pmwd_for_forward_and_gradients)


if __name__ == "__main__":
    test_gravity_matches_pmwd_for_forward_and_gradients()
    print("gravity gradient regression passed")
