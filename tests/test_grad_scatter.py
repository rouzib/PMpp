import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.particles import Particles as ParticlesPMWD
from pmwd.scatter import scatter as scatter_pmwd

from pmpp.particles import Particles
from pmpp.scatter import scatter

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


def test_scatter_gradient_matches_pmwd_for_unique_pmid_particles():
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("scatter gradient test requires at least 1 GPU")
        raise SystemExit("scatter gradient test requires at least 1 GPU")

    conf = init_conf(
        num_ptcl=6,
        mesh_shape=1,
        box_size=100.0,
        num_devices=2,
        max_ptcl_per_slice=1.6,
        max_share_ptcl=20000,
        max_share_gather_ptcl=50000,
        multigpu_mode="particle_halo",
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf.ptcl_spacing,
        ptcl_grid_shape=conf.ptcl_grid_shape,
        mesh_shape=conf.mesh_shape,
        a_start=conf.a_start,
        a_nbody_maxstep=conf.a_nbody_maxstep,
    )

    ptcl_pmwd = ParticlesPMWD.gen_grid(conf_pmwd)
    key = jax.random.PRNGKey(42)
    disp = jax.random.uniform(
        key,
        shape=ptcl_pmwd.disp.shape,
        minval=-0.45 * conf.cell_size,
        maxval=0.45 * conf.cell_size,
    )
    ptcl_pmwd = ptcl_pmwd.replace(disp=disp.astype(conf.float_dtype))
    ptcl_pmpp = Particles.from_ptcl(ptcl_pmwd, conf)
    default_val = (~ptcl_pmpp.unused_index).astype(conf.float_dtype) * (conf.mesh_size / conf.ptcl_num)
    pid_slots, valid_slots, first_slot = _particle_slot_mapping(ptcl_pmwd, conf)

    def loss_pmwd(disp):
        ptcl = ptcl_pmwd.replace(disp=disp)
        dens = scatter_pmwd(ptcl, conf_pmwd)
        return 0.5 * jnp.sum(dens**2)

    def loss_pmpp(disp):
        ptcl = ptcl_pmpp.replace(disp=disp)
        dens = scatter(ptcl, conf)
        return 0.5 * jnp.sum(dens**2)

    def loss_pmpp_explicit(disp):
        ptcl = ptcl_pmpp.replace(disp=disp)
        dens = scatter(ptcl, conf, val=default_val)
        return 0.5 * jnp.sum(dens**2)

    dens_pmpp_default = np.asarray(jax.device_get(scatter(ptcl_pmpp, conf)))
    dens_pmpp_explicit = np.asarray(jax.device_get(scatter(ptcl_pmpp, conf, val=default_val)))
    # This passes at 1e-8 in float64; float32 GPU accumulation differs at the 1e-7 level.
    assert np.allclose(dens_pmpp_default, dens_pmpp_explicit, atol=1e-6, rtol=1e-6)

    grad_pmwd = np.asarray(jax.device_get(jax.grad(loss_pmwd)(ptcl_pmwd.disp)))
    grad_pmpp_slots = np.asarray(jax.device_get(jax.grad(loss_pmpp)(ptcl_pmpp.disp)))
    grad_pmpp_explicit_slots = np.asarray(jax.device_get(jax.grad(loss_pmpp_explicit)(ptcl_pmpp.disp)))
    assert np.allclose(grad_pmpp_slots, grad_pmpp_explicit_slots, atol=1e-6, rtol=1e-6)

    grad_pmpp = _sum_duplicate_slot_gradients(
        grad_pmpp_slots,
        pid_slots,
        valid_slots,
        conf.ptcl_num,
    )

    assert np.allclose(grad_pmpp, grad_pmwd, atol=1e-6, rtol=1e-6)


if pytest is not None:
    test_scatter_gradient_matches_pmwd_for_unique_pmid_particles = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="scatter gradient test requires at least 1 GPU",
    )(test_scatter_gradient_matches_pmwd_for_unique_pmid_particles)


if __name__ == "__main__":
    test_scatter_gradient_matches_pmwd_for_unique_pmid_particles()
    print("scatter gradient regression passed")
