import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import SimpleLCDM as SimpleLCDM_PMWD
from pmwd.nbody import kick as kick_pmwd, kick_adj as kick_adj_pmwd
from pmwd.particles import Particles as ParticlesPMWD

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.cosmo import SimpleLCDM as SimpleLCDM_PMPP
from src.particles import Particles
from src.steps import kick as kick_pmpp, kick_adj as kick_adj_pmpp

from test_utils import init_conf

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def _build_state():
    conf = init_conf(
        num_ptcl=16,
        mesh_shape=1,
        box_size=100.0,
        num_devices=2,
        max_ptcl_per_slice=1.25,
        max_share_ptcl=32,
        max_share_gather_ptcl=128,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf.ptcl_spacing,
        ptcl_grid_shape=conf.ptcl_grid_shape,
        mesh_shape=conf.mesh_shape,
        a_start=conf.a_start,
        a_nbody_maxstep=conf.a_nbody_maxstep,
    )
    cosmo_pmwd = boltzmann_pmwd(SimpleLCDM_PMWD(conf_pmwd), conf_pmwd)
    cosmo_pmpp = boltzmann_pmpp(SimpleLCDM_PMPP(conf), conf)

    ptcl_pmwd = ParticlesPMWD.gen_grid(conf_pmwd, vel=True, acc=True)
    key = jax.random.PRNGKey(0)
    key_disp, key_vel, key_acc = jax.random.split(key, 3)
    ptcl_pmwd = ptcl_pmwd.replace(
        disp=jax.random.uniform(
            key_disp,
            ptcl_pmwd.disp.shape,
            minval=-0.25 * conf.cell_size,
            maxval=0.25 * conf.cell_size,
        ).astype(conf.float_dtype),
        vel=(jax.random.normal(key_vel, ptcl_pmwd.vel.shape) * 0.15).astype(conf.float_dtype),
        acc=(jax.random.normal(key_acc, ptcl_pmwd.acc.shape) * 0.2).astype(conf.float_dtype),
    )
    ptcl_pmpp = Particles.from_ptcl(ptcl_pmwd, conf)
    return conf, conf_pmwd, cosmo_pmpp, cosmo_pmwd, ptcl_pmpp, ptcl_pmwd


def _slot_mapping(ptcl_pmwd, conf):
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

    missing = np.flatnonzero(first_slot < 0)
    if missing.size:
        raise AssertionError(f"Missing particle ids: {missing[:10].tolist()}")

    return pid_slots, valid_slots, first_slot


def _reduce_input_slots(slot_values, pid_slots, valid_slots, conf):
    reduced = np.zeros((conf.ptcl_num, slot_values.shape[-1]), dtype=np.float64)
    for slot, pid in enumerate(pid_slots):
        if valid_slots[slot]:
            reduced[pid] += slot_values[slot]
    return reduced


def _tree_max_abs_diff(ref_tree, got_tree):
    diffs = []
    for ref, got in zip(tree_leaves(ref_tree), tree_leaves(got_tree)):
        if ref is None or got is None:
            continue
        ref_np = np.asarray(jax.device_get(ref))
        got_np = np.asarray(jax.device_get(got))
        if ref_np.dtype.kind not in "fc":
            continue
        diffs.append(float(np.max(np.abs(got_np - ref_np))))
    return max(diffs, default=0.0)


def test_kick_matches_pmwd_for_forward_and_adjoint():
    if GPU_COUNT < 2:
        if pytest is not None:
            pytest.skip("kick gradient test requires 2 GPUs")
        raise SystemExit("kick gradient test requires 2 GPUs")

    conf, conf_pmwd, cosmo_pmpp, cosmo_pmwd, ptcl_pmpp, ptcl_pmwd = _build_state()
    pid_slots, valid_slots, first_slot = _slot_mapping(ptcl_pmwd, conf)

    a_acc = conf.a_start
    a_prev = conf.a_start
    a_next = conf.a_start * 1.4

    out_pmwd = kick_pmwd(a_acc, a_prev, a_next, ptcl_pmwd, cosmo_pmwd, conf_pmwd)
    out_pmpp = kick_pmpp(a_acc, a_prev, a_next, ptcl_pmpp, cosmo_pmpp, conf)

    assert np.allclose(np.asarray(jax.device_get(out_pmpp.disp))[first_slot], np.asarray(jax.device_get(out_pmwd.disp)), atol=1e-8, rtol=1e-8)
    assert np.allclose(np.asarray(jax.device_get(out_pmpp.vel))[first_slot], np.asarray(jax.device_get(out_pmwd.vel)), atol=1e-8, rtol=1e-8)
    assert np.allclose(np.asarray(jax.device_get(out_pmpp.acc))[first_slot], np.asarray(jax.device_get(out_pmwd.acc)), atol=1e-8, rtol=1e-8)

    key = jax.random.PRNGKey(1)
    key_disp, key_vel, key_acc = jax.random.split(key, 3)
    cot_disp_unique = jax.random.normal(key_disp, out_pmwd.disp.shape, dtype=out_pmwd.disp.dtype)
    cot_vel_unique = jax.random.normal(key_vel, out_pmwd.vel.shape, dtype=out_pmwd.vel.dtype)
    cot_acc_unique = jax.random.normal(key_acc, out_pmwd.acc.shape, dtype=out_pmwd.acc.dtype)
    ptcl_cot_pmwd = out_pmwd.replace(disp=cot_disp_unique, vel=cot_vel_unique, acc=cot_acc_unique)

    first_slot_j = jnp.asarray(first_slot)
    ptcl_cot_pmpp = out_pmpp.replace(
        disp=jnp.zeros(out_pmpp.disp.shape, dtype=out_pmpp.disp.dtype).at[first_slot_j].set(cot_disp_unique),
        vel=jnp.zeros(out_pmpp.vel.shape, dtype=out_pmpp.vel.dtype).at[first_slot_j].set(cot_vel_unique),
        acc=jnp.zeros(out_pmpp.acc.shape, dtype=out_pmpp.acc.dtype).at[first_slot_j].set(cot_acc_unique),
    )

    zero_cosmo_pmwd = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmwd)
    zero_cosmo_pmpp = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmpp)
    zero_force_pmwd = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmwd)
    zero_force_pmpp = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmpp)

    _, in_cot_pmwd, cosmo_cot_pmwd = kick_adj_pmwd(
        a_acc, a_prev, a_next, ptcl_pmwd, ptcl_cot_pmwd, cosmo_pmwd, zero_cosmo_pmwd, zero_force_pmwd, conf_pmwd
    )
    _, in_cot_pmpp, cosmo_cot_pmpp = kick_adj_pmpp(
        a_acc, a_prev, a_next, ptcl_pmpp, ptcl_cot_pmpp, cosmo_pmpp, zero_cosmo_pmpp, zero_force_pmpp, conf
    )

    disp_in_pmpp = _reduce_input_slots(np.asarray(jax.device_get(in_cot_pmpp.disp)), pid_slots, valid_slots, conf)
    vel_in_pmpp = _reduce_input_slots(np.asarray(jax.device_get(in_cot_pmpp.vel)), pid_slots, valid_slots, conf)
    acc_in_pmpp = _reduce_input_slots(np.asarray(jax.device_get(in_cot_pmpp.acc)), pid_slots, valid_slots, conf)

    assert np.allclose(disp_in_pmpp, np.asarray(jax.device_get(in_cot_pmwd.disp)), atol=1e-8, rtol=1e-8)
    assert np.allclose(vel_in_pmpp, np.asarray(jax.device_get(in_cot_pmwd.vel)), atol=1e-8, rtol=1e-8)
    assert np.allclose(acc_in_pmpp, np.asarray(jax.device_get(in_cot_pmwd.acc)), atol=1e-8, rtol=1e-8)

    assert _tree_max_abs_diff(cosmo_cot_pmwd, cosmo_cot_pmpp) < 1e-5


if pytest is not None:
    test_kick_matches_pmwd_for_forward_and_adjoint = pytest.mark.skipif(
        GPU_COUNT < 2,
        reason="kick gradient test requires 2 GPUs",
    )(test_kick_matches_pmwd_for_forward_and_adjoint)


if __name__ == "__main__":
    test_kick_matches_pmwd_for_forward_and_adjoint()
    print("kick regression passed")
