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
from pmwd.nbody import drift as drift_pmwd, drift_adj as drift_adj_pmwd, drift_factor
from pmwd.particles import Particles as ParticlesPMWD

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.cosmo import SimpleLCDM as SimpleLCDM_PMPP
from src.particles import Particles
from src.steps import drift as drift_pmpp, drift_adj as drift_adj_pmpp

from test_utils import init_conf

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def _build_crossing_state():
    conf = init_conf(
        num_ptcl=16,
        mesh_shape=1,
        box_size=100.0,
        num_devices=2,
        max_ptcl_per_slice=1.25,
        max_share_ptcl=1024,
        max_share_gather_ptcl=2048,
        multigpu_mode="particle_halo",
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
    key_disp, key_vel = jax.random.split(key)
    disp = jax.random.uniform(
        key_disp,
        ptcl_pmwd.disp.shape,
        minval=-0.25 * conf.cell_size,
        maxval=0.25 * conf.cell_size,
    )

    a_vel = conf.a_start
    a_prev = conf.a_start
    a_next = conf.a_start * 1.5
    factor = drift_factor(a_vel, a_prev, a_next, cosmo_pmwd, conf_pmwd)

    base_vel = jax.random.normal(key_vel, ptcl_pmwd.vel.shape) * 0.03
    particle_ids = jnp.arange(conf.ptcl_num)
    pmid_x = ptcl_pmwd.pmid[:, 0]
    move_right = (pmid_x == 7) & (particle_ids % 31 == 0)
    move_left = (pmid_x == 8) & (particle_ids % 37 == 0)
    vel_x = base_vel[:, 0]
    vel_x = vel_x + move_right.astype(base_vel.dtype) * (1.25 * conf.cell_size / factor)
    vel_x = vel_x - move_left.astype(base_vel.dtype) * (1.15 * conf.cell_size / factor)
    vel = base_vel.at[:, 0].set(vel_x)

    pid_payload = jnp.repeat(jnp.arange(conf.ptcl_num, dtype=conf.float_dtype)[:, None], 3, axis=1)
    ptcl_pmwd = ptcl_pmwd.replace(
        disp=disp.astype(conf.float_dtype),
        vel=vel.astype(conf.float_dtype),
        acc=pid_payload,
    )
    ptcl_pmpp = Particles.from_ptcl(ptcl_pmwd, conf)

    return conf, conf_pmwd, cosmo_pmpp, cosmo_pmwd, ptcl_pmpp, ptcl_pmwd, a_vel, a_prev, a_next


def _first_output_slots(ptcl_pmpp_out, conf):
    pid_slots = np.asarray(jax.device_get(ptcl_pmpp_out.acc[:, 0]))
    unused = np.asarray(jax.device_get(ptcl_pmpp_out.unused_index))
    valid = ~unused

    first_slot = np.full(conf.ptcl_num, -1, dtype=np.int32)
    for slot, pid in enumerate(pid_slots):
        if valid[slot]:
            pid_i = int(round(float(pid)))
            if 0 <= pid_i < conf.ptcl_num and first_slot[pid_i] < 0:
                first_slot[pid_i] = slot

    missing = np.flatnonzero(first_slot < 0)
    if missing.size:
        raise AssertionError(f"Missing output ids: {missing[:10].tolist()}")
    return first_slot


def _reduce_input_slots(slot_values, ptcl_pmpp_in, conf):
    pid_slots = np.asarray(jax.device_get(ptcl_pmpp_in.acc[:, 0]))
    unused = np.asarray(jax.device_get(ptcl_pmpp_in.unused_index))
    valid = ~unused

    reduced = np.zeros((conf.ptcl_num, slot_values.shape[-1]), dtype=np.float64)
    for slot, pid in enumerate(pid_slots):
        if valid[slot]:
            pid_i = int(round(float(pid)))
            if 0 <= pid_i < conf.ptcl_num:
                reduced[pid_i] += slot_values[slot]
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


def test_drift_matches_pmwd_for_forward_and_adjoint():
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("drift gradient test requires at least 1 GPU")
        raise SystemExit("drift gradient test requires at least 1 GPU")

    conf, conf_pmwd, cosmo_pmpp, cosmo_pmwd, ptcl_pmpp, ptcl_pmwd, a_vel, a_prev, a_next = _build_crossing_state()

    out_pmwd = drift_pmwd(a_vel, a_prev, a_next, ptcl_pmwd, cosmo_pmwd, conf_pmwd)
    out_pmpp = drift_pmpp(a_vel, a_prev, a_next, ptcl_pmpp, cosmo_pmpp, conf)
    first_slot_out = _first_output_slots(out_pmpp, conf)

    out_disp_pmpp = np.asarray(jax.device_get(out_pmpp.disp))[first_slot_out]
    out_vel_pmpp = np.asarray(jax.device_get(out_pmpp.vel))[first_slot_out]
    out_disp_pmwd = np.asarray(jax.device_get(out_pmwd.disp))
    out_vel_pmwd = np.asarray(jax.device_get(out_pmwd.vel))

    assert np.allclose(out_disp_pmpp, out_disp_pmwd, atol=1e-8, rtol=1e-8)
    assert np.allclose(out_vel_pmpp, out_vel_pmwd, atol=1e-8, rtol=1e-8)

    key = jax.random.PRNGKey(1)
    key_disp, key_vel, key_acc = jax.random.split(key, 3)
    cot_disp_unique = jax.random.normal(key_disp, out_pmwd.disp.shape, dtype=out_pmwd.disp.dtype)
    cot_vel_unique = jax.random.normal(key_vel, out_pmwd.vel.shape, dtype=out_pmwd.vel.dtype)
    cot_acc_unique = jax.random.normal(key_acc, out_pmwd.acc.shape, dtype=out_pmwd.acc.dtype)

    ptcl_cot_pmwd = out_pmwd.replace(disp=cot_disp_unique, vel=cot_vel_unique, acc=cot_acc_unique)
    first_slot_out_j = jnp.asarray(first_slot_out)
    zeros = jnp.zeros(out_pmpp.disp.shape, dtype=out_pmpp.disp.dtype)
    ptcl_cot_pmpp = out_pmpp.replace(
        disp=zeros.at[first_slot_out_j].set(cot_disp_unique),
        vel=jnp.zeros(out_pmpp.vel.shape, dtype=out_pmpp.vel.dtype).at[first_slot_out_j].set(cot_vel_unique),
        acc=jnp.zeros(out_pmpp.acc.shape, dtype=out_pmpp.acc.dtype).at[first_slot_out_j].set(cot_acc_unique),
    )

    zero_cosmo_pmwd = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmwd)
    zero_cosmo_pmpp = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmpp)

    _, in_cot_pmwd, cosmo_cot_pmwd = drift_adj_pmwd(
        a_vel, a_prev, a_next, ptcl_pmwd, ptcl_cot_pmwd, cosmo_pmwd, zero_cosmo_pmwd, conf_pmwd
    )
    _, in_cot_pmpp, cosmo_cot_pmpp = drift_adj_pmpp(
        a_vel, a_prev, a_next, ptcl_pmpp, ptcl_cot_pmpp, cosmo_pmpp, zero_cosmo_pmpp, conf
    )

    disp_in_pmpp = _reduce_input_slots(np.asarray(jax.device_get(in_cot_pmpp.disp)), ptcl_pmpp, conf)
    vel_in_pmpp = _reduce_input_slots(np.asarray(jax.device_get(in_cot_pmpp.vel)), ptcl_pmpp, conf)
    acc_in_pmpp = _reduce_input_slots(np.asarray(jax.device_get(in_cot_pmpp.acc)), ptcl_pmpp, conf)

    assert np.allclose(disp_in_pmpp, np.asarray(jax.device_get(in_cot_pmwd.disp)), atol=1e-8, rtol=1e-8)
    assert np.allclose(vel_in_pmpp, np.asarray(jax.device_get(in_cot_pmwd.vel)), atol=1e-8, rtol=1e-8)
    assert np.allclose(acc_in_pmpp, np.asarray(jax.device_get(in_cot_pmwd.acc)), atol=1e-8, rtol=1e-8)

    assert _tree_max_abs_diff(cosmo_cot_pmwd, cosmo_cot_pmpp) < 2e-5


if pytest is not None:
    test_drift_matches_pmwd_for_forward_and_adjoint = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="drift gradient test requires at least 1 GPU",
    )(test_drift_matches_pmwd_for_forward_and_adjoint)


if __name__ == "__main__":
    test_drift_matches_pmwd_for_forward_and_adjoint()
    print("drift regression passed")
