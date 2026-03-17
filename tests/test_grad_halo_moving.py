import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.particles import Particles
from src.steps import _halo_move_vjp
from test_utils import init_conf

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def _build_probe_state(conf):
    key = jax.random.PRNGKey(0)
    ptcl = Particles.gen_grid(conf, vel=True, acc=True)

    key_vel, key_acc, key_disp = jax.random.split(key, 3)
    vel = jax.random.normal(key_vel, ptcl.vel.shape, dtype=ptcl.vel.dtype) * 0.2
    acc = jax.random.normal(key_acc, ptcl.acc.shape, dtype=ptcl.acc.dtype) * 0.2

    pmid = ptcl.pmid.reshape(conf.num_devices, conf.max_ptcl_per_slice, 3)
    disp = ptcl.disp.reshape(conf.num_devices, conf.max_ptcl_per_slice, 3)
    unused = ptcl.unused_index.reshape(conf.num_devices, conf.max_ptcl_per_slice)
    slot_ids = jnp.arange(conf.max_ptcl_per_slice)[None, :]

    x = pmid[..., 0]
    move_right = (x == 2) & (~unused) & (slot_ids % 17 == 0)
    move_left = (x == 5) & (~unused) & (slot_ids % 19 == 0)
    shift_x = move_right.astype(disp.dtype) * (1.25 * conf.cell_size)
    shift_x -= move_left.astype(disp.dtype) * (1.15 * conf.cell_size)
    shift = jnp.zeros_like(disp).at[..., 0].set(shift_x)
    disp = (disp + shift).reshape(ptcl.disp.shape)

    cot_disp = jax.random.normal(key_disp, ptcl.disp.shape, dtype=ptcl.disp.dtype)
    cot_vel = jax.random.normal(key_vel, ptcl.vel.shape, dtype=ptcl.vel.dtype)
    cot_acc = jax.random.normal(key_acc, ptcl.acc.shape, dtype=ptcl.acc.dtype)

    return ptcl, disp, vel, acc, cot_disp, cot_vel, cot_acc


def test_halo_move_vjp_matches_true_vjp():
    if GPU_COUNT < 2:
        if pytest is not None:
            pytest.skip("halo moving gradient test requires 2 GPUs")
        raise SystemExit("halo moving gradient test requires 2 GPUs")

    conf = init_conf(
        num_ptcl=8,
        mesh_shape=1,
        box_size=100.0,
        num_devices=2,
        max_ptcl_per_slice=2.5,
        max_share_ptcl=32,
        max_share_gather_ptcl=128,
    )
    ptcl, disp, vel, acc, cot_disp, cot_vel, cot_acc = _build_probe_state(conf)
    share_only_right = conf.num_devices == 2

    def outputs_only(disp_in, vel_in, acc_in):
        _, disp_out, vel_out, acc_out, *_ = conf.mGPU_halo_moving(
            ptcl.pmid,
            disp_in,
            vel_in,
            acc_in,
            conf.halo_start,
            conf.halo_end,
            ptcl.halo_mask,
            ptcl.unused_index,
            share_only_right,
        )
        return disp_out, vel_out, acc_out

    _, vjp_fn = jax.vjp(outputs_only, disp, vel, acc)
    vjp_disp, vjp_vel, vjp_acc = vjp_fn((cot_disp, cot_vel, cot_acc))

    helper_disp, helper_vel, helper_acc = _halo_move_vjp(
        ptcl,
        disp,
        vel,
        acc,
        cot_disp,
        cot_vel,
        cot_acc,
        share_only_right,
        conf,
    )

    forward = conf.mGPU_halo_moving(
        ptcl.pmid,
        disp,
        vel,
        acc,
        conf.halo_start,
        conf.halo_end,
        ptcl.halo_mask,
        ptcl.unused_index,
        share_only_right,
    )
    pmid_out, halo_out, unused_out = forward[0], forward[4], forward[5]
    legacy_disp, legacy_vel, legacy_acc = conf.mGPU_halo_moving(
        pmid_out,
        cot_disp,
        cot_vel,
        cot_acc,
        conf.halo_start,
        conf.halo_end,
        halo_out,
        unused_out,
        False,
    )[1:4]

    max_moved = int(np.asarray(jax.device_get(forward[-1])))
    assert max_moved > 0

    assert np.allclose(np.asarray(jax.device_get(helper_disp)), np.asarray(jax.device_get(vjp_disp)), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(jax.device_get(helper_vel)), np.asarray(jax.device_get(vjp_vel)), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(jax.device_get(helper_acc)), np.asarray(jax.device_get(vjp_acc)), atol=1e-6, rtol=1e-6)

    legacy_max = max(
        float(np.max(np.abs(np.asarray(jax.device_get(legacy_disp - vjp_disp))))),
        float(np.max(np.abs(np.asarray(jax.device_get(legacy_vel - vjp_vel))))),
        float(np.max(np.abs(np.asarray(jax.device_get(legacy_acc - vjp_acc))))),
    )
    assert legacy_max > 1e-2


if pytest is not None:
    test_halo_move_vjp_matches_true_vjp = pytest.mark.skipif(
        GPU_COUNT < 2,
        reason="halo moving gradient test requires 2 GPUs",
    )(test_halo_move_vjp_matches_true_vjp)


if __name__ == "__main__":
    test_halo_move_vjp_matches_true_vjp()
    print("halo moving gradient regression passed")
