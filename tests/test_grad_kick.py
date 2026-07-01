import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import SimpleLCDM as SimpleLCDM_PMWD
from pmwd.nbody import kick as kick_pmwd
from pmwd.particles import Particles as ParticlesPMWD

from pmpp.boltzmann import boltzmann as boltzmann_pmpp
from pmpp.cosmo import SimpleLCDM as SimpleLCDM_PMPP
from pmpp.particles import Particles
from pmpp.steps import kick as kick_pmpp, kick_adj as kick_adj_pmpp
from pmpp.utils import pmid_to_idx

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


def _first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp):
    particle_keys = np.asarray(jax.device_get(pmid_to_idx(ptcl_pmwd.pmid, conf_pmwd)))
    slot_keys = np.asarray(jax.device_get(pmid_to_idx(ptcl_pmpp.pmid, conf_pmpp, ptcl_pmpp.unused_index)))

    key_to_particle = {int(key): pid for pid, key in enumerate(particle_keys)}
    first_slot = np.full(conf_pmpp.ptcl_num, -1, dtype=np.int32)

    for slot, key in enumerate(slot_keys):
        if key < 0:
            continue
        pid = key_to_particle.get(int(key))
        if pid is not None and first_slot[pid] < 0:
            first_slot[pid] = slot

    missing = np.flatnonzero(first_slot < 0)
    if missing.size:
        raise AssertionError(f"Missing particle ids: {missing[:10].tolist()}")

    return first_slot


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


def test_kick_forward_matches_pmwd_and_adjoint_matches_local_vjp():
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("kick gradient test requires at least 1 GPU")
        raise SystemExit("kick gradient test requires at least 1 GPU")

    conf, conf_pmwd, cosmo_pmpp, cosmo_pmwd, ptcl_pmpp, ptcl_pmwd = _build_state()
    first_slot = _first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf)

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
    ptcl_cot_pmpp = out_pmpp.replace(
        disp=jax.random.normal(key_disp, out_pmpp.disp.shape, dtype=out_pmpp.disp.dtype),
        vel=jax.random.normal(key_vel, out_pmpp.vel.shape, dtype=out_pmpp.vel.dtype),
        acc=jax.random.normal(key_acc, out_pmpp.acc.shape, dtype=out_pmpp.acc.dtype),
    )

    zero_cosmo_pmpp = jax.tree.map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmpp)
    _, in_cot_pmpp, cosmo_cot_pmpp = kick_adj_pmpp(
        a_acc, a_prev, a_next, ptcl_pmpp, ptcl_cot_pmpp, cosmo_pmpp, zero_cosmo_pmpp, conf
    )

    def kick_only(ptcl, cosmo):
        return kick_pmpp(a_acc, a_prev, a_next, ptcl, cosmo, conf)

    _, kick_vjp = jax.vjp(kick_only, ptcl_pmpp, cosmo_pmpp)
    true_ptcl_cot, true_cosmo_cot = kick_vjp(ptcl_cot_pmpp)

    assert _tree_max_abs_diff(true_ptcl_cot, in_cot_pmpp) < 1e-8
    assert _tree_max_abs_diff(true_cosmo_cot, cosmo_cot_pmpp) < 1e-8


if pytest is not None:
    test_kick_forward_matches_pmwd_and_adjoint_matches_local_vjp = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="kick gradient test requires at least 1 GPU",
    )(test_kick_forward_matches_pmwd_and_adjoint_matches_local_vjp)


if __name__ == "__main__":
    test_kick_forward_matches_pmwd_and_adjoint_matches_local_vjp()
    print("kick regression passed")
