import sys
from pathlib import Path
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

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
from src.scatter import scatter as scatter_pmpp
from src.utils import create_compute_mesh, pmid_to_idx

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def _init_confs():
    box_size = 100.0
    num_ptcl = 4
    ptcl_grid_shape = (num_ptcl,) * 3
    ptcl_spacing = box_size / num_ptcl

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)

    conf_pmpp = Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=1,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=int(num_ptcl**3 / len(gpu_devices) * 2.5),
        max_share_ptcl=4000,
        max_share_gather_ptcl=8000,
        a_start=1 / 60,
        a_stop=1 / 15,
        a_nbody_maxstep=1 / 60,
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


def _target_density(conf_pmwd):
    cosmo = boltzmann_pmwd(SimpleLCDM_PM(conf_pmwd), conf_pmwd)
    modes = white_noise_pmwd(0, conf_pmwd)
    modes = linear_modes_pmwd(modes, cosmo, conf_pmwd)
    ptcl, _ = lpt_pmwd(modes, cosmo, conf_pmwd)
    ptcl, _ = nbody_pmwd(ptcl, None, cosmo, conf_pmwd)
    return scatter_pmwd(ptcl, conf_pmwd)


def _pmwd_forward(modes_real, base_cosmo, conf):
    cosmo = boltzmann_pmwd(base_cosmo, conf)
    modes = linear_modes_pmwd(modes_real, cosmo, conf)
    ptcl, _ = lpt_pmwd(modes, cosmo, conf)
    ptcl, _ = nbody_pmwd(ptcl, None, cosmo, conf)
    dens = scatter_pmwd(ptcl, conf)
    return ptcl, dens


def _pmpp_forward(modes_real, base_cosmo, conf):
    cosmo = boltzmann_pmpp(base_cosmo, conf)
    modes = linear_modes_pmpp(modes_real, cosmo, conf)
    ptcl = lpt_pmpp(modes, cosmo, conf.replace(max_share_ptcl=conf.max_share_ptcl * 4))
    assert ptcl.acc is None
    ptcl = nbody_pmpp(ptcl, cosmo, conf)
    dens = scatter_pmpp(ptcl, conf)
    return ptcl, dens


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


def test_nbody_matches_pmwd_for_forward_and_notebook_style_mode_gradient():
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("nbody gradient test requires at least 1 GPU")
        raise SystemExit("nbody gradient test requires at least 1 GPU")

    conf_pmpp, conf_pmwd = _init_confs()
    target_dens = _target_density(conf_pmwd)

    base_cosmo_pmpp = SimpleLCDM_PP(conf_pmpp)
    base_cosmo_pmwd = SimpleLCDM_PM(conf_pmwd)

    modes_real_pmwd = white_noise_pmwd(1, conf_pmwd, real=True)
    modes_real_pmpp = white_noise_pmpp(1, conf_pmpp, real=True)

    pmwd_forward = jax.jit(_pmwd_forward, static_argnames=("conf",))
    pmpp_forward = jax.jit(_pmpp_forward, static_argnames=("conf",))

    ptcl_pmwd, dens_pmwd = pmwd_forward(modes_real_pmwd, base_cosmo_pmwd, conf_pmwd)
    ptcl_pmpp, dens_pmpp = pmpp_forward(modes_real_pmpp, base_cosmo_pmpp, conf_pmpp)

    first_slot = _first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp)

    disp_pmwd = np.asarray(jax.device_get(ptcl_pmwd.disp))
    vel_pmwd = np.asarray(jax.device_get(ptcl_pmwd.vel))
    acc_pmwd = np.asarray(jax.device_get(ptcl_pmwd.acc))
    disp_pmpp = np.asarray(jax.device_get(ptcl_pmpp.disp))[first_slot]
    vel_pmpp = np.asarray(jax.device_get(ptcl_pmpp.vel))[first_slot]
    acc_pmpp = np.asarray(jax.device_get(ptcl_pmpp.acc))[first_slot]

    assert np.allclose(disp_pmpp, disp_pmwd, atol=1e-8, rtol=1e-8)
    assert np.allclose(vel_pmpp, vel_pmwd, atol=1e-8, rtol=1e-8)
    assert np.allclose(acc_pmpp, acc_pmwd, atol=1e-8, rtol=1e-8)
    assert np.allclose(
        np.asarray(jax.device_get(dens_pmpp)),
        np.asarray(jax.device_get(dens_pmwd)),
        atol=1e-8,
        rtol=1e-8,
    )

    def loss_pmwd(tgt_dens, modes_real, cosmo, conf):
        dens = _pmwd_forward(modes_real, cosmo, conf)[1]
        return (dens - tgt_dens).var()

    def loss_pmpp(tgt_dens, modes_real, cosmo, conf):
        dens = _pmpp_forward(modes_real, cosmo, conf)[1]
        return (dens - tgt_dens).var()

    grad_pmwd_fn = jax.jit(jax.grad(loss_pmwd, argnums=(1, 2)), static_argnames=("conf",))
    grad_pmpp_fn = jax.jit(jax.grad(loss_pmpp, argnums=(1, 2)), static_argnames=("conf",))

    grad_modes_pmwd, grad_cosmo_pmwd = grad_pmwd_fn(target_dens, modes_real_pmwd, base_cosmo_pmwd, conf_pmwd)
    grad_modes_pmpp, grad_cosmo_pmpp = grad_pmpp_fn(target_dens, modes_real_pmpp, base_cosmo_pmpp, conf_pmpp)

    assert np.allclose(
        np.asarray(jax.device_get(grad_modes_pmpp)),
        np.asarray(jax.device_get(grad_modes_pmwd)),
        atol=1e-8,
        rtol=1e-8,
    )

    for field_name in ("A_s_1e9", "n_s", "Omega_m", "Omega_b", "h"):
        grad_pmwd = np.asarray(jax.device_get(getattr(grad_cosmo_pmwd, field_name)))
        grad_pmpp = np.asarray(jax.device_get(getattr(grad_cosmo_pmpp, field_name)))
        assert np.array_equal(np.isnan(grad_pmpp), np.isnan(grad_pmwd))
        finite = np.isfinite(grad_pmwd) & np.isfinite(grad_pmpp)
        if finite.any():
            assert np.allclose(grad_pmpp[finite], grad_pmwd[finite], atol=1e-8, rtol=1e-8)


if pytest is not None:
    test_nbody_matches_pmwd_for_forward_and_notebook_style_mode_gradient = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="nbody gradient test requires at least 1 GPU",
    )(test_nbody_matches_pmwd_for_forward_and_notebook_style_mode_gradient)


if __name__ == "__main__":
    test_nbody_matches_pmwd_for_forward_and_notebook_style_mode_gradient()
    print("nbody gradient regression passed")
