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
from pmwd.scatter import scatter as scatter_pmwd

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.configuration import Configuration
from src.cosmo import SimpleLCDM as SimpleLCDM_PP
from src.lpt import lpt as lpt_pmpp
from src.modes import linear_modes as linear_modes_pmpp
from src.modes import white_noise as white_noise_pmpp
from src.scatter import scatter as scatter_pmpp
from src.utils import create_compute_mesh

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def _init_confs(lpt_order=1):
    box_size = 100.0
    num_ptcl = 8
    ptcl_grid_shape = (num_ptcl,) * 3
    ptcl_spacing = box_size / num_ptcl

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)

    conf_pmpp = Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=1,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=int(num_ptcl**3 / len(gpu_devices) * 2.2),
        max_share_ptcl=4000,
        max_share_gather_ptcl=8000,
        a_start=1 / 60,
        a_nbody_maxstep=1 / 60,
        lpt_order=lpt_order,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
        lpt_order=conf_pmpp.lpt_order,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    return conf_pmpp, conf_pmwd


def _dens_pmwd(modes_real, base_cosmo, conf):
    cosmo = boltzmann_pmwd(base_cosmo, conf)
    modes = linear_modes_pmwd(modes_real, cosmo, conf)
    ptcl, _ = lpt_pmwd(modes, cosmo, conf)
    return scatter_pmwd(ptcl, conf)


def _dens_pmpp(modes_real, base_cosmo, conf):
    cosmo = boltzmann_pmpp(base_cosmo, conf)
    modes = linear_modes_pmpp(modes_real, cosmo, conf)
    ptcl = lpt_pmpp(modes, cosmo, conf.replace(max_share_ptcl=conf.max_share_ptcl * 2))
    assert ptcl.acc is None
    return scatter_pmpp(ptcl, conf)


def test_lpt_matches_pmwd_for_real_input_forward_and_mode_gradients():
    if GPU_COUNT < 2:
        if pytest is not None:
            pytest.skip("LPT gradient test requires 2 GPUs")
        raise SystemExit("LPT gradient test requires 2 GPUs")

    for lpt_order in (1, 2):
        conf_pmpp, conf_pmwd = _init_confs(lpt_order=lpt_order)

        base_cosmo_pmpp = SimpleLCDM_PP(conf_pmpp)
        base_cosmo_pmwd = SimpleLCDM_PM(conf_pmwd)

        modes_real_pmwd = white_noise_pmwd(0, conf_pmwd, real=True)
        modes_real_pmpp = white_noise_pmpp(0, conf_pmpp, real=True)
        assert np.allclose(
            np.asarray(jax.device_get(modes_real_pmpp)),
            np.asarray(jax.device_get(modes_real_pmwd)),
            atol=1e-12,
            rtol=1e-12,
        )

        target_modes_real = white_noise_pmwd(1, conf_pmwd, real=True)
        target_dens = _dens_pmwd(target_modes_real, base_cosmo_pmwd, conf_pmwd)

        dens_pmwd = _dens_pmwd(modes_real_pmwd, base_cosmo_pmwd, conf_pmwd)
        dens_pmpp = _dens_pmpp(modes_real_pmpp, base_cosmo_pmpp, conf_pmpp)
        assert np.allclose(
            np.asarray(jax.device_get(dens_pmpp)),
            np.asarray(jax.device_get(dens_pmwd)),
            atol=1e-12,
            rtol=1e-12,
        )

        def loss_pmwd(modes_real):
            dens = _dens_pmwd(modes_real, base_cosmo_pmwd, conf_pmwd)
            return jnp.mean((dens - target_dens) ** 2)

        def loss_pmpp(modes_real):
            dens = _dens_pmpp(modes_real, base_cosmo_pmpp, conf_pmpp)
            return jnp.mean((dens - target_dens) ** 2)

        grad_pmwd = np.asarray(jax.device_get(jax.jit(jax.grad(loss_pmwd))(modes_real_pmwd)))
        grad_pmpp = np.asarray(jax.device_get(jax.jit(jax.grad(loss_pmpp))(modes_real_pmpp)))
        assert np.allclose(grad_pmpp, grad_pmwd, atol=1e-8, rtol=1e-8)

        probe = jax.random.normal(jax.random.PRNGKey(123), conf_pmwd.mesh_shape, dtype=conf_pmwd.float_dtype)

        def linear_loss_pmwd(modes_real):
            dens = _dens_pmwd(modes_real, base_cosmo_pmwd, conf_pmwd)
            return jnp.sum(dens * probe)

        def linear_loss_pmpp(modes_real):
            dens = _dens_pmpp(modes_real, base_cosmo_pmpp, conf_pmpp)
            return jnp.sum(dens * probe)

        linear_grad_pmwd = np.asarray(jax.device_get(jax.jit(jax.grad(linear_loss_pmwd))(modes_real_pmwd)))
        linear_grad_pmpp = np.asarray(jax.device_get(jax.jit(jax.grad(linear_loss_pmpp))(modes_real_pmpp)))
        assert np.allclose(linear_grad_pmpp, linear_grad_pmwd, atol=1e-12, rtol=1e-12)


if pytest is not None:
    test_lpt_matches_pmwd_for_real_input_forward_and_mode_gradients = pytest.mark.skipif(
        GPU_COUNT < 2,
        reason="LPT gradient test requires 2 GPUs",
    )(test_lpt_matches_pmwd_for_real_input_forward_and_mode_gradients)


if __name__ == "__main__":
    test_lpt_matches_pmwd_for_real_input_forward_and_mode_gradients()
    print("LPT gradient regression passed")
