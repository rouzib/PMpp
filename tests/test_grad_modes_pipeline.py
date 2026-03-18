import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import Cosmology as CosmologyPMWD
from pmwd.modes import linear_modes as linear_modes_pmwd
from pmwd.modes import white_noise as white_noise_pmwd

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.configuration import Configuration
from src.cosmo import Cosmology as CosmologyPMPP
from src.modes import linear_modes as linear_modes_pmpp
from src.modes import white_noise as white_noise_pmpp
from src.utils import create_compute_mesh

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def _init_confs():
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
        max_ptcl_per_slice=int(num_ptcl**3 / 2 * 1.7),
        max_share_ptcl=2000,
        max_share_gather_ptcl=6000,
        to_save_z=[1, 2 / 3, 1 / 3, 0],
        a_start=1 / 60,
        a_nbody_maxstep=1 / 60,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    return conf_pmpp, conf_pmwd


def test_linear_modes_gradients_match_pmwd():
    if GPU_COUNT < 2:
        if pytest is not None:
            pytest.skip("linear-modes gradient test requires 2 GPUs")
        raise SystemExit("linear-modes gradient test requires 2 GPUs")

    conf_pmpp, conf_pmwd = _init_confs()

    seed = 0
    sigma8 = 0.8
    omega_m = 0.31
    n_s = 0.9652
    omega_b = 0.02233
    h = 0.6737

    white_pmwd = white_noise_pmwd(seed, conf_pmwd)
    white_pmpp = white_noise_pmpp(seed, conf_pmpp)
    assert np.allclose(
        np.asarray(jax.device_get(white_pmpp)),
        np.asarray(jax.device_get(white_pmwd)),
        atol=1e-8,
        rtol=1e-8,
    )

    probe = (
        jax.random.normal(jax.random.PRNGKey(123), white_pmwd.shape, dtype=conf_pmpp.float_dtype)
        + 1j * jax.random.normal(jax.random.PRNGKey(124), white_pmwd.shape, dtype=conf_pmpp.float_dtype)
    ).astype(jnp.complex128)

    def loss_pmwd(params):
        sigma8_value, omega_m_value = params
        cosmo = CosmologyPMWD.from_sigma8(
            conf_pmwd,
            sigma8=sigma8_value,
            n_s=n_s,
            Omega_m=omega_m_value,
            Omega_b=omega_b,
            h=h,
        )
        cosmo = boltzmann_pmwd(cosmo, conf_pmwd)
        modes = linear_modes_pmwd(white_pmwd, cosmo, conf_pmwd)
        return jnp.real(jnp.vdot(modes, probe)) / modes.size

    def loss_pmpp(params):
        sigma8_value, omega_m_value = params
        cosmo = CosmologyPMPP.from_sigma8(
            conf_pmpp,
            sigma8=sigma8_value,
            n_s=n_s,
            Omega_m=omega_m_value,
            Omega_b=omega_b,
            h=h,
        )
        cosmo = boltzmann_pmpp(cosmo, conf_pmpp)
        modes = linear_modes_pmpp(white_pmpp, cosmo, conf_pmpp)
        return jnp.real(jnp.vdot(modes, probe)) / modes.size

    params = jnp.array([sigma8, omega_m], dtype=jnp.float64)

    grad_pmwd = np.asarray(jax.device_get(jax.jit(jax.grad(loss_pmwd))(params)))
    grad_pmpp = np.asarray(jax.device_get(jax.jit(jax.grad(loss_pmpp))(params)))
    assert np.allclose(grad_pmpp, grad_pmwd, atol=1e-8, rtol=1e-8)

    fd_sigma8 = (float(loss_pmpp(params + jnp.array([1e-5, 0.0]))) - float(loss_pmpp(params - jnp.array([1e-5, 0.0])))) / (
        2e-5
    )
    fd_omega_m = (
        float(loss_pmpp(params + jnp.array([0.0, 3e-4])))
        - float(loss_pmpp(params - jnp.array([0.0, 3e-4])))
    ) / (6e-4)
    fd = np.array([fd_sigma8, fd_omega_m], dtype=np.float64)

    # PMPP and PMWD match to machine precision above; this finite-difference check is
    # limited by Boltzmann interpolation noise, especially for Omega_m.
    assert np.allclose(grad_pmpp, fd, atol=1e-3, rtol=1e-6)


if pytest is not None:
    test_linear_modes_gradients_match_pmwd = pytest.mark.skipif(
        GPU_COUNT < 2,
        reason="linear-modes gradient test requires 2 GPUs",
    )(test_linear_modes_gradients_match_pmwd)


if __name__ == "__main__":
    test_linear_modes_gradients_match_pmwd()
    print("linear-modes gradient regression passed")
