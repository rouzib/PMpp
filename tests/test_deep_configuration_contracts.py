import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmpp.core.configuration as configuration_module
from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh


@pytest.mark.parametrize(
    "kwargs, message", [({
        "ptcl_grid_shape": (4, 4, 4),
        "mesh_shape": (4, 4)
    }, "dimensions differ"), ({
        "ptcl_grid_shape": (4, 4, 4),
        "mesh_shape": (2, 2, 2)
    }, "cannot be smaller"), ({
        "ptcl_grid_shape": (4, 2, 2),
        "mesh_shape": (8, 5, 4)
    }, "aspect ratios differ"), ({
        "cosmo_dtype": jnp.int32
    }, "cosmo_dtype must be floating"), ({
        "pmid_dtype": jnp.uint32
    }, "pmid_dtype must be signed"), ({
        "float_dtype": jnp.int16
    }, "float_dtype must be floating"), ({
        "symp_splits": ((0.5, 0.5, 0.0), (0.5, 0.5))
    }, "not supported"), ({
        "symp_splits": ((0.2, 0.5), (0.2, 0.5))
    }, "symplectic splits"), ],
)
def test_configuration_rejects_physically_inconsistent_shapes_dtypes_and_integrators(kwargs, message):
    defaults = {"ptcl_spacing": 1.0, "ptcl_grid_shape": (4, 4, 4), "mesh_shape": 1, "pallas_cic": False}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        Configuration(**defaults)


def test_configuration_emits_the_intended_runtime_warning_for_requested_unavailable_pallas(monkeypatch):
    monkeypatch.setattr(configuration_module, "pallas_cic_supported", lambda dtype: False)
    with pytest.warns(RuntimeWarning, match="Pallas CIC was requested but is unavailable"):
        conf = Configuration(1.0, (2, 2, 2), mesh_shape=1, pallas_cic=True)
    assert conf.pallas_cic is True


def test_configuration_normalizes_schedules_save_fields_and_derived_units_exactly():
    conf = Configuration(
        2.5, (2, 3, 4), mesh_shape=(4, 6, 8), pallas_cic=False, a_start=0.2, a_stop=0.8, a_lpt_maxstep=0.07,
        a_nbody_maxstep=0.21, a_custom=[0.2, 0.31, 0.8], growth_a_custom=np.asarray([0.0, 0.2, 0.8]),
        to_save_z=[3, 1, 0], slice_to_save=[0, 2, 4],
    )
    assert conf.a_custom == (0.2, 0.31, 0.8)
    assert conf.growth_a_custom == (0.0, 0.2, 0.8)
    np.testing.assert_allclose(np.asarray(conf.a_nbody), [0.2, 0.31, 0.8])
    np.testing.assert_allclose(np.asarray(conf.growth_a), [0.0, 0.2, 0.8])
    np.testing.assert_allclose(np.asarray(conf.to_save_a), [0.25, 0.5, 1.0])
    np.testing.assert_array_equal(np.asarray(conf.slice_to_save), [0, 2, 4])
    assert conf.a_lpt_num == 3
    assert conf.a_nbody_num == 3
    assert conf.a_lpt_step <= 0.07
    assert conf.a_nbody_step <= 0.21 + 1e-12
    assert conf.ptcl_num == 24
    assert conf.ptcl_cell_vol == 2.5**3
    assert conf.box_size == (5.0, 7.5, 10.0)
    assert conf.box_vol == 375.0
    assert conf.cell_size == 1.25
    assert conf.disp_size == 0.8
    assert conf.mesh_size == 192
    assert conf.local_mesh_size == 192
    assert conf.V == conf.L / conf.T
    assert conf.H_0 == conf.H_0_SI * conf.T
    assert conf.c == conf.c_SI / conf.V
    assert conf.G == conf.G_SI * conf.M / (conf.L * conf.V**2)
    np.testing.assert_allclose(float(conf.rho_crit), 3 * conf.H_0**2 / (8 * np.pi * conf.G), rtol=1e-6)
    assert conf.transfer_k_num > 2
    assert conf.transfer_lgk_step <= conf.transfer_lgk_maxstep
    assert conf.transfer_k[0] == 0
    assert conf.transfer_k.shape == (conf.transfer_k_num, )
    assert conf.var_tophat.y.shape == conf.transfer_k[1:].shape
    assert conf.varlin_R.shape == conf.transfer_k[1:].shape


def test_configuration_single_device_fallback_attributes_and_unknown_names_are_explicit():
    conf = Configuration(1.0, (2, 2, 2), mesh_shape=2, pallas_cic=False)
    assert conf.use_mGPU is False
    assert conf.local_mesh_shape == conf.mesh_shape
    assert conf.local_mesh_with_halo_shape == conf.mesh_shape
    assert conf.mesh_halo_width == 0
    assert conf.ptcl_halo_width == 0
    assert conf.mGPU_scatter is None
    with pytest.raises(AttributeError, match="no attribute 'definitely_missing'"):
        _ = conf.definitely_missing


def test_nested_multigpu_seed_inherits_legacy_mesh_and_mode_only_when_missing():
    device = jax.devices("gpu")[:1]
    mesh = create_compute_mesh(device)
    inherited = Configuration(
        1.0, (2, 2, 2), mesh_shape=1, compute_mesh=mesh, multigpu_mode="particle_halo",
        multigpu=MultiGPUConfiguration(compute_mesh=None, mode=None), pallas_cic=False,
    )
    assert inherited.compute_mesh is mesh
    assert inherited.multigpu.mode == "particle_halo"

    explicit = Configuration(
        1.0, (2, 2, 2), mesh_shape=1, compute_mesh=mesh, multigpu_mode="particle_halo",
        multigpu=MultiGPUConfiguration(compute_mesh=mesh, mode="mesh_halo"), pallas_cic=False,
    )
    assert explicit.multigpu.mode == "mesh_halo"
    assert explicit.num_devices == 1


def test_configuration_custom_schedule_properties_fall_back_to_generated_grids():
    conf = Configuration(
        1.0, (2, 2, 2), mesh_shape=1, pallas_cic=False, a_start=0.1, a_stop=0.4, a_lpt_maxstep=0.04,
        a_nbody_maxstep=0.11,
    )
    assert conf.a_custom is None
    assert conf.growth_a_custom is None
    np.testing.assert_allclose(np.diff(np.asarray(conf.a_lpt)), conf.a_lpt_step, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.diff(np.asarray(conf.a_nbody)), conf.a_nbody_step, rtol=1e-6, atol=1e-7)
    expected_growth = np.concatenate((np.asarray(conf.a_lpt), np.asarray(conf.a_nbody)[1:]))
    np.testing.assert_allclose(np.asarray(conf.growth_a), expected_growth)
