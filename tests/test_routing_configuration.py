"""Configuration rules for the canonical mesh-halo routing path."""

import jax
import jax.numpy as jnp
import pytest

from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, build_multigpu_configuration, create_compute_mesh


def _base_configuration():
    return Configuration(1.0, (4, 4, 4), mesh_shape=1)


def _resolve(seed):
    return build_multigpu_configuration(
        _base_configuration(), seed.replace(compute_mesh=create_compute_mesh(jax.devices()[:1])),
    )


def test_routing_uses_the_single_canonical_path():
    default = _resolve(MultiGPUConfiguration())
    assert default.mode == "mesh_halo"
    assert not hasattr(default, "linear_particle_merge")
    assert not hasattr(default, "chunked_exchange")


def test_pallas_cic_is_enabled_by_default():
    conf = _base_configuration()
    assert conf.pallas_cic is True


def test_cuda_routing_none_resolves_to_automatic_selection():
    resolved = _resolve(MultiGPUConfiguration(cuda_routing=None))
    assert isinstance(resolved.cuda_routing, bool)


def test_unsupported_pallas_configuration_warns_before_jit():
    with pytest.warns(RuntimeWarning, match="Pallas CIC was requested"):
        Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float64, )
