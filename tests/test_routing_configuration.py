"""Configuration rules for the canonical mesh-halo routing path."""

import jax
import jax.numpy as jnp
import pytest

import pmpp.distributed.configuration as distributed_configuration
from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, build_multigpu_configuration, create_compute_mesh
from pmpp.distributed.cuda import requested_backend


@pytest.fixture(autouse=True)
def _clear_cuda_routing_backend_override(monkeypatch):
    monkeypatch.delenv("PMPP_CUDA_ROUTING_BACKEND", raising=False)


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
    assert resolved.cuda_routing_backend == "bidir_mergepath"


def test_cuda_routing_backend_defaults_to_bidir_mergepath():
    seed = MultiGPUConfiguration()
    assert seed.cuda_routing_backend == "bidir_mergepath"
    assert requested_backend() == "bidir_mergepath"
    assert requested_backend(seed) == "bidir_mergepath"


def test_cuda_routing_backend_can_select_cuda_merge():
    resolved = _resolve(MultiGPUConfiguration(cuda_routing_backend="cuda_merge"))
    assert resolved.cuda_routing_backend == "cuda_merge"
    assert requested_backend(resolved) == "cuda_merge"


def test_cuda_routing_backend_uses_backend_specific_qualification(monkeypatch):
    monkeypatch.setattr(distributed_configuration, "cuda_bidir_routing_supported", lambda *args, **kwargs: True)
    monkeypatch.setattr(distributed_configuration, "cuda_routing_supported", lambda *args, **kwargs: False)
    bidir = _resolve(MultiGPUConfiguration(cuda_routing=True))
    assert bidir.cuda_routing is True

    monkeypatch.setattr(distributed_configuration, "cuda_bidir_routing_supported", lambda *args, **kwargs: False)
    monkeypatch.setattr(distributed_configuration, "cuda_routing_supported", lambda *args, **kwargs: True)
    cuda_merge = _resolve(MultiGPUConfiguration(cuda_routing=True, cuda_routing_backend="cuda_merge"))
    assert cuda_merge.cuda_routing is True


def test_cuda_routing_backend_environment_override_is_preserved(monkeypatch):
    monkeypatch.setenv("PMPP_CUDA_ROUTING_BACKEND", "current")
    resolved = _resolve(MultiGPUConfiguration())
    assert resolved.cuda_routing_backend == "cuda_merge"


def test_cuda_routing_backend_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unsupported cuda_routing_backend"):
        _resolve(MultiGPUConfiguration(cuda_routing_backend="unknown"))


def test_configuration_static_pytree_preserves_initialized_instance():
    conf = _base_configuration()
    leaves, treedef = jax.tree_util.tree_flatten(conf)

    assert leaves == []
    assert jax.tree_util.tree_unflatten(treedef, leaves) is conf
    assert conf.kvec


def test_unsupported_pallas_configuration_warns_before_jit():
    with pytest.warns(RuntimeWarning, match="Pallas CIC was requested"):
        Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float64, )
