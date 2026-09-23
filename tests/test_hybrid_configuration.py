"""Configuration gates for opt-in exceptional particle migration."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from src.pmpp.core.utils import AXIS_NAME
from src.pmpp.core.configuration import Configuration
from src.pmpp.distributed import configuration as distributed_configuration
from src.pmpp.distributed.configuration import MultiGPUConfiguration, build_multigpu_configuration


def _base_conf():
    return SimpleNamespace(
        compute_mesh=None, multigpu_mode="mesh_halo", replicated_mesh=False,
        static_mesh_halo_width=0, mesh_shape=(16, 16, 16),
        ptcl_grid_shape=(16, 16, 16), ptcl_num=16**3,
        max_ptcl_per_slice=32, max_share_ptcl=8, max_halo_share_ptcl=4,
        max_share_gather_ptcl=4, cell_size=1.0, float_dtype=jnp.float32,
        pmid_dtype=jnp.int16,
    )


def _seed(devices, **kwargs):
    return MultiGPUConfiguration(
        compute_mesh=Mesh(np.asarray(devices), (AXIS_NAME,)),
        cuda_routing=True, migration_policy="hybrid", **kwargs,
    )


def test_hybrid_resolves_explicit_capacities_and_replacement(monkeypatch):
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    monkeypatch.setattr(distributed_configuration, "cuda_bidir_routing_supported", lambda *a, **k: True)
    monkeypatch.setattr(distributed_configuration, "cuda_hybrid_routing_supported", lambda *a, **k: True)
    runtime = build_multigpu_configuration(
        _base_conf(), _seed(devices[:4], far_send_capacity=8,
                            far_recv_capacity=12, far_chunk_size=4),
    )
    assert runtime.migration_policy == "hybrid"
    assert (runtime.far_send_capacity, runtime.far_recv_capacity, runtime.far_chunk_size) == (8, 12, 4)
    assert runtime.replace(far_recv_capacity=10).far_recv_capacity == 10


def test_configuration_replace_preserves_hybrid_policy_and_capacities(monkeypatch):
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    monkeypatch.setattr(distributed_configuration, "cuda_bidir_routing_supported", lambda *a, **k: True)
    monkeypatch.setattr(distributed_configuration, "cuda_hybrid_routing_supported", lambda *a, **k: True)
    conf = Configuration(
        1.0, (4, 4, 4), mesh_shape=1, pallas_cic=False,
        multigpu=_seed(devices[:4], far_send_capacity=8,
                       far_recv_capacity=12, far_chunk_size=4),
        max_ptcl_per_slice=16, max_share_ptcl=8,
        max_halo_share_ptcl=4, max_share_gather_ptcl=4,
    )
    replaced = conf.replace(a_start=0.2)
    assert replaced.multigpu.migration_policy == "hybrid"
    assert (replaced.multigpu.far_send_capacity, replaced.multigpu.far_recv_capacity,
            replaced.multigpu.far_chunk_size) == (8, 12, 4)


@pytest.mark.parametrize("capacities", [
    dict(far_send_capacity=None, far_recv_capacity=4, far_chunk_size=2),
    dict(far_send_capacity=4, far_recv_capacity=0, far_chunk_size=2),
    dict(far_send_capacity=4, far_recv_capacity=4, far_chunk_size=5),
    dict(far_send_capacity=True, far_recv_capacity=4, far_chunk_size=2),
])
def test_hybrid_rejects_invalid_capacities(capacities):
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    with pytest.raises(ValueError, match="far_"):
        build_multigpu_configuration(_base_conf(), _seed(devices[:4], **capacities))


def test_hybrid_requires_new_native_abi(monkeypatch):
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    monkeypatch.setattr(distributed_configuration, "cuda_bidir_routing_supported", lambda *a, **k: True)
    monkeypatch.setattr(distributed_configuration, "cuda_hybrid_routing_supported", lambda *a, **k: False)
    with pytest.raises(ValueError, match="hybrid augmented-merge target"):
        build_multigpu_configuration(
            _base_conf(), _seed(devices[:4], far_send_capacity=8,
                                far_recv_capacity=8, far_chunk_size=2),
        )


def test_hybrid_rejects_particle_halo_mode():
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    with pytest.raises(ValueError, match="dynamic mesh_halo"):
        build_multigpu_configuration(
            _base_conf(), _seed(devices[:4], mode="particle_halo",
                                far_send_capacity=8, far_recv_capacity=8, far_chunk_size=2),
        )
