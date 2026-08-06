"""Validate packed mesh-halo particle exchange and its collective count."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_worker() -> None:
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from pmpp.core import Configuration, pmid_to_idx
    from pmpp.distributed import (
        AXIS_NAME, MultiGPUConfiguration, _exchange_compacted_particles_packed, create_compute_mesh,
    )

    devices = jax.devices("cpu")
    assert len(devices) == 4
    mesh = create_compute_mesh(devices)
    conf = Configuration(
        1.0, (8, 8, 8), mesh_shape=1, multigpu=MultiGPUConfiguration(compute_mesh=mesh, mode="mesh_halo"),
        max_ptcl_per_slice=16, max_share_ptcl=8, max_halo_share_ptcl=8, max_share_gather_ptcl=32,
    )
    perm = tuple((index, (index - 1) % len(devices)) for index in range(len(devices)))
    shard_capacity = 8
    size = len(devices) * shard_capacity

    ids = jnp.arange(size, dtype=jnp.int32) % conf.mesh_size
    pmid = jnp.stack((ids // 64, (ids // 8) % 8, ids % 8), axis=-1, ).astype(conf.pmid_dtype)
    valid = (jnp.arange(size) % shard_capacity) < 6
    keys = jnp.where(valid, pmid_to_idx(pmid, conf), conf.mesh_size)
    base = jnp.arange(size * 3, dtype=conf.float_dtype).reshape(size, 3)

    def check_exchange(payload: tuple[jax.Array, ...]) -> None:
        compacted = (keys, pmid, *payload, valid)
        specs = (P(AXIS_NAME), ) * len(compacted)

        def packed_local(*values):
            return _exchange_compacted_particles_packed(values, perm, conf)

        reference = jax.jit(
            shard_map(
                lambda *values: jax.lax.ppermute(values, axis_name=AXIS_NAME, perm=perm), mesh=mesh, in_specs=specs,
                out_specs=specs, check_rep=False,
            )
        )
        packed = jax.jit(shard_map(packed_local, mesh=mesh, in_specs=specs, out_specs=specs, check_rep=False, ))

        reference_result = reference(*compacted)
        packed_result = packed(*compacted)
        for expected, actual in zip(reference_result, packed_result):
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))

        packed_hlo = packed.lower(*compacted).compiler_ir(dialect="hlo").as_hlo_text()
        assert len(re.findall(r"collective-permute\(", packed_hlo)) == 2

    check_exchange((base, base + 1_000))
    check_exchange((base, base + 1_000, base + 2_000))


def test_mesh_halo_packed_exchange_matches_tuple_exchange() -> None:
    """Run in an isolated process so four simulated CPU devices are reliable."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    pythonpath = [str(REPO_ROOT / "src"), str(REPO_ROOT), str(REPO_ROOT / "tests")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=REPO_ROOT, env=env, check=True, )


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("expected --worker")
    _run_worker()
