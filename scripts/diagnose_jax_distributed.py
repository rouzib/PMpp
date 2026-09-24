"""Isolate four-process JAX collectives without importing PM++.

Launch one process per device with SLURM_PROCID and
PMPP_COORDINATOR_ADDRESS set. The GPU launcher should expose one H100 per task.
"""

from __future__ import annotations

import argparse
import os

import jax
import jaxlib
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true", help="permit a logical CPU test")
    args = parser.parse_args()
    rank = int(os.environ["SLURM_PROCID"])
    jax.distributed.initialize(
        coordinator_address=os.environ["PMPP_COORDINATOR_ADDRESS"],
        num_processes=4,
        process_id=rank,
        local_device_ids=[0],
    )
    devices = jax.devices()
    local_devices = jax.local_devices()
    print(
        f"rank={rank} jax={jax.__version__} jaxlib={jaxlib.__version__} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')} "
        f"local_devices={local_devices} global_devices={devices}",
        flush=True,
    )
    if len(devices) != 4 or len(local_devices) != 1:
        raise RuntimeError("expected four global devices and one local device per rank")
    if not args.allow_cpu and any(device.platform != "gpu" for device in devices):
        raise RuntimeError("expected four GPU devices")

    mesh = Mesh(np.asarray(devices), ("gpus",))
    source = np.arange(64, dtype=np.float32).reshape(16, 4)
    sharding = NamedSharding(mesh, P("gpus", None))
    x = jax.make_array_from_process_local_data(
        sharding, source[rank * 4:(rank + 1) * 4], global_shape=source.shape,
    )

    ring = jax.jit(jax.shard_map(
        lambda local: jax.lax.ppermute(
            local, "gpus", perm=[(i, (i + 1) % 4) for i in range(4)],
        ),
        mesh=mesh, in_specs=P("gpus", None), out_specs=P("gpus", None),
    ))
    ring_local = np.asarray(ring(x).addressable_data(0))
    previous_rank = (rank - 1) % 4
    np.testing.assert_array_equal(ring_local, source[previous_rank * 4:(previous_rank + 1) * 4])
    print(f"rank {rank}: JAX ring ppermute passed", flush=True)

    all_to_all = jax.jit(jax.shard_map(
        lambda local: jax.lax.all_to_all(local, "gpus", split_axis=0, concat_axis=0),
        mesh=mesh, in_specs=P("gpus", None), out_specs=P("gpus", None),
    ))
    all_to_all_local = np.asarray(all_to_all(x).addressable_data(0))
    np.testing.assert_array_equal(all_to_all_local, source[rank::4])
    print(f"rank {rank}: JAX all-to-all passed", flush=True)


if __name__ == "__main__":
    main()
