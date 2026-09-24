"""Four-process, one-GPU-per-process hybrid routing smoke test.

Run with PMPP_COORDINATOR_ADDRESS set to the batch node host:port. All four
processes must see the same PM++ CUDA library and JAX environment. A launcher
timeout is required so a collective-ordering bug cannot hang the allocation.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def main():
    rank = int(os.environ["SLURM_PROCID"])
    coordinator = os.environ["PMPP_COORDINATOR_ADDRESS"]
    jax.distributed.initialize(
        coordinator_address=coordinator, num_processes=4, process_id=rank,
        local_device_ids=[0],
    )
    from src.pmpp.core.configuration import Configuration
    from src.pmpp.core.utils import AXIS_NAME
    from src.pmpp.distributed.configuration import MultiGPUConfiguration
    from src.pmpp.distributed.cuda import extension_status

    devices = jax.devices("gpu")
    if len(devices) != 4 or jax.local_device_count() != 1:
        raise RuntimeError(f"expected four ranks with one GPU each: {devices}")
    mesh = Mesh(np.asarray(devices), (AXIS_NAME,))
    conf = Configuration(
        1.0, (16, 16, 16), mesh_shape=1, float_dtype=jnp.float32,
        multigpu=MultiGPUConfiguration(
            compute_mesh=mesh, mode="mesh_halo", cuda_routing=True,
            cuda_routing_backend="bidir_mergepath", migration_policy="hybrid",
            far_send_capacity=4, far_recv_capacity=4, far_chunk_size=2,
        ),
        max_ptcl_per_slice=8, max_share_ptcl=4,
        max_halo_share_ptcl=4, max_share_gather_ptcl=4,
    )
    # Configuration construction registers the native FFI targets. A status
    # query before that point reports a valid but not-yet-registered library.
    status = extension_status()
    if not status["hybrid_registered"]:
        raise RuntimeError(
            "qualified hybrid native ABI is unavailable after configuration: "
            f"library={status['library']}, hybrid_feature={status['hybrid_feature']}"
        )

    pmid = np.zeros((8, 3), np.int16)
    disp = np.zeros((8, 3), np.float32)
    vel = np.zeros((8, 3), np.float32)
    unused = np.ones(8, bool)
    for row in range(4):
        pmid[row] = (rank * 4 + row, rank, row)
        unused[row] = False
    if rank == 0:
        disp[0, 0] = 8  # Only this rank sends an exceptional particle.

    def global_array(local, tail):
        sharding = NamedSharding(mesh, P(AXIS_NAME, *([None] * len(tail))))
        return jax.make_array_from_process_local_data(
            sharding, local, global_shape=(4 * 8, *tail),
        )

    args = (
        global_array(pmid, (3,)), global_array(disp, (3,)),
        global_array(vel, (3,)), jnp.float32(0), global_array(unused, ()),
    )
    result = jax.jit(conf.mGPU_halo_moving_low_memory)(*args)
    jax.block_until_ready(result)
    if bool(result[5]) or int(result[7]):
        raise RuntimeError(f"rank {rank}: route reported failure: {result[5]}, {result[7]}")
    local_pmid = np.asarray(result[0].addressable_data(0))
    local_unused = np.asarray(result[4].addressable_data(0))
    expected = [(source * 4 + row, source, row)
                for source in range(4) for row in range(4)
                if (2 if source == 0 and row == 0 else source) == rank]
    expected.sort()
    np.testing.assert_array_equal(local_unused, [False] * len(expected) + [True] * (8 - len(expected)))
    np.testing.assert_array_equal(local_pmid[:len(expected)], np.asarray(expected, np.int16))
    print(f"rank {rank}: {len(expected)} authoritative particles; distributed hybrid PASS", flush=True)


if __name__ == "__main__":
    main()
