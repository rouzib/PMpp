# Multi-node execution

PM++ can use a Slurm allocation in two different ways. Choose the one that
matches the scientific unit of work:

- **Independent simulations:** each node runs a complete, independent PM++
  simulation on its local GPUs. Partition seeds, parameter points, or other
  independent work across nodes. This is the appropriate pattern for a map or
  training-set campaign.
- **One distributed simulation:** JAX joins the processes into one global
  device set, and PM++ shards a single simulation across every GPU in the
  allocation. Use this when one simulation does not fit on a node or needs the
  aggregate accelerator count.

These patterns are not interchangeable. Starting several independent Python
processes does not make one PM++ calculation multi-node; conversely, every
process in a JAX-distributed run must execute the same collective program.

## Independent simulations on each node

The `scripts/pmpp_1024_maps.sh` and `scripts/pmpp_1024_maps.py` workflow is an
example of throughput scaling. Slurm starts one Python process per node. Each
process sees its four local GPUs and uses `SLURM_NODEID` to select a disjoint
subset of the ordered seed list.

```bash
#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=10G
#SBATCH --account=<account>
#SBATCH --chdir=<repository>
#SBATCH --output=<scratch>/pmpp_maps_%j.out

module load gcc cuda/<version> nccl/<version> python/<version>
source <virtualenv>/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
  <virtualenv>/bin/python scripts/pmpp_1024_maps.py \
  --seed-start 24 --seed-end 2616
```

In the worker, use the Slurm node number, not the process-local GPU index, to
partition the campaign. Round-robin assignment keeps the work balanced when
individual seeds have different runtimes.

```python
import os

def seeds_for_this_node(seeds):
    node_id = int(os.environ.get("SLURM_NODEID", "0"))
    n_nodes = int(
        os.environ.get("SLURM_NNODES")
        or os.environ.get("SLURM_JOB_NUM_NODES")
        or "1"
    )
    return [seed for index, seed in enumerate(seeds) if index % n_nodes == node_id]
```

Do not call `jax.distributed.initialize` for this pattern. Construct the
PM++ device mesh from the local visible GPUs, as in the [Multi-GPU
execution](multigpu.md) guide. Give every output a seed-specific name so that
two nodes never write the same path. A retrying `srun` only retries the full
assigned subset; make the output writer idempotent, or have it skip already
completed seeds, before enabling retries.

## One simulation across multiple nodes

For a single calculation spanning nodes, launch one process per node and make
JAX discover the Slurm allocation before querying any devices. After
initialization, `jax.devices()` is the global device set, whereas
`jax.local_devices()` contains only the GPUs attached to the current node.
Pass the global set to `create_compute_mesh`.

```python
import jax

jax.distributed.initialize(cluster_detection_method="slurm")

from pmpp import Configuration, MultiGPUConfiguration
from pmpp.distributed import create_compute_mesh

print(
    f"process {jax.process_index()} of {jax.process_count()}; "
    f"local devices={jax.local_device_count()}; "
    f"global devices={jax.device_count()}"
)

devices = [device for device in jax.devices() if device.platform == "gpu"]
if not devices:
    raise RuntimeError("No global GPU devices were discovered")

conf = Configuration(
    ptcl_spacing=<particle_spacing>,
    ptcl_grid_shape=(<n>, <n>, <n>),
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(devices),
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=<validated_particle_capacity>,
    max_share_ptcl=<validated_migration_capacity>,
    max_halo_share_ptcl=<validated_halo_capacity>,
    max_share_gather_ptcl=<validated_gather_capacity>,
)
```

The corresponding Slurm header requests one process for each node, not one
process for each GPU. JAX uses the GPUs visible to that node inside the
process.

```bash
#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=30G
#SBATCH --account=<account>
#SBATCH --chdir=<repository>
#SBATCH --output=<scratch>/pmpp_multinode_%j.out

module load gcc cuda/<version> nccl/<version> python/<version>
source <virtualenv>/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export PYTHONUNBUFFERED=1

srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
  <virtualenv>/bin/python scripts/run_multinode.py --seed 0
```

All ranks must reach the same PM++ and JAX collectives in the same order. Do
not partition seeds by `SLURM_NODEID` in this mode, and do not conditionally
skip a collective on a nonzero process. It is fine to restrict logging and
filesystem writes to rank zero:

```python
if jax.process_index() == 0:
    save_result(result)
```

Wait for the result before timing or saving it, for example with
`jax.block_until_ready(result)`. A process failure, shape mismatch, or a rank
taking a different branch can leave the other ranks waiting in a collective;
`--kill-on-bad-exit=1` ensures Slurm stops the remaining tasks instead of
leaving an unusable allocation running.