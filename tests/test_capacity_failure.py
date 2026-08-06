import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from pmpp.core import raise_error
from pmpp.nbody import _assert_halo_move_succeeded

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compiled_capacity_failure_raises_instead_of_continuing():

    @jax.jit
    def checked(count):

        def fail(value):
            raise_error(
                "[ERROR] static capacity overflow: required={x}, capacity={y}", x=value,
                y=jnp.asarray(1, dtype=value.dtype),
            )
            return value

        return jax.lax.cond(count > 1, fail, lambda value: value, count)

    assert int(checked(jnp.asarray(1, dtype=jnp.int32))) == 1
    with pytest.raises(Exception, match="static capacity overflow"):
        checked(jnp.asarray(2, dtype=jnp.int32)).block_until_ready()


def test_reported_halo_move_failure_raises_instead_of_being_discarded():
    checked = jax.jit(_assert_halo_move_succeeded)

    checked(jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32)).block_until_ready()
    with pytest.raises(Exception, match="Particle migration reported"):
        checked(jnp.asarray(True), jnp.asarray(7, dtype=jnp.int32)).block_until_ready()


def _run_routing_overflow_worker():
    """Trigger a true mesh-halo migration overflow inside shard_map."""

    import itertools

    from pmpp.core import Configuration
    from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh

    devices = jax.devices("cpu")
    assert len(devices) == 2
    conf = Configuration(
        1.0, (4, 4, 4), mesh_shape=1, multigpu=MultiGPUConfiguration(compute_mesh=create_compute_mesh(devices)),
        max_ptcl_per_slice=32, max_share_ptcl=1, max_halo_share_ptcl=8, max_share_gather_ptcl=16,
    )
    # Each logical shard owns 32 positions. Only the first shard has two
    # particles cross its wrapped left boundary, exceeding the one-row routing
    # buffer. The mover must synchronize that failure before its ppermute.
    pmid = jnp.asarray(
        list(itertools.product(range(2), range(4), range(4))) +
        list(itertools.product(range(2, 4), range(4), range(4))), dtype=conf.pmid_dtype,
    )
    disp = jnp.zeros((64, 3), dtype=conf.float_dtype).at[:2, 0].set(-1.0)
    vel = jnp.zeros_like(disp)
    acc = jnp.zeros_like(disp)
    unused = jnp.zeros((64, ), dtype=jnp.bool_)
    routed = jax.jit(conf.mGPU_halo_moving)
    try:
        routed(pmid, disp, disp, vel, acc, conf.halo_start, conf.halo_end, unused)[-1].block_until_ready()
    except Exception as exc:
        if "Exceeded migration share capacity" in str(exc):
            return
        raise
    raise AssertionError("expected real mesh-halo routing overflow to raise")


def test_mesh_halo_routing_overflow_fails_closed():
    """Run with two simulated CPUs in a fresh process for stable device setup."""

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    paths = [str(REPO_ROOT / "src"), str(REPO_ROOT), str(REPO_ROOT / "tests")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--routing-overflow-worker"], cwd=REPO_ROOT, env=env,
                   check=True,
                   )


if __name__ == "__main__":
    if sys.argv[1:] != ["--routing-overflow-worker"]:
        raise SystemExit("expected --routing-overflow-worker")
    _run_routing_overflow_worker()
