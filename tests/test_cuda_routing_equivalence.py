"""Native equivalence tests for every production particle-routing backend."""

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh
from pmpp.distributed import cuda as cuda_routing
from pmpp.nbody import Particles


def _required_on_this_worker():
    return os.environ.get("PMPP_REQUIRE_CUDA_ROUTING_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _require_native_backends(float_dtype):
    problems = []
    gpu_devices = jax.devices("gpu")
    if len(gpu_devices) < 2:
        problems.append(f"requires two GPUs, found {len(gpu_devices)}")

    try:
        registered = cuda_routing._register_targets(strict=_required_on_this_worker())
    except Exception as exc:
        problems.append(f"CUDA FFI registration failed: {exc!r}")
        registered = False

    status = cuda_routing.extension_status()
    if not registered or not status["registered"]:
        problems.append("cuda_merge targets are not registered")
    if not status["bidir_registered"]:
        problems.append("bidir_mergepath targets are not registered")
    if jnp.dtype(float_dtype) == jnp.float64:
        if not status["float64_registered"]:
            problems.append("float64 cuda_merge targets are not registered")
        if not status["float64_bidir_registered"]:
            problems.append("float64 bidir_mergepath targets are not registered")

    if problems:
        message = "; ".join(dict.fromkeys(problems))
        if _required_on_this_worker():
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    return gpu_devices[:2], status


def _configuration(compute_mesh, float_dtype, *, cuda, backend="bidir_mergepath"):
    return Configuration(
        1.0, (4, 4, 4), mesh_shape=1, float_dtype=float_dtype, pallas_cic=False, multigpu=MultiGPUConfiguration(
            compute_mesh=compute_mesh, mode="mesh_halo", cuda_routing=cuda, cuda_routing_backend=backend,
        ), max_ptcl_per_slice=64, max_share_ptcl=24, max_halo_share_ptcl=40, max_share_gather_ptcl=24,
    )


def _adversarial_prestate(conf, float_dtype):
    single = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=float_dtype, pallas_cic=False)
    grid = Particles.gen_grid(single)
    particle_id = jnp.arange(single.ptcl_num, dtype=float_dtype)

    # x=0 wraps from GPU 0 to GPU 1; x=2 crosses from GPU 1 to GPU 0.
    # Remaining rows stay local, and the fixed per-device capacity contributes
    # invalid padding. The transverse values make payload swaps detectable.
    velocity = jnp.stack((
        jnp.where((grid.pmid[:, 0] == 0) |
                  (grid.pmid[:, 0] == 2), jnp.asarray(-0.35, float_dtype), jnp.asarray(0.08, float_dtype),
                  ), (jnp.sin(particle_id) * jnp.asarray(0.013, float_dtype)),
        (jnp.cos(particle_id) * jnp.asarray(-0.017, float_dtype)),
    ), axis=1)
    acceleration = jnp.stack((
        particle_id + jnp.asarray(0.25, float_dtype), particle_id * jnp.asarray(-2.0, float_dtype) +
        jnp.asarray(1.5, float_dtype), particle_id * particle_id * jnp.asarray(0.01, float_dtype),
    ), axis=1)
    particles = Particles.from_pmid(conf, grid.pmid, grid.disp, vel=velocity, acc=acceleration)
    return particles, particles.disp + particles.vel


def _run_route_and_pullback(conf, particles, displacement_after):
    forward = conf.mGPU_halo_moving_no_acc(
        particles.pmid, particles.disp, displacement_after, particles.vel, conf.halo_start, conf.halo_end,
        particles.unused_index,
    )
    slot = jnp.arange(particles.disp.size, dtype=conf.float_dtype).reshape(particles.disp.shape)
    displacement_cotangent = (slot + jnp.asarray(0.5, conf.float_dtype)) / jnp.asarray(97, conf.float_dtype)
    velocity_cotangent = jnp.flip(displacement_cotangent, axis=0) * jnp.asarray(-1.75, conf.float_dtype)
    acceleration_cotangent = jnp.sin(displacement_cotangent * jnp.asarray(3.0, conf.float_dtype))
    pullback = conf.mGPU_halo_move_pullback(
        particles.pmid, particles.disp, displacement_after, particles.vel, particles.acc, conf.halo_end,
        particles.unused_index, displacement_cotangent, velocity_cotangent, acceleration_cotangent,
    )
    return forward, pullback


def _assert_tree_bitwise_equal(actual, expected, *, actual_name, expected_name):
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    assert actual_tree == expected_tree
    assert len(actual_leaves) == len(expected_leaves)
    for leaf_index, (actual_leaf, expected_leaf) in enumerate(zip(actual_leaves, expected_leaves)):
        np.testing.assert_array_equal(
            np.asarray(actual_leaf), np.asarray(expected_leaf),
            err_msg=f"leaf {leaf_index}: {actual_name} disagrees with {expected_name}",
        )


@pytest.mark.gpu2
@pytest.mark.parametrize("float_dtype", (jnp.float32, jnp.float64))
def test_native_cuda_routes_match_each_other_and_canonical_jax_forward_and_pullback(float_dtype):
    devices, status = _require_native_backends(float_dtype)
    compute_mesh = create_compute_mesh(devices)
    jax_conf = _configuration(compute_mesh, float_dtype, cuda=False)
    cuda_merge_conf = _configuration(compute_mesh, float_dtype, cuda=True, backend="cuda_merge")
    bidir_conf = _configuration(compute_mesh, float_dtype, cuda=True, backend="bidir_mergepath")

    assert not jax_conf.cuda_routing
    assert cuda_merge_conf.cuda_routing, status
    assert cuda_merge_conf.cuda_routing_backend == "cuda_merge"
    assert bidir_conf.cuda_routing, status
    assert bidir_conf.cuda_routing_backend == "bidir_mergepath"

    particles, displacement_after = _adversarial_prestate(jax_conf, float_dtype)
    jax_result = _run_route_and_pullback(jax_conf, particles, displacement_after)
    cuda_merge_result = _run_route_and_pullback(cuda_merge_conf, particles, displacement_after)
    bidir_result = _run_route_and_pullback(bidir_conf, particles, displacement_after)

    _assert_tree_bitwise_equal(cuda_merge_result, jax_result, actual_name="cuda_merge", expected_name="canonical JAX", )
    _assert_tree_bitwise_equal(bidir_result, jax_result, actual_name="bidir_mergepath", expected_name="canonical JAX", )
    _assert_tree_bitwise_equal(
        bidir_result, cuda_merge_result, actual_name="bidir_mergepath", expected_name="cuda_merge",
    )
