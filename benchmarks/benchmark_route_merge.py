"""Isolate native merge time with 98 percent stays and two 1 percent arrivals."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from pmpp.distributed import cuda


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--particles", type=int, default=1048576)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert cuda._register_targets(strict=True)
    count = args.particles
    rows = jnp.arange(count, dtype=jnp.int32)
    pmid = jnp.stack((rows // (128 * 128), rows // 128 % 128, rows % 128), axis=1).astype(jnp.int16)
    disp = jnp.sin(pmid.astype(jnp.float32))
    vel = disp + .01
    x_mod = jnp.where(rows % 100 == 0, 16., jnp.where(rows % 100 == 1, 80., 48.)).astype(jnp.float32)
    records = jax.jit(
        lambda: cuda.route_pack_bidir_cuda(
            pmid, disp, vel, jnp.ones(count, dtype=jnp.uint8), x_mod, global_nmesh=128, mesh_shape=(128, 128, 128),
            owned_start=32, owned_end=64, slice_width=32, num_devices=4, capacity=count // 100 + 1, stay_capacity=count
        )
    )()
    inputs = (pmid, disp, vel, records[5], records[6], records[0], records[2], records[1], records[3])
    result = {
        "particles": count,
        "jax": jax.__version__,
        "build": cuda.extension_status()["build_identifier"],
        "phases": {}
    }
    for name, fn in (("primal", cuda.route_merge_bidir_primal_i16), ("metadata", cuda.route_merge_bidir_cuda)):
        compiled = jax.jit(lambda *a: fn(*a, mesh_shape=(128, 128, 128), capacity=int(count * 1.4))).lower(*inputs
                                                                                                           ).compile()
        for _ in range(3):
            out = jax.block_until_ready(compiled(*inputs))
        np.testing.assert_array_equal(np.asarray(out[0])[:count], np.asarray(pmid))
        np.testing.assert_array_equal(np.asarray(out[1])[:count], np.asarray(disp))
        assert int(out[-1]) == count
        samples = []
        while len(samples) < 30 or sum(samples) < 2.:
            start = time.perf_counter()
            jax.block_until_ready(compiled(*inputs))
            samples.append(time.perf_counter() - start)
        result["phases"][name] = {"median_seconds": float(np.median(samples)), "seconds": samples}
        print(name, result["phases"][name]["median_seconds"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
