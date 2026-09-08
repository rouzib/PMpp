"""Reproducible synchronized performance checks; run each variant in a fresh process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh
from pmpp.cosmology import SimpleLCDM, boltzmann
from pmpp.nbody import Particles, gravity
from pmpp.nbody.solver import nbody_init, nbody_step
from pmpp.nbody import nbody
from pmpp.cic import scatter
from pmpp.initial_conditions.lpt import _L_streaming_2lpt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--phases", nargs="+",
        choices=("fft_pullback", "force", "step", "step_grad", "nbody", "nbody_grad", "streaming_2lpt", "validation"),
        help="Select timing phases; use validation for only the final-state checks"
    )
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--nvtx", action="store_true", help="Mark warmed timing ranges for Nsight Systems")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nvtx = None
    if args.nvtx:
        import ctypes
        import ctypes.util
        nvtx = ctypes.CDLL(ctypes.util.find_library("nvToolsExt") or "libnvToolsExt.so.1")
        nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    mesh = create_compute_mesh(jax.devices()[:args.devices])
    assert mesh.size == args.devices
    n = args.size
    conf = Configuration(
        1., (n, ) * 3, mesh_shape=1, pmid_dtype=jnp.int16,
        multigpu=MultiGPUConfiguration(compute_mesh=mesh, mode="mesh_halo",
                                       cuda_routing=args.cuda), max_ptcl_per_slice=int(n**3 / args.devices * 1.4),
        max_share_ptcl=n * n * 4, a_start=.2, a_stop=.24, a_nbody_maxstep=.01,
    )
    if args.cuda:
        assert conf.cuda_routing, "Native benchmark must not silently fall back"
    ptcl = Particles.gen_grid(conf)
    # Positive sub-cell shifts keep the initial grid on its authoritative slab.
    disp = jnp.where(ptcl.unused_index[:, None], 0., .2 + .1 * jnp.sin(ptcl.pmid.astype(jnp.float32)))
    ptcl = ptcl.replace(disp=disp, vel=disp * .001)
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    dens = jax.jit(lambda p: scatter(p, conf))(ptcl)
    pot = conf.mGPU_rfftn_transposed(dens - 1)
    package_root = Path(__import__("pmpp").__file__).parent
    result = {
        "size": n,
        "devices": [str(d) for d in mesh.devices.flat],
        "jax": jax.__version__,
        "cuda": conf.cuda_routing,
        "workspace_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "density_mean": float(dens.mean()),
        "valid_particles": int((~ptcl.unused_index).sum()),
        "particle_capacity_per_device": conf.max_ptcl_per_slice,
        "migration_capacity_per_direction": conf.max_share_ptcl,
        "nbody_steps": conf.a_nbody_num,
        "package_source": str(__import__("pmpp").__file__),
        "source_sha256": {
            str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package_root.rglob("*.py"))
        },
        "phases": {}
    }
    if args.cuda:
        from pmpp.distributed.cuda import extension_status
        result["cuda_build_manifest"] = extension_status()["manifest"]

    def step(d):
        initial = nbody_init(.2, ptcl.replace(disp=d), cosmo, conf)
        final = nbody_step(.2, .21, initial, cosmo, conf)
        return jnp.concatenate((final.disp, final.vel), axis=-1)

    def simulation(d):
        final = nbody(ptcl.replace(disp=d), cosmo, conf)
        return jnp.concatenate((final.disp, final.vel), axis=-1)

    phases = {
        "fft_pullback": (lambda p: jax.vjp(conf.mGPU_rfftn_transposed, dens)[1](p)[0], pot),
        "force": (lambda d: gravity(.2, ptcl.replace(disp=d), cosmo, conf), disp),
        "step": (step, disp),
        "step_grad": (jax.grad(lambda d: jnp.sum(step(d)**2) * .5), disp),
        "nbody": (simulation, disp),
        "nbody_grad": (jax.grad(lambda d: jnp.sum(simulation(d)**2) * .5), disp),
        "streaming_2lpt": (lambda p: _L_streaming_2lpt(conf.kvec, p, conf), pot),
    }
    for name, (fn, value) in phases.items():
        if args.phases and name not in args.phases:
            continue
        print(f"Compiling {name}", flush=True)
        try:
            compiled = jax.jit(fn).lower(value).compile()
        except ValueError as error:
            if name != "step_grad" or "cannot be differentiated" not in str(error):
                raise
            result["phases"][name] = {"status": "unsupported", "error": str(error)}
            print(name, "unsupported by this implementation", flush=True)
            continue
        for _ in range(3):
            output = jax.block_until_ready(compiled(value))
        times = []
        if nvtx is not None:
            nvtx.nvtxRangePushA(("measure/" + name).encode())
        while len(times) < args.repeats or sum(times) < 1.:
            start = time.perf_counter()
            output = jax.block_until_ready(compiled(value))
            times.append(time.perf_counter() - start)
        if nvtx is not None:
            nvtx.nvtxRangePop()
        array = np.asarray(output)
        assert np.isfinite(array).all(), name
        memory = compiled.memory_analysis()
        if args.dump_hlo:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_name(args.output.stem + "_" + name + ".hlo").write_text(compiled.as_text())
        result["phases"][name] = {
            "median_seconds": float(np.median(times)),
            "seconds": times,
            "temporary_bytes": memory.temp_size_in_bytes,
            "argument_bytes": memory.argument_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes,
            "output_sharding": str(output.sharding),
            "l2": float(np.linalg.norm(array.astype(np.float64)))
        }
        print(name, {k: v for k, v in result["phases"][name].items() if k != "seconds"}, flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    if args.phases and not {"nbody", "validation"}.intersection(args.phases):
        return
    final = jax.jit(lambda p: nbody(p, cosmo, conf))(ptcl)
    final_density = jax.jit(lambda p: scatter(p, conf))(final)
    final_count = int((~final.unused_index).sum())
    mean_density = float(final_density.mean())
    assert final_count == n**3 and abs(mean_density - 1.) < 2e-6
    result["final_validation"] = {
        "valid_particles": final_count,
        "density_mean": mean_density,
        "finite_density": bool(jnp.isfinite(final_density).all())
    }
    assert result["final_validation"]["finite_density"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
