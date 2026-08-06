"""Build the optional PM++ CUDA routing FFI shared library.

This script intentionally lives outside the Hatchling wheel build.  A normal
``pip install pmpp`` remains pure Python/JAX; users who have a compatible CUDA
toolchain can run this script and point PM++ at the resulting library with
``PMPP_CUDA_ROUTING_LIBRARY``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _query(command, *, cwd=None):
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_manifest(build_dir: Path, python: str, architectures: str):
    artifact = next(
        (candidate for candidate in (build_dir / "libpmpp_cuda_routing.so", build_dir / "pmpp_cuda_routing.so") if candidate.exists()),
        None,
    )
    jax_version = _query([python, "-c", "import jax; print(jax.__version__)"])
    jaxlib_version = _query([python, "-c", "import jaxlib; print(jaxlib.__version__)"])
    commit = _query(["git", "rev-parse", "HEAD"], cwd=ROOT)
    dirty = bool(_query(["git", "status", "--porcelain"], cwd=ROOT))
    artifact_hash = None
    if artifact is not None:
        artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "build_identifier": artifact_hash[:16] if artifact_hash else None,
        "compiler": _query(["nvcc", "--version"]),
        "cuda_toolkit": _query(["nvcc", "--version"]),
        "pmpp_commit": commit,
        "pmpp_dirty": dirty,
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "embedded_cuda_architectures": [part for part in architectures.split(";") if part],
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "record_format_version": 2,
        "record_words_by_dtype": {"float32": 8, "float64": 14},
        "particle_float_dtypes": ["float32", "float64"],
        "artifact": None if artifact is None else artifact.name,
        "artifact_sha256": artifact_hash,
        "routing_targets": [
            "pmpp_route_pack",
            "pmpp_route_merge",
            "pmpp_route_merge_aux",
            "pmpp_route_transpose_split",
            "pmpp_route_transpose_scatter",
            "pmpp_route_bidir_pack",
            "pmpp_route_merge_bidir",
            "pmpp_route_pack_f64",
            "pmpp_route_merge_f64",
            "pmpp_route_merge_aux_f64",
            "pmpp_route_transpose_split_f64",
            "pmpp_route_transpose_scatter_f64",
            "pmpp_route_bidir_pack_f64",
            "pmpp_route_merge_bidir_f64",
        ],
    }
    manifest_path = build_dir / "pmpp_cuda_routing.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "cuda" / "build")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--cuda-architectures", default="80;86;90;90-virtual")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if args.clean and build_dir.exists():
        if ROOT not in build_dir.parents:
            raise SystemExit(f"refusing to clean outside the repository: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "cuda"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={args.python}",
            f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_architectures}",
            f"-DPMPP_CUDA_ARCHITECTURES={args.cuda_architectures}",
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release"], check=True, cwd=ROOT, env=env)
    manifest = _write_manifest(build_dir, args.python, args.cuda_architectures)
    print(build_dir / "libpmpp_cuda_routing.so")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
