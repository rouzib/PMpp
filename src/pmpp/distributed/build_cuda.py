"""Build the optional CUDA routing extension from an installed PM++ wheel."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

from ._cuda_paths import package_cuda_directory, user_cache_cuda_directory

_FALLBACK_ARCHITECTURES = "80;86;90;90-virtual"
_RECORD_FORMAT_VERSION = 2


def _supported_jax_version(version: str) -> bool:
    """Return whether a JAX package version supports the required FFI floor."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        return False
    major, minor, patch = (int(part or 0) for part in match.groups())
    return (major, minor, patch) >= (0, 6, 0)


def _query(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_compute_capabilities(output: str | None) -> tuple[str, ...]:
    """Parse ``nvidia-smi`` compute capabilities into CMake architecture IDs."""
    if not output:
        return ()
    architectures: set[str] = set()
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", line)
        if match:
            architectures.add(f"{int(match.group(1))}{int(match.group(2))}")
    return tuple(sorted(architectures, key=int))


def detect_cuda_architectures() -> str:
    """Detect visible GPU architectures and retain PTX for the newest one."""
    output = _query(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", ])
    detected = _parse_compute_capabilities(output)
    if not detected:
        print(
            "WARNING: GPU compute capability detection failed; building the "
            f"portable PM++ set {_FALLBACK_ARCHITECTURES}.", file=sys.stderr,
        )
        return _FALLBACK_ARCHITECTURES
    return ";".join((*detected, f"{detected[-1]}-virtual"))


def _source_directory() -> Path:
    candidates = (Path(__file__).resolve().parent / "cuda", Path(__file__).resolve().parents[3] / "cuda", )
    required = ("build_cuda_routing.py", "CMakeLists.txt", "route_kernels.cu")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"PM++ CUDA build sources are missing; searched: {searched}")


def _ensure_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".pmpp-write-test-", dir=directory):
        pass


def _select_target_directory(explicit: Path | None) -> tuple[Path, bool]:
    if explicit is not None:
        target = explicit.expanduser().resolve()
        _ensure_writable(target)
        return target, False

    package_target = package_cuda_directory()
    try:
        _ensure_writable(package_target)
    except OSError:
        cache_target = user_cache_cuda_directory()
        _ensure_writable(cache_target)
        return cache_target, True
    return package_target, False


def _preflight() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("CUDA routing can currently be compiled only on Linux or WSL2")
    missing = [tool for tool in ("cmake", "nvcc") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "missing required CUDA build tools: " + ", ".join(missing) +
            ". Install CMake 3.24+ and a CUDA toolkit that provides nvcc."
        )
    try:
        jax_version = metadata.version("jax")
        jaxlib_version = metadata.version("jaxlib")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("JAX and jaxlib must be installed before building") from error
    if not _supported_jax_version(jax_version) or not _supported_jax_version(jaxlib_version):
        raise RuntimeError(
            "CUDA routing requires JAX and jaxlib 0.6.0 or newer, but this "
            f"environment has JAX {jax_version} and jaxlib {jaxlib_version}."
        )


def _existing_artifact_matches(target: Path, architectures: str) -> bool:
    libraries = (target / "libpmpp_cuda_routing.so", target / "pmpp_cuda_routing.so", )
    manifest_path = target / "pmpp_cuda_routing.manifest.json"
    if not any(path.is_file() for path in libraries) or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pmpp_version = metadata.version("pmpp")
        jaxlib_version = metadata.version("jaxlib")
    except (OSError, ValueError, metadata.PackageNotFoundError):
        return False
    return (
        int(manifest.get("record_format_version", -1)) == _RECORD_FORMAT_VERSION
        and manifest.get("pmpp_version") == pmpp_version and manifest.get("jaxlib_version") == jaxlib_version
        and tuple(manifest.get("embedded_cuda_architectures",
                               ())) == tuple(part for part in architectures.split(";") if part)
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _build(*, source_dir: Path, target_dir: Path, architectures: str, dry_run: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="pmpp-cuda-routing-") as temporary:
        build_dir = Path(temporary)
        command = [
            sys.executable,
            str(source_dir / "build_cuda_routing.py"), "--build-dir",
            str(build_dir), "--python", sys.executable, "--cuda-architectures", architectures,
        ]
        if dry_run:
            print("Build command:", subprocess.list2cmdline(command))
            return
        subprocess.run(command, check=True)
        library = next((
            path for path in (build_dir / "libpmpp_cuda_routing.so", build_dir / "pmpp_cuda_routing.so",
                              ) if path.is_file()
        ), None,
                       )
        manifest = build_dir / "pmpp_cuda_routing.manifest.json"
        if library is None or not manifest.is_file():
            raise RuntimeError("CUDA build completed without a library and manifest")
        target_dir.mkdir(parents=True, exist_ok=True)
        _copy_atomic(library, target_dir / library.name)
        _copy_atomic(manifest, target_dir / manifest.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile CUDA routing for the current PM++ environment.")
    parser.add_argument(
        "--cuda-architectures", help="CMake CUDA architecture list; defaults to locally detected GPUs",
    )
    parser.add_argument(
        "--target-dir", type=Path, help="artifact destination; defaults to pmpp/_cuda with a cache fallback",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild even when the installed artifact matches this environment",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        _preflight()
        source_dir = _source_directory()
        target_dir, used_cache = _select_target_directory(args.target_dir)
        architectures = args.cuda_architectures or detect_cuda_architectures()
        if not args.force and _existing_artifact_matches(target_dir, architectures):
            print(f"CUDA routing is already current: {target_dir}")
            return 0

        print(f"CUDA source: {source_dir}")
        print(f"CUDA architectures: {architectures}")
        print(f"Artifact destination: {target_dir}")
        if used_cache:
            print("Package directory is read-only; using the PM++ user cache.")
        _build(source_dir=source_dir, target_dir=target_dir, architectures=architectures, dry_run=args.dry_run, )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not args.dry_run:
        print(f"CUDA routing installed in {target_dir}")
        print("Select it with cuda_routing=True in MultiGPUConfiguration.")
        print("For the fastest qualified backend, set:")
        print("  export PMPP_CUDA_ROUTING_BACKEND=bidir_mergepath")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
