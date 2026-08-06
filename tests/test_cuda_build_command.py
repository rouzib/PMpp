"""Tests for the installed CUDA-routing build command."""

from pathlib import Path

from pmpp.distributed import build_cuda as build_cuda_routing, cuda as cuda_routing


def test_compute_capability_parser_deduplicates_and_sorts():
    assert build_cuda_routing._parse_compute_capabilities("8.6\n9.0\n8.6\n") == ("86", "90", )


def test_architecture_detection_adds_ptx_for_newest_gpu(monkeypatch):
    monkeypatch.setattr(build_cuda_routing, "_query", lambda command: "8.0\n8.6")
    assert build_cuda_routing.detect_cuda_architectures() == "80;86;86-virtual"


def test_architecture_detection_has_a_headless_fallback(monkeypatch):
    monkeypatch.setattr(build_cuda_routing, "_query", lambda command: None)
    assert build_cuda_routing.detect_cuda_architectures() == "80;86;90;90-virtual"


def test_loader_searches_the_configured_user_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("PMPP_CUDA_ROUTING_CACHE", str(tmp_path))
    candidates = cuda_routing._candidate_library_paths()
    assert tmp_path / "libpmpp_cuda_routing.so" in candidates
    assert tmp_path / "pmpp_cuda_routing.so" in candidates


def test_read_only_package_directory_falls_back_to_user_cache(monkeypatch, tmp_path):
    package_target = tmp_path / "package" / "_cuda"
    cache_target = tmp_path / "cache"
    monkeypatch.setattr(build_cuda_routing, "package_cuda_directory", lambda: package_target)
    monkeypatch.setattr(build_cuda_routing, "user_cache_cuda_directory", lambda: cache_target)
    original_ensure_writable = build_cuda_routing._ensure_writable

    def fake_ensure_writable(directory):
        if directory == package_target:
            raise PermissionError("read-only site-packages")
        original_ensure_writable(directory)

    monkeypatch.setattr(build_cuda_routing, "_ensure_writable", fake_ensure_writable)
    target, used_cache = build_cuda_routing._select_target_directory(None)
    assert target == cache_target
    assert used_cache is True


def test_cli_uses_detected_architecture_and_selected_target(monkeypatch, tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    captured = {}

    monkeypatch.setattr(build_cuda_routing, "_preflight", lambda: None)
    monkeypatch.setattr(build_cuda_routing, "_source_directory", lambda: source)
    monkeypatch.setattr(build_cuda_routing, "_select_target_directory", lambda explicit: (target, False), )
    monkeypatch.setattr(build_cuda_routing, "detect_cuda_architectures", lambda: "86;86-virtual", )
    monkeypatch.setattr(build_cuda_routing, "_existing_artifact_matches", lambda directory, architectures: False, )

    def fake_build(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(build_cuda_routing, "_build", fake_build)

    assert build_cuda_routing.main([]) == 0
    assert captured == {
        "source_dir": source,
        "target_dir": target,
        "architectures": "86;86-virtual",
        "dry_run": False,
    }


def test_wheel_configuration_ships_the_builder_sources():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'pmpp-build-cuda-routing = "pmpp.distributed:build_cuda_main"' in pyproject
    assert '"cuda/CMakeLists.txt" = "pmpp/distributed/cuda/CMakeLists.txt"' in pyproject
    assert '"cuda/build_cuda_routing.py" = "pmpp/distributed/cuda/build_cuda_routing.py"' in pyproject
    assert '"cuda/route_kernels.cu" = "pmpp/distributed/cuda/route_kernels.cu"' in pyproject
