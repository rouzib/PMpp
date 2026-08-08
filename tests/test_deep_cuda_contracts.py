import json
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.distributed import _cuda_paths
from pmpp.distributed import build_cuda
from pmpp.distributed import cuda


class _FakeFFI:

    def __init__(self):
        self.calls = []

    def ffi_call(self, target, outputs):
        specs = outputs if isinstance(outputs, tuple) else (outputs, )

        def call(*args, **kwargs):
            self.calls.append((target, specs, args, kwargs))
            result = tuple(jnp.zeros(spec.shape, spec.dtype) for spec in specs)
            return result if isinstance(outputs, tuple) else result[0]

        return call


def _routing_arrays(dtype=jnp.float32):
    pmid = jnp.arange(12, dtype=jnp.int32).reshape(4, 3)
    disp = jnp.arange(12, dtype=dtype).reshape(4, 3)
    vel = disp + 1
    valid = jnp.asarray([True, True, False, True])
    x_mod = disp[:, 0]
    return pmid, disp, vel, valid, x_mod


def test_cuda_ffi_wrappers_preserve_all_shapes_dtypes_targets_and_attributes(monkeypatch):
    fake = _FakeFFI()
    monkeypatch.setattr(cuda.jax.ffi, "ffi_call", fake.ffi_call)
    monkeypatch.setattr(cuda, "_FLOAT64_REGISTERED", True)
    monkeypatch.setattr(cuda, "_FLOAT64_BIDIR_REGISTERED", True)
    monkeypatch.setattr(cuda, "_FUSED_PRIMAL_REGISTERED", True)

    for dtype, words, suffix in ((jnp.float32, 8, ""), (jnp.float64, 14, "_f64")):
        pmid, disp, vel, valid, x_mod = _routing_arrays(dtype)
        packed = cuda.route_pack(
            pmid, disp, vel, valid, x_mod, global_nmesh=8, mesh_shape=(8, 2, 1), owned_start=0, owned_end=4,
            slice_width=4, direction=-1, num_devices=2, capacity=3,
        )
        assert [value.shape for value in packed] == [(3, words), (), (4, )]
        assert [value.dtype for value in packed] == [jnp.uint32, jnp.int32, jnp.uint8]
        target, _, args, attrs = fake.calls[-1]
        assert target == f"pmpp_route_pack{suffix}"
        assert len(args) == 8
        assert attrs == {
            "global_nmesh": np.int32(8),
            "mesh_x": np.int32(8),
            "mesh_y": np.int32(2),
            "mesh_z": np.int32(1),
            "direction": np.int32(-1),
            "num_devices": np.int32(2),
            "capacity": np.int32(3),
        }

        bidir = cuda.route_pack_bidir_cuda(
            pmid, disp, vel, valid, x_mod, global_nmesh=8, mesh_shape=(8, 2, 1), owned_start=0, owned_end=4,
            slice_width=4, num_devices=2, capacity=3,
        )
        assert [value.shape for value in bidir] == [(3, words), (3, words), (), (), (4, ), (3, ), ()]
        assert fake.calls[-1][0] == f"pmpp_route_bidir_pack{suffix}"
        assert fake.calls[-1][-1]["stay_capacity"] == np.int32(3)

        pmid_i16 = pmid.astype(jnp.int16)
        bidir_i16 = cuda.route_pack_bidir_cuda(
            pmid_i16, disp, vel, valid, x_mod, global_nmesh=8, mesh_shape=(8, 2, 1), owned_start=0, owned_end=4,
            slice_width=4, num_devices=2, capacity=3,
        )
        assert bidir_i16[5].dtype == jnp.int32
        assert fake.calls[-1][0] == f"pmpp_route_bidir_pack{suffix}_i16"
        assert fake.calls[-1][2][0].dtype == jnp.int16

        bidir_large_stay = cuda.route_pack_bidir_cuda(
            pmid, disp, vel, valid, x_mod, global_nmesh=8, mesh_shape=(8, 2, 1), owned_start=0, owned_end=4,
            slice_width=4, num_devices=2, capacity=3, stay_capacity=4,
        )
        assert bidir_large_stay[5].shape == (4, )

        records = jnp.zeros((3, words), dtype=jnp.uint32)
        merged_bidir = cuda.route_merge_bidir_cuda(
            pmid, disp, vel, jnp.arange(4, dtype=jnp.int32), jnp.int32(3), records, jnp.int32(2), records, jnp.int32(1),
            mesh_shape=(8, 2, 1), capacity=4,
        )
        assert [value.shape for value in merged_bidir] == [(4, 3), (4, 3), (4, 3), (4, ), (4, ), (4, ), (4, 2), ()]
        assert fake.calls[-1][0] == f"pmpp_route_merge_bidir{suffix}"

        merged_bidir_i16 = cuda.route_merge_bidir_cuda(
            pmid_i16, disp, vel, jnp.arange(4, dtype=jnp.int32), jnp.int32(3), records, jnp.int32(2), records,
            jnp.int32(1), mesh_shape=(8, 2, 1), capacity=4,
        )
        assert merged_bidir_i16[0].dtype == jnp.int16
        assert fake.calls[-1][0] == f"pmpp_route_merge_bidir{suffix}_i16"
        assert fake.calls[-1][2][0].dtype == jnp.int16
        if dtype == jnp.float32:
            merged_primal = cuda.route_merge_bidir_primal_i16(
                pmid_i16, disp, vel, jnp.arange(4, dtype=jnp.int32), jnp.int32(3), records, jnp.int32(2), records,
                jnp.int32(1), mesh_shape=(8, 2, 1), capacity=4,
            )
            assert [value.shape for value in merged_primal] == [(4, 3), (4, 3), (4, 3), (4, ), ()]
            assert fake.calls[-1][0] == "pmpp_route_merge_bidir_primal_i16"

            packed_fused = cuda.route_pack_bidir_drift_primal_i16(
                pmid_i16, disp, vel, valid, jnp.float32(0.5), disp_size=1.0, global_nmesh=8, mesh_shape=(8, 2, 1),
                owned_start=jnp.int32(0), owned_end=jnp.int32(4), slice_width=4, num_devices=2, capacity=3,
            )
            assert [value.shape for value in packed_fused] == [(3, 8), (3, 8), (), (), (1, ), (), ()]
            assert fake.calls[-1][0] == "pmpp_route_bidir_drift_pack_primal_i16"
            assert fake.calls[-1][2][3].dtype == jnp.bool_
            merged_fused = cuda.route_merge_bidir_drift_primal_i16(
                pmid_i16, disp, vel, valid, jnp.float32(0.5), packed_fused[4], packed_fused[5], packed_fused[0],
                packed_fused[2], packed_fused[1], packed_fused[3], disp_size=1.0, global_nmesh=8, mesh_shape=(8, 2, 1),
                owned_start=jnp.int32(0), owned_end=jnp.int32(4), slice_width=4, num_devices=2, record_capacity=3,
                capacity=4,
            )
            assert [value.shape for value in merged_fused] == [(4, 3), (4, 3), (4, 3), (4, ), ()]
            assert merged_fused[3].dtype == jnp.bool_
            assert fake.calls[-1][0] == "pmpp_route_bidir_drift_merge_primal_i16"

        for auxiliary, expected_target, expected_length in ((False, f"pmpp_route_merge{suffix}", 4),
                                                            (True, f"pmpp_route_merge_aux{suffix}", 6),
                                                            ):
            merged = cuda.route_merge(
                pmid, disp, vel, valid, records, jnp.int32(2), mesh_shape=(8, 2, 1), capacity=4, auxiliary=auxiliary,
            )
            assert len(merged) == expected_length
            assert fake.calls[-1][0] == expected_target

        cot = jnp.arange(12, dtype=dtype).reshape(4, 3)
        split = cuda.route_transpose_split(
            cot, jnp.asarray([0, 1, 2, 3], jnp.uint8), jnp.arange(4, dtype=jnp.int32), auth_size=4, share_capacity=2,
        )
        assert [value.shape for value in split] == [(4, 3), (2, 3), (2, 3)]
        assert fake.calls[-1][0] == f"pmpp_route_transpose_split{suffix}"

        scattered = cuda.route_transpose_scatter(
            cot, cot[:2], cot[:2], jnp.arange(4), valid, jnp.arange(2), valid[:2], jnp.arange(2), valid[:2],
            auth_size=4, share_capacity=2,
        )
        assert scattered.shape == cot.shape
        assert scattered.dtype == dtype
        assert fake.calls[-1][0] == f"pmpp_route_transpose_scatter{suffix}"


def test_cuda_float_abi_rejects_mixed_unsupported_and_unregistered_payloads(monkeypatch):
    f32 = jnp.zeros((2, 3), jnp.float32)
    f64 = jnp.zeros((2, 3), jnp.float64)
    with pytest.raises(TypeError, match="share one dtype"):
        cuda._float_abi(f32, f64)
    with pytest.raises(TypeError, match="float32 or float64"):
        cuda._float_abi(jnp.zeros((2, 3), jnp.float16))
    with pytest.raises(TypeError, match="int16 or int32"):
        cuda._coordinate_abi(jnp.zeros((2, 3), jnp.uint16))

    monkeypatch.setattr(cuda, "_FLOAT64_REGISTERED", False)
    with pytest.raises(RuntimeError, match="no float64 ABI"):
        cuda._float_abi(f64)
    monkeypatch.setattr(cuda, "_FLOAT64_REGISTERED", True)
    monkeypatch.setattr(cuda, "_FLOAT64_BIDIR_REGISTERED", False)
    pmid, disp, vel, valid, x_mod = _routing_arrays(jnp.float64)
    with pytest.raises(RuntimeError, match="no float64 bidirectional ABI"):
        cuda.route_pack_bidir_cuda(
            pmid, disp, vel, valid, x_mod, global_nmesh=8, mesh_shape=(8, 1, 1), owned_start=0, owned_end=4,
            slice_width=4, num_devices=2, capacity=4,
        )
    with pytest.raises(RuntimeError, match="no float64 bidirectional ABI"):
        cuda.route_merge_bidir_cuda(
            pmid, disp, vel, jnp.zeros(4, jnp.int32), jnp.int32(0), jnp.zeros((4, 14), jnp.uint32), jnp.int32(0),
            jnp.zeros((4, 14), jnp.uint32), jnp.int32(0), mesh_shape=(8, 1, 1), capacity=4,
        )


def _reset_registration(monkeypatch):
    for name in (
        "_REGISTERED", "_BIDIR_REGISTERED", "_FLOAT64_REGISTERED", "_FLOAT64_BIDIR_REGISTERED",
        "_FUSED_PRIMAL_REGISTERED",
    ):
        monkeypatch.setattr(cuda, name, False)


def test_cuda_target_registration_qualifies_complete_typed_library(monkeypatch):
    _reset_registration(monkeypatch)
    targets = (
        cuda._CURRENT_TARGETS + cuda._BIDIR_TARGETS + cuda._FLOAT64_TARGETS + cuda._FLOAT64_BIDIR_TARGETS +
        cuda._FUSED_PRIMAL_TARGETS
    )
    library = SimpleNamespace(_name="qualified.so", **{target: object() for target in targets})
    registered = []
    monkeypatch.setattr(
        cuda, "_load_build_manifest", lambda: {
            "record_format_version": 3,
            "features": [cuda._FUSED_PRIMAL_FEATURE],
        },
    )
    monkeypatch.setattr(cuda, "_load_library", lambda: library)
    monkeypatch.setattr(cuda.jax.ffi, "pycapsule", lambda symbol: symbol)
    monkeypatch.setattr(
        cuda.jax.ffi, "register_ffi_target", lambda target, capsule, platform: registered.append((target, platform)),
    )

    assert cuda._register_targets(strict=True) is True
    assert [target for target, _ in registered] == list(targets)
    assert all(platform == "CUDA" for _, platform in registered)
    assert cuda._REGISTERED and cuda._BIDIR_REGISTERED and cuda._FLOAT64_REGISTERED and cuda._FLOAT64_BIDIR_REGISTERED
    assert cuda._FUSED_PRIMAL_REGISTERED
    assert cuda._register_targets() is True
    assert len(registered) == len(targets)

    monkeypatch.setattr(
        cuda, "_load_build_manifest", lambda: {
            "record_format_version": 3,
            "features": [cuda._FUSED_PRIMAL_FEATURE],
            "build_identifier": "abc",
            "embedded_cuda_architectures": ["80", "90"],
        }
    )
    status = cuda.extension_status()
    assert status["library"] == "qualified.so"
    assert status["registered"] and status["bidir_registered"]
    assert status["float64_registered"] and status["float64_bidir_registered"]
    assert status["fused_primal_registered"] and status["fused_primal_feature"]
    assert status["bidir_targets"] == cuda._BIDIR_TARGETS
    assert status["build_identifier"] == "abc"
    assert status["embedded_architectures"] == ("80", "90")


def test_cuda_registration_failures_are_optional_unless_strict(monkeypatch):
    _reset_registration(monkeypatch)
    monkeypatch.setattr(cuda, "_load_build_manifest", lambda: {"record_format_version": 1})
    assert cuda._register_targets() is False
    with pytest.raises(RuntimeError, match="incompatible record format"):
        cuda._register_targets(strict=True)

    monkeypatch.setattr(cuda, "_load_build_manifest", lambda: {"record_format_version": 2})
    assert cuda._register_targets() is False
    with pytest.raises(RuntimeError, match="incompatible record format"):
        cuda._register_targets(strict=True)

    monkeypatch.setattr(cuda, "_load_build_manifest", lambda: None)
    monkeypatch.setattr(cuda, "_load_library", lambda: None)
    assert cuda._register_targets() is False
    with pytest.raises(RuntimeError, match="missing its record format manifest"):
        cuda._register_targets(strict=True)

    monkeypatch.setattr(cuda, "_load_build_manifest", lambda: {"record_format_version": 3})
    assert cuda._register_targets() is False
    with pytest.raises(RuntimeError, match="library or JAX FFI is unavailable"):
        cuda._register_targets(strict=True)

    library = SimpleNamespace(**{target: object() for target in cuda._CURRENT_TARGETS})
    monkeypatch.setattr(cuda, "_load_library", lambda: library)
    monkeypatch.setattr(cuda.jax.ffi, "pycapsule", lambda symbol: symbol)
    monkeypatch.setattr(
        cuda.jax.ffi, "register_ffi_target", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert cuda._register_targets() is False
    with pytest.raises(ValueError, match="bad"):
        cuda._register_targets(strict=True)


def test_cuda_environment_manifest_and_library_discovery_are_failure_safe(tmp_path, monkeypatch):
    monkeypatch.delenv("PMPP_CUDA_ROUTING", raising=False)
    monkeypatch.delenv("PMPP_CUDA_ROUTING_LIBRARY", raising=False)
    monkeypatch.delenv("PMPP_CUDA_ROUTING_MANIFEST", raising=False)
    assert cuda._truthy_env("PMPP_CUDA_ROUTING", True) is True
    for value in ("0", " false ", "NO", "off"):
        monkeypatch.setenv("PMPP_CUDA_ROUTING", value)
        assert cuda._truthy_env("PMPP_CUDA_ROUTING", True) is False
    monkeypatch.setenv("PMPP_CUDA_ROUTING", "yes")
    assert cuda._truthy_env("PMPP_CUDA_ROUTING", False) is True

    explicit_library = tmp_path / "custom.so"
    monkeypatch.setenv("PMPP_CUDA_ROUTING_LIBRARY", str(explicit_library))
    assert cuda._candidate_library_paths()[0] == explicit_library
    assert len(cuda._candidate_library_paths()) == len(set(cuda._candidate_library_paths()))

    good_manifest = tmp_path / "good.json"
    good_manifest.write_text('{"record_format_version": 3}', encoding="utf-8")
    monkeypatch.setenv("PMPP_CUDA_ROUTING_MANIFEST", str(good_manifest))
    assert cuda._load_build_manifest() == {"record_format_version": 3}
    good_manifest.write_text("not-json", encoding="utf-8")
    assert cuda._load_build_manifest() is None

    monkeypatch.delenv("PMPP_CUDA_ROUTING_MANIFEST")
    monkeypatch.setattr(cuda, "_LIBRARY", None)
    monkeypatch.setenv("PMPP_CUDA_ROUTING", "0")
    assert cuda._load_library() is None

    monkeypatch.setattr(cuda, "_load_build_manifest", lambda: None)
    assert cuda.extension_status()["record_format_version"] is None

    explicit_library.write_bytes(b"placeholder")
    monkeypatch.setenv("PMPP_CUDA_ROUTING", "1")
    attempts = []

    def fake_cdll(path, mode):
        attempts.append((path, mode))
        if len(attempts) == 1:
            raise OSError("wrong ABI")
        return SimpleNamespace(_name=path)

    second = tmp_path / "second.so"
    second.write_bytes(b"placeholder")
    monkeypatch.setattr(cuda, "_candidate_library_paths", lambda: (explicit_library, second))
    monkeypatch.setattr(cuda.ctypes, "CDLL", fake_cdll)
    loaded = cuda._load_library()
    assert loaded._name == str(second)
    assert cuda._load_library() is loaded


def test_cuda_manifest_must_belong_to_the_selected_library(tmp_path, monkeypatch):
    monkeypatch.delenv("PMPP_CUDA_ROUTING_MANIFEST", raising=False)
    selected = tmp_path / "selected" / "libpmpp_cuda_routing.so"
    unrelated = tmp_path / "unrelated" / "libpmpp_cuda_routing.so"
    selected.parent.mkdir()
    unrelated.parent.mkdir()
    unrelated_manifest = unrelated.parent / "pmpp_cuda_routing.manifest.json"
    unrelated_manifest.write_text('{"record_format_version": 3}', encoding="utf-8")
    monkeypatch.setattr(cuda, "_LIBRARY", SimpleNamespace(_name=str(selected)))
    monkeypatch.setattr(cuda, "_candidate_library_paths", lambda: (selected, unrelated))

    assert cuda._load_build_manifest() is None
    selected_manifest = selected.parent / "pmpp_cuda_routing.manifest.json"
    selected_manifest.write_text('{"record_format_version": 3}', encoding="utf-8")
    assert cuda._load_build_manifest() == {"record_format_version": 3}


def _configuration(**overrides):
    values = {
        "float_dtype": jnp.float32,
        "pmid_dtype": jnp.int32,
        "num_devices": 2,
        "multigpu_mode": "mesh_halo",
        "mesh_shape": (8, 8, 8),
        "cuda_routing": True,
        "cuda_routing_backend": "bidir_mergepath",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cuda_configuration_gates_every_static_abi_invariant(monkeypatch):
    monkeypatch.setattr(cuda, "_qualified_jax", lambda: True)
    monkeypatch.setattr(cuda.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda, "_register_targets", lambda: True)
    monkeypatch.setattr(cuda, "_BIDIR_REGISTERED", True)
    monkeypatch.setattr(cuda, "_FLOAT64_REGISTERED", True)
    monkeypatch.setattr(cuda, "_FLOAT64_BIDIR_REGISTERED", True)
    monkeypatch.setattr(cuda, "_FUSED_PRIMAL_REGISTERED", True)
    assert cuda.supported_configuration(_configuration())
    assert cuda.supported_configuration(_configuration(num_devices=0), num_devices=2, mode="mesh_halo")
    assert cuda.supported_configuration(_configuration(float_dtype=jnp.float64))
    assert cuda.supported_configuration(_configuration(mesh_shape=(2048, 2048, 2048)))
    assert cuda.supported_bidir_configuration(_configuration(float_dtype=jnp.float32))
    assert cuda.supported_bidir_configuration(_configuration(float_dtype=jnp.float64))
    assert cuda.supported_fused_primal_configuration(_configuration(pmid_dtype=jnp.int16))
    assert not cuda.supported_fused_primal_configuration(_configuration(pmid_dtype=jnp.int32))
    assert not cuda.supported_fused_primal_configuration(_configuration(pmid_dtype=jnp.int16, float_dtype=jnp.float64))
    assert not cuda.supported_fused_primal_configuration(
        _configuration(pmid_dtype=jnp.int16, cuda_routing_backend="cuda_merge")
    )

    monkeypatch.setattr(cuda, "_qualified_jax", lambda: False)
    assert not cuda.supported_configuration(_configuration())
    monkeypatch.setattr(cuda, "_qualified_jax", lambda: True)
    monkeypatch.setattr(cuda.jax, "default_backend", lambda: "cpu")
    assert not cuda.supported_configuration(_configuration())
    monkeypatch.setattr(cuda.jax, "default_backend", lambda: "gpu")

    for conf in (
        _configuration(float_dtype=jnp.float16), _configuration(pmid_dtype=jnp.int64), _configuration(num_devices=1),
        _configuration(multigpu_mode="particle_halo"), _configuration(mesh_shape=None),
        _configuration(mesh_shape=(2**32, 2, 1)),
    ):
        assert not cuda.supported_configuration(conf)

    monkeypatch.setattr(cuda, "_register_targets", lambda: False)
    assert not cuda.supported_configuration(_configuration())
    monkeypatch.setattr(cuda, "_register_targets", lambda: True)
    monkeypatch.setattr(cuda, "_FLOAT64_REGISTERED", False)
    assert not cuda.supported_configuration(_configuration(float_dtype=jnp.float64))
    monkeypatch.setattr(cuda, "_BIDIR_REGISTERED", False)
    assert not cuda.supported_bidir_configuration(_configuration())
    assert not cuda.supported_bidir_configuration(_configuration(num_devices=1))


def test_cuda_backend_selection_is_explicit_and_never_silently_changes(monkeypatch):
    monkeypatch.delenv("PMPP_CUDA_ROUTING_BACKEND", raising=False)
    assert cuda.requested_backend() == "bidir_mergepath"
    assert cuda.requested_backend(_configuration(cuda_routing_backend="cuda_merge")) == "cuda_merge"
    monkeypatch.setenv("PMPP_CUDA_ROUTING_BACKEND", " CURRENT ")
    assert cuda.requested_backend(_configuration()) == "cuda_merge"
    monkeypatch.setenv("PMPP_CUDA_ROUTING_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unsupported cuda_routing_backend"):
        cuda.requested_backend()

    monkeypatch.delenv("PMPP_CUDA_ROUTING_BACKEND")
    assert not cuda.enabled_for_configuration(_configuration(cuda_routing=False))
    monkeypatch.setattr(cuda, "supported_bidir_configuration", lambda conf: True)
    assert cuda.enabled_for_configuration(_configuration(cuda_routing_backend="bidir_mergepath"))
    monkeypatch.setattr(cuda, "supported_configuration", lambda conf: True)
    assert cuda.enabled_for_configuration(_configuration(cuda_routing_backend="cuda_merge"))


def test_cuda_path_resolution_is_versioned_sanitized_and_overridable(tmp_path, monkeypatch):
    assert _cuda_paths.package_cuda_directory().name == "_cuda"
    monkeypatch.setattr(_cuda_paths.metadata, "version", lambda name: "0.9.1+worker build")
    assert _cuda_paths._distribution_version("jaxlib") == "0.9.1_worker_build"
    monkeypatch.setattr(
        _cuda_paths.metadata, "version", lambda name: (_ for _ in ()).throw(metadata.PackageNotFoundError(name)),
    )
    assert _cuda_paths._distribution_version("missing") == "source"

    explicit = tmp_path / "explicit"
    monkeypatch.setenv("PMPP_CUDA_ROUTING_CACHE", str(explicit))
    assert _cuda_paths.user_cache_cuda_directory() == explicit.resolve()
    monkeypatch.delenv("PMPP_CUDA_ROUTING_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    cache = _cuda_paths.user_cache_cuda_directory()
    assert cache.parts[-4:] == ("pmpp", "cuda-routing", "pmpp-source", "jaxlib-source")
    assert str(cache).startswith(str(tmp_path / "xdg"))


def test_cuda_build_helpers_parse_detect_and_preflight_strictly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_cuda.subprocess, "check_output", lambda *args, **kwargs: " 9.0\n8.0\n9.0\nbad ")
    assert build_cuda._query(["nvidia-smi"]) == "9.0\n8.0\n9.0\nbad"
    assert build_cuda._parse_compute_capabilities(build_cuda._query(["nvidia-smi"])) == ("80", "90")
    monkeypatch.setattr(build_cuda, "_query", lambda command: "8.6\n9.0")
    assert build_cuda.detect_cuda_architectures() == "86;90;90-virtual"
    monkeypatch.setattr(build_cuda, "_query", lambda command: None)
    assert build_cuda.detect_cuda_architectures() == build_cuda._FALLBACK_ARCHITECTURES
    assert "detection failed" in capsys.readouterr().err

    monkeypatch.setattr(
        build_cuda.subprocess, "check_output", lambda *args, **kwargs:
        (_ for _ in ()).throw(subprocess.CalledProcessError(1, "bad")),
    )
    assert build_cuda._query(["bad"]) is None
    monkeypatch.setattr(build_cuda.subprocess, "check_output", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    assert build_cuda._query(["missing"]) is None

    monkeypatch.setattr(build_cuda.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="only on Linux"):
        build_cuda._preflight()
    monkeypatch.setattr(build_cuda.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_cuda.shutil, "which", lambda tool: None if tool == "nvcc" else "/bin/cmake")
    with pytest.raises(RuntimeError, match="nvcc"):
        build_cuda._preflight()
    monkeypatch.setattr(build_cuda.shutil, "which", lambda tool: f"/bin/{tool}")
    monkeypatch.setattr(
        build_cuda.metadata, "version", lambda name: (_ for _ in ()).throw(metadata.PackageNotFoundError(name)),
    )
    with pytest.raises(RuntimeError, match="must be installed"):
        build_cuda._preflight()
    monkeypatch.setattr(build_cuda.metadata, "version", lambda name: "0.9.0")
    with pytest.raises(RuntimeError, match="0.9.1 or newer"):
        build_cuda._preflight()
    monkeypatch.setattr(build_cuda.metadata, "version", lambda name: "0.9.1")
    build_cuda._preflight()
    assert build_cuda._supported_jax_version("0.9.1+worker")
    assert not build_cuda._supported_jax_version("development")


def test_cuda_build_target_selection_manifest_reuse_and_atomic_copy(tmp_path, monkeypatch):
    explicit = tmp_path / "target"
    target, used_cache = build_cuda._select_target_directory(explicit)
    assert target == explicit.resolve() and not used_cache and target.is_dir()

    package = tmp_path / "package"
    cache = tmp_path / "cache"
    monkeypatch.setattr(build_cuda, "package_cuda_directory", lambda: package)
    monkeypatch.setattr(build_cuda, "user_cache_cuda_directory", lambda: cache)
    original_ensure = build_cuda._ensure_writable

    def package_read_only(path):
        if path == package:
            raise OSError("read only")
        return original_ensure(path)

    monkeypatch.setattr(build_cuda, "_ensure_writable", package_read_only)
    target, used_cache = build_cuda._select_target_directory(None)
    assert target == cache and used_cache
    monkeypatch.setattr(build_cuda, "_ensure_writable", original_ensure)
    target, used_cache = build_cuda._select_target_directory(None)
    assert target == package and not used_cache

    assert not build_cuda._existing_artifact_matches(target, "80;90")
    (target / "libpmpp_cuda_routing.so").write_bytes(b"library")
    manifest = target / "pmpp_cuda_routing.manifest.json"
    monkeypatch.setattr(build_cuda.metadata, "version", lambda name: {"pmpp": "0.1.6", "jaxlib": "0.9.1"}[name])
    payload = {
        "record_format_version": 3,
        "routing_key_format": "uint64_le_limbs",
        "features": [build_cuda._FUSED_PRIMAL_FEATURE],
        "routing_targets": sorted(build_cuda._REQUIRED_ROUTING_TARGETS),
        "pmpp_version": "0.1.6",
        "jaxlib_version": "0.9.1",
        "embedded_cuda_architectures": ["80", "90"],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert build_cuda._existing_artifact_matches(target, "80;90")
    payload["features"] = []
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert not build_cuda._existing_artifact_matches(target, "80;90")
    payload["features"] = [build_cuda._FUSED_PRIMAL_FEATURE]
    payload["routing_targets"].remove("pmpp_route_merge_bidir_primal_i16")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert not build_cuda._existing_artifact_matches(target, "80;90")
    payload["routing_targets"].append("pmpp_route_merge_bidir_primal_i16")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert not build_cuda._existing_artifact_matches(target, "86")
    manifest.write_text("bad json", encoding="utf-8")
    assert not build_cuda._existing_artifact_matches(target, "80;90")

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("scientific artifact", encoding="utf-8")
    build_cuda._copy_atomic(source, destination)
    assert destination.read_text(encoding="utf-8") == "scientific artifact"
    assert not list(tmp_path.glob(".destination.*.tmp"))


def test_cuda_build_executes_isolated_command_and_requires_both_artifacts(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "build_cuda_routing.py").write_text("# builder", encoding="utf-8")
    target_dir = tmp_path / "target"

    build_cuda._build(source_dir=source_dir, target_dir=target_dir, architectures="80;90", dry_run=True)
    output = capsys.readouterr().out
    assert "--cuda-architectures" in output and "80;90" in output

    def successful_run(command, check):
        assert check is True
        build_dir = Path(command[command.index("--build-dir") + 1])
        (build_dir / "libpmpp_cuda_routing.so").write_bytes(b"library")
        (build_dir / "pmpp_cuda_routing.manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(build_cuda.subprocess, "run", successful_run)
    build_cuda._build(source_dir=source_dir, target_dir=target_dir, architectures="80;90", dry_run=False)
    assert (target_dir / "libpmpp_cuda_routing.so").read_bytes() == b"library"
    assert (target_dir / "pmpp_cuda_routing.manifest.json").is_file()

    monkeypatch.setattr(build_cuda.subprocess, "run", lambda command, check: None)
    with pytest.raises(RuntimeError, match="without a library and manifest"):
        build_cuda._build(source_dir=source_dir, target_dir=target_dir, architectures="80", dry_run=False)


def test_cuda_build_cli_reports_reuse_cache_install_and_failure(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    target = tmp_path / "target"
    monkeypatch.setattr(build_cuda, "_preflight", lambda: None)
    monkeypatch.setattr(build_cuda, "_source_directory", lambda: source)
    monkeypatch.setattr(build_cuda, "_select_target_directory", lambda explicit: (target, True))
    monkeypatch.setattr(build_cuda, "detect_cuda_architectures", lambda: "90;90-virtual")
    monkeypatch.setattr(build_cuda, "_existing_artifact_matches", lambda path, arch: True)
    assert build_cuda.main([]) == 0
    assert "already current" in capsys.readouterr().out

    built = []
    monkeypatch.setattr(build_cuda, "_existing_artifact_matches", lambda path, arch: False)
    monkeypatch.setattr(build_cuda, "_build", lambda **kwargs: built.append(kwargs))
    assert build_cuda.main(["--force", "--cuda-architectures", "80;86"]) == 0
    assert built[-1] == {"source_dir": source, "target_dir": target, "architectures": "80;86", "dry_run": False}
    output = capsys.readouterr().out
    assert "read-only" in output and "installed" in output and "bidir_mergepath" in output

    assert build_cuda.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "installed" not in output
    monkeypatch.setattr(build_cuda, "_preflight", lambda: (_ for _ in ()).throw(RuntimeError("no nvcc")))
    assert build_cuda.main([]) == 1
    assert "ERROR: no nvcc" in capsys.readouterr().err


def test_cuda_source_directory_fails_with_actionable_search_list(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(RuntimeError, match="build sources are missing; searched"):
        build_cuda._source_directory()
