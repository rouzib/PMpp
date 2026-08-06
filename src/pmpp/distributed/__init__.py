"""Multi-device topology, FFTs, halo exchange, and particle routing."""

from .._api import install_lazy_api

_EXPORTS = {
    "AXIS_NAME": ("..core", "AXIS_NAME"),
    "MultiGPUConfiguration": (".configuration", "MultiGPUConfiguration"),
    "_exchange_compacted_particles_packed": (".routing", "_exchange_compacted_particles_packed"),
    "build_cuda": (".build_cuda", None),
    "build_cuda_main": (".build_cuda", "main"),
    "build_multigpu_configuration": (".configuration", "build_multigpu_configuration"),
    "create_batched_transposed_real_ffts": (".fft", "create_batched_transposed_real_ffts"),
    "create_compute_mesh": ("..core", "create_compute_mesh"),
    "create_ffts": (".fft", "create_ffts"),
    "cuda": (".cuda", None),
    "extension_status": (".cuda", "extension_status"),
    "initialize_multigpu_runtime": (".configuration", "initialize_multigpu_runtime"),
    "routing": (".routing", None),
    "supported_configuration": (".cuda", "supported_configuration"),
}

install_lazy_api(__name__, _EXPORTS)
