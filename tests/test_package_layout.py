"""Import-contract tests for the feature-oriented PM++ package layout."""

import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pmpp
import pmpp.cic as cic_package
import pmpp.core as core_package
import pmpp.cosmology as cosmology_package
import pmpp.nbody as nbody_package
from pmpp.analysis import density_to_pk
from pmpp.cic import gather, scatter
from pmpp.core import Configuration, pmid_to_idx
from pmpp.corrections import apply_potential_correction
from pmpp.cosmology import Cosmology, boltzmann
from pmpp.distributed import MultiGPUConfiguration, build_cuda_main, create_compute_mesh
from pmpp.extras.camels import CamelsMetadata
from pmpp.extras.quijote import QuijoteCanonicalization
from pmpp.initial_conditions import linear_modes, lpt, white_noise
from pmpp.nbody import Particles, nbody
from pmpp.numerics import fftfwd, odeint

LEGACY_FLAT_IMPORTS = {
    "FFT_distributed", "_api", "_compat", "_cuda_paths", "boltzmann", "build_cuda_routing", "camels_io",
    "configuration", "cosmo", "cuda_routing", "enmesh", "fft", "gather", "gravity", "growth", "halo_moving", "lpt",
    "mesh_halo", "modes", "multigpu_configuration", "nbody_observers", "ode_util", "pallas_cic", "particles",
    "plotting_utils", "potential_correction", "power_spectrum", "quijote_io", "quijote_metrics", "scatter", "steps",
    "utils",
}
LEGACY_FLAT_FILES = {f"{name}.py" for name in LEGACY_FLAT_IMPORTS} | {"nbody.py"}


def _import_subprocess_environment():
    """Keep pure package-import probes independent of the parent's GPU client."""
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "PMPP_CUDA_ROUTING": "0"})
    return environment


def test_feature_package_initializers_expose_supported_apis():
    """Feature packages expose the supported task-level interfaces."""
    assert pmpp.Configuration is Configuration
    assert pmpp.MultiGPUConfiguration is MultiGPUConfiguration
    assert inspect.getattr_static(core_package, "Configuration") is Configuration
    assert inspect.getattr_static(cic_package, "scatter") is scatter
    assert inspect.getattr_static(cosmology_package, "boltzmann") is boltzmann
    assert inspect.getattr_static(nbody_package, "nbody") is nbody
    for api in (
        CamelsMetadata, Cosmology, Particles, QuijoteCanonicalization, apply_potential_correction, boltzmann,
        build_cuda_main, create_compute_mesh, density_to_pk, fftfwd, gather, linear_modes, lpt, nbody, odeint,
        pmid_to_idx, scatter, white_noise,
    ):
        assert callable(api)


def test_feature_api_is_independent_of_implementation_import_order():
    """Loaded submodules must not shadow functions curated by package initializers."""
    command = [
        sys.executable, "-c",
        (
            "import importlib; import pmpp; "
            "importlib.import_module('pmpp.cic.scatter'); "
            "importlib.import_module('pmpp.cic.gather'); "
            "importlib.import_module('pmpp.cosmology.growth'); "
            "importlib.import_module('pmpp.nbody.gravity'); "
            "from pmpp.cic import gather, scatter; "
            "from pmpp.cosmology import growth; "
            "from pmpp.nbody import gravity; "
            "assert all(map(callable, (gather, scatter, growth, gravity)))"
        ),
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1], env=_import_subprocess_environment(), )


def test_nbody_is_a_package_with_the_solver_surface():
    """The grouped N-body package owns the solver and implementation modules."""
    spec = importlib.util.find_spec("pmpp.nbody")
    assert spec is not None
    assert spec.submodule_search_locations is not None
    assert nbody.__module__ == "pmpp.nbody.solver"
    assert importlib.util.find_spec("pmpp.nbody.gravity") is not None
    assert importlib.util.find_spec("pmpp.nbody.integrator") is not None


def test_legacy_flat_modules_and_files_are_absent():
    """The completed refactor leaves no importable flat compatibility modules."""
    package_dir = Path(pmpp.__file__).resolve().parent
    existing_flat_files = {path.name for path in package_dir.glob("*.py")}
    assert existing_flat_files.isdisjoint(LEGACY_FLAT_FILES)
    for legacy_name in sorted(LEGACY_FLAT_IMPORTS):
        assert importlib.util.find_spec(f"pmpp.{legacy_name}") is None


def test_package_root_does_not_eagerly_import_dataset_extras():
    """Importing the solver root must not pull optional dataset adapters in."""
    command = [
        sys.executable, "-c",
        (
            "import sys; import pmpp; "
            "assert not any(name == 'pmpp.extras' or name.startswith('pmpp.extras.') "
            "for name in sys.modules)"
        ),
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1], env=_import_subprocess_environment(), )


def test_canonical_cuda_builder_module_is_runnable():
    """The canonical distributed CUDA-builder module supports ``python -m``."""
    subprocess.run([sys.executable, "-m", "pmpp.distributed.build_cuda", "--help"], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                   env=_import_subprocess_environment(),
                   )
