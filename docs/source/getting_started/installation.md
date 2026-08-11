# Installation

PM++ requires Python 3.10 or newer. The project name is **PM++**. Both the
distribution and import name are `pmpp`. PM++ supports JAX 0.9.1 through 0.10
(`jax>=0.9.1,<0.11`); Python 3.10 resolves to the newest compatible JAX release.

## Install the user environment

Start with the official
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) and
choose the CUDA 12 or CUDA 13 build that matches the available driver. PM++
also installs the plotting and I/O libraries used by simulations.

Install JAX before PM++ so that the accelerator build is explicit. For a CUDA
12 machine and an environment used only to run simulations:

```bash
python -m pip install --upgrade pip
python -m pip install "jax[cuda12]>=0.9.1,<0.11"
python -m pip install pmpp
```

CUDA and cuDNN user-space libraries are supplied by the JAX wheel.

Choose a PM++ extra when the environment has additional responsibilities:

```bash
# Run the repository test suite.
python -m pip install "pmpp[dev]"

# Build the documentation.
python -m pip install "pmpp[docs]"

# Develop, test, and build the documentation.
python -m pip install "pmpp[dev,docs]"
```

Each extra includes the base PM++ installation. Install Jupyter separately if
you want to open the example notebooks interactively:

```bash
python -m pip install jupyter
```

When a compatible CUDA development toolkit and CMake are available, finish by
building the optional accelerated routing extension in the same environment:

```bash
pmpp-build-cuda-routing
```

If `nvcc` is unavailable, skip this final command. Simulations and tests still
use the supported portable JAX router.

Set the following variable before launching runs on a shared machine:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

```{note}
The standard package uses the portable JAX router. For the optional compiled
extension, install a CUDA development toolkit and run
`pmpp-build-cuda-routing` once in the PM++ environment. See the
[CUDA routing guide](../user_guide/cuda_routing.md) for requirements,
architecture detection, and verification. The toolkit's `nvcc` must be a
version supported by the CUDA build of JAX installed in that environment.
```

## Install on an HPC cluster

Installation details vary across HPC sites because Python modules, CUDA
toolkits, wheel catalogues, and outbound network access are cluster-specific.
The same general sequence applies on Compute Canada and similar systems:

1. Load a site-supported Python module. Load a CUDA development module as well
   if you plan to build accelerated routing.
2. Create a virtual environment on the filesystem recommended by the site,
   then activate it before every install or run.
3. Install the CUDA-enabled JAX wheel first, followed by `pmpp` for simulations
   or the appropriate PM++ extra for tests and documentation.
4. Run `pmpp-build-cuda-routing` after `nvcc` and CMake are visible. Run the
   builder inside a GPU allocation when possible so it can detect the assigned
   GPU architecture.
5. Verify JAX and PM++ inside a scheduled GPU job. A login node may expose no
   GPUs even when the environment is installed correctly.

A portable command sequence is:

```bash
# Load the Python and, when needed, CUDA modules recommended by your site.
python -m venv /path/recommended/by/your/cluster/pmpp-venv
source /path/recommended/by/your/cluster/pmpp-venv/bin/activate

python -m pip install --upgrade pip
python -m pip install "jax[cuda12]>=0.9.1,<0.11"
python -m pip install pmpp              # simulations only
# python -m pip install "pmpp[dev]"     # repository tests instead
# python -m pip install "pmpp[dev,docs]"  # tests and documentation

pmpp-build-cuda-routing
```

Replace `jax[cuda12]` with the accelerator build required by the cluster. The
CUDA toolkit that provides `nvcc` must be compatible with that JAX build. If
the builder runs on a login node where GPUs are hidden, provide the target
architectures explicitly as described in the
[CUDA routing guide](../user_guide/cuda_routing.md).

Some compute nodes cannot download packages from PyPI. In that case, use the
cluster's Python wheel catalogue or mirror. If the site allows downloads only
on a login or transfer node, prepare a local wheel directory there:

```bash
python -m pip download --dest pmpp-wheels \
  "jax[cuda12]>=0.9.1,<0.11" pmpp
python -m pip install --no-index --find-links pmpp-wheels \
  "jax[cuda12]>=0.9.1,<0.11" pmpp
```

Add `"pmpp[dev]"` or `"pmpp[dev,docs]"` to both commands when those extras are
needed. Do not use `--no-index` unless the site wheel catalogue or local
directory contains every required dependency.

## Verify the backend

```python
import jax

print("JAX:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
```

The documented PM++ workflows require multiple visible GPUs and a CUDA-enabled
JAX installation.

## Verify PM++ and the device mesh

```python
import jax

from pmpp.distributed import create_compute_mesh

gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("PM++ simulation examples require at least two GPUs")

compute_mesh = create_compute_mesh(gpu_devices)

print("PM++ devices:", gpu_devices)
print("mesh size:", compute_mesh.size)
print("mesh axes:", compute_mesh.axis_names)
```

`compute_mesh.size` should match the number of detected GPU devices. Passing
the ordered device list establishes the named axis and corresponding x-slab
order used by PM++ sharding.

## Common installation problems

- **No GPU devices appear:** verify the CUDA-enabled JAX wheel and driver.
  PM++ cannot make an unavailable accelerator visible.
- **Import resolves to an old checkout:** inspect `pmpp.__file__`, reactivate the
  intended environment, and reinstall with `-e .`.
- **A first call is slow:** the first call includes JAX compilation. Time a
  blocked second call when measuring execution.
- **GPU memory is allocated eagerly:** set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` before starting Python when sharing a
  development machine.
