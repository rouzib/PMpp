# Installation

PM++ requires Python 3.10 or newer. The project name is **PM++**; both the
distribution and import name are `pmpp`.

## Install the user environment

On Linux x86_64 or WSL2 with an NVIDIA driver visible to Linux, PM++ requests
a CUDA 12-enabled JAX build and installs the plotting/I/O libraries used by
the documentation. A user environment needs only PM++ and Jupyter:

```bash
# Run inside the Ubuntu/WSL2 shell, not Windows PowerShell.
python3.10 -m venv ~/.venvs/pmpp
source ~/.venvs/pmpp/bin/activate
python -m pip install --upgrade pip
python -m pip install pmpp jupyter

# The checkout supplies the notebooks; PM++ itself remains pip-installed.
git clone https://github.com/rouzib/PMpp.git ~/PMpp
cd ~/PMpp
jupyter lab docs/source/notebooks
```

Keep the virtual environment in the WSL2 Linux filesystem (for example under
`~/.venvs`), not in a Windows-mounted directory. Do not install the checkout in
editable mode for this user workflow and do not add its `src/` directory to
`PYTHONPATH`; the notebooks must import the package installed by pip.

The machine must already have a sufficiently recent NVIDIA driver; CUDA and
cuDNN user-space libraries are supplied by the JAX wheel. On a Windows host,
run this environment under WSL2 rather than native Windows Python and confirm
that both GPUs are visible there. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
before launching shared development runs.

On non-Linux platforms, PM++ installs the regular JAX distribution so that
the package and CPU-safe utilities remain importable. The documented simulation
gallery still requires exactly two CUDA GPUs.

## Verify the backend

```python
import jax

print("JAX:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
```

The documented PM++ workflows require at least two visible GPUs and a
CUDA-enabled JAX installation. Each example selects exactly two devices.

## Verify PM++ and the two-device mesh

```python
import jax

from pmpp.utils import create_compute_mesh

gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("PM++ simulation examples require at least two GPUs")

selected_devices = gpu_devices[:2]
compute_mesh = create_compute_mesh(selected_devices)

print("PM++ devices:", selected_devices)
print("mesh size:", compute_mesh.size)
print("mesh axes:", compute_mesh.axis_names)
```

`compute_mesh.size` should be `2`. Passing the ordered device list establishes
the named axis and the corresponding x-slab order used by PM++ sharding.

## Documentation environment

```bash
git clone https://github.com/rouzib/PMpp.git
cd PMpp
python -m pip install -e ".[docs]" jupyter
sphinx-build -W --keep-going -b html docs/source docs/build/html
python -m http.server --directory docs/build/html 8000
```

Read the Docs builds the committed notebook outputs; it does not execute the
notebooks or require GPUs.

## Common installation problems

- **No GPU devices appear:** verify the CUDA-enabled JAX wheel and driver;
  PM++ cannot make an unavailable accelerator visible.
- **Import resolves to an old checkout:** inspect `pmpp.__file__`, reactivate the
  intended environment, and reinstall with `-e .`.
- **A first call is slow:** the first call includes JAX compilation. Time a
  blocked second call when measuring execution.
- **GPU memory is allocated eagerly:** set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` before starting Python when sharing a
  development machine.
