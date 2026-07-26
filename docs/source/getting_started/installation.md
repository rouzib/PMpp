# Installation

PM++ requires Python 3.10 or newer. The project name is **PM++**; both the
distribution and import name are `pmpp`.

## Create an environment

For local development or documentation work:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On native Windows, PowerShell activation is
`.venv\Scripts\Activate.ps1`; use that environment for documentation work.
For NVIDIA and multi-GPU PM++ on a Windows host, run Linux Python under WSL2
rather than native Windows Python, create/activate the environment inside WSL,
and make sure the GPUs are visible there. On Linux or WSL, install the JAX wheel
appropriate for the driver/CUDA stack by following the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then install PM++ with the command above. Set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before launching shared development runs.

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
python -m pip install -e ".[docs]"
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
