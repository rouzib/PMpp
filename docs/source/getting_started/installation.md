# Installation

PM++ requires Python 3.10 or newer. The project name is **PM++**. Both the
distribution and import name are `pmpp`. PM++ supports JAX 0.9.1 through 0.10
(`jax>=0.9.1,<0.11`); Python 3.10 resolves to the newest compatible JAX release.

## Install the user environment

Start with the official
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) and
choose the CUDA 12 or CUDA 13 build that matches the available driver. PM++
also installs the plotting and I/O libraries used by the documentation. A user
environment needs only PM++ and Jupyter:

```bash
python -m pip install pmpp jupyter
```

CUDA and cuDNN user-space libraries are supplied by the JAX wheel.

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
