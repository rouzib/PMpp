# Installation

PM++ currently requires Python 3.10 or newer. The project/display name is **PM++**, the distribution name is `pmpp`, and the import name is `pmpp`.

## CPU / basic JAX install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the core package and a CPU-compatible JAX stack from the package requirements.

## CUDA / GPU JAX install

JAX CUDA installation depends on the CUDA, cuDNN, driver, and JAX versions available on your system. Install the JAX wheel recommended by the official JAX installation instructions for your CUDA stack, then install PM++:

```bash
python -m pip install -e .
```

Verify visible devices with:

```python
import jax
print(jax.devices())
```

## Editable/developer install

```bash
python -m pip install -e ".[dev]"
```

## Documentation install

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

Preview locally with:

```bash
python -m http.server --directory docs/build/html 8000
```

## Multi-GPU environment check

```python
import jax

gpus = [device for device in jax.devices() if device.platform == "gpu"]
print(gpus)
if len(gpus) < 2:
    raise RuntimeError("multi-GPU tutorials require at least two GPU devices")
```

## Common install failures

- **No GPU devices shown:** confirm the CUDA-enabled JAX wheel, GPU driver, and `CUDA_VISIBLE_DEVICES` setting.
- **First run is slow:** JAX is compiling. Repeated calls with the same static shapes should be faster.
- **Documentation import failures:** install `.[docs]`; optional heavy dependencies are mocked during docs builds where appropriate.
