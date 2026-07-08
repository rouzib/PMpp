# Configure a cluster environment

Cluster jobs should make the JAX device view explicit and avoid relying on login-node state.

## Minimal checks

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

If the list is empty or CPU-only, inspect the scheduler GPU request, CUDA module, driver, and `CUDA_VISIBLE_DEVICES`.

## Recommended job pattern

- Create and activate a virtual environment inside the job or load a known environment.
- Install a JAX wheel compatible with the cluster CUDA stack.
- Install PM++ with `python -m pip install -e .`.
- Print `jax.devices()` and the PM++ commit hash before the run.
- Store capacities, mesh shape, device count, and seed in the output directory.
