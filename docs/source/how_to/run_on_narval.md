# Run on Narval

Narval-specific commands depend on the active module stack and allocation policy. Keep cluster paths and account names out of general tutorials; put them here instead.

## Checklist

1. Request GPU nodes through the scheduler.
2. Load the CUDA/JAX-compatible environment used by your group.
3. Verify `jax.devices()` inside the allocated job, not on the login node.
4. Run the small multi-GPU tutorial before launching a production mesh.
5. Record scheduler resources, PM++ commit, capacity settings, and random seed.

If local project paths or account names are needed, document them in a private runbook rather than in the public PM++ user guide.
