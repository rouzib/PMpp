"""Isolate a four-device JAX all-to-all without importing PM++.

Run with ``NCCL_DEBUG=INFO`` on the target GPU node. This checks the collective
used by distributed FFT reshards independently of the PM++ routing code.
"""

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P


def main():
    devices = jax.devices()
    print(f"jax={jax.__version__} jaxlib={jaxlib.__version__}", flush=True)
    print(f"devices={devices}", flush=True)
    if len(devices) != 4 or any(device.platform != devices[0].platform for device in devices):
        raise RuntimeError("expected exactly four devices on one platform")

    mesh = Mesh(np.asarray(devices), ("gpus",))
    source = np.arange(64, dtype=np.float32).reshape(16, 4)
    expected = np.concatenate([source[rank::4] for rank in range(4)], axis=0)

    def exchange(local):
        return jax.lax.all_to_all(local, "gpus", split_axis=0, concat_axis=0)

    route = jax.jit(jax.shard_map(
        exchange, mesh=mesh, in_specs=P("gpus", None),
        out_specs=P("gpus", None),
    ))
    actual = np.asarray(route(jnp.asarray(source)))
    np.testing.assert_array_equal(actual, expected)
    print("four-device JAX all-to-all passed", flush=True)


if __name__ == "__main__":
    main()
