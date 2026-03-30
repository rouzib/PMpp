import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from tqdm.auto import tqdm

# Enable 64-bit precision for accurate Finite Difference checks
config.update("jax_enable_x64", True)

from src.particles import Particles
from src.scatter import scatter

from pmwd.configuration import Configuration as Configuration_pmwd
from pmwd.scatter import scatter as scatter_pmwd
from pmwd.particles import Particles as Particles_pmwd

from test_utils import init_conf


def get_test_conf():
    box_size = 100.0
    mesh_shape = 1
    num_ptcl = 8

    conf = init_conf(num_ptcl, mesh_shape, box_size, num_devices=None, max_ptcl_per_slice=1.5)
    return conf


def test_scatter():
    conf = get_test_conf()

    conf_pmwd = Configuration_pmwd(ptcl_spacing=conf.ptcl_spacing,
                                   ptcl_grid_shape=conf.ptcl_grid_shape,
                                   mesh_shape=conf.mesh_shape, a_start=conf.a_start,
                                   a_nbody_maxstep=conf.a_nbody_maxstep)

    # 1. Initialize Particles
    key = jax.random.PRNGKey(42)
    pos = jax.random.uniform(key, shape=(conf.ptcl_num, 3)) * conf.box_size[0]

    # Create particle object
    ptcl = Particles.from_pos(conf, pos)
    ptcl_pmwd = Particles_pmwd.from_pos(conf_pmwd, pos)

    # 2. Define a Scalar Loss Function
    def loss_fn(disp_array, fn, conf, particle_template):
        p_temp = particle_template.replace(disp=disp_array)

        # Run scatter
        mesh = fn(p_temp, conf)

        # Return scalar
        return 0.5 * jnp.sum(mesh ** 2)

    print("\n--- Starting Scatter Gradient Test ---")

    # 3. Compute Analytic Gradient (using your code's VJP)
    print("Computing JAX AD Gradient...")
    grad_fn = jax.grad(loss_fn, argnums=0)
    grad_ad = grad_fn(ptcl.disp, scatter, conf, ptcl)

    # 4. Compute Finite Difference Gradient
    print("Computing PMWD AD Gradient...")
    grad_fn = jax.grad(loss_fn, argnums=0)
    grad_ad_pmwd = grad_fn(ptcl_pmwd.disp, scatter_pmwd, conf_pmwd, ptcl_pmwd)

    # 5. Compare
    diff = grad_ad - grad_ad_pmwd
    abs_err = jnp.abs(diff)
    rel_err = abs_err / (jnp.abs(grad_ad_pmwd) + 1e-10)

    print("\n--- Results ---")
    print(f"Max Absolute Error: {jnp.max(abs_err):.2e}")
    print(f"Max Relative Error: {jnp.max(rel_err):.2e}")
    print(f"Mean Relative Error: {jnp.mean(rel_err):.2e}")

    # Check specific failure cases
    is_close = jnp.allclose(grad_ad, grad_ad_pmwd, atol=1e-8, rtol=1e-8)
    if is_close:
        print("\nSUCCESS: Analytic gradients match Finite Differences.")
    else:
        print("\nFAILURE: Gradients do not match.")
        print("Sample Mismatch (Index 0 particle):")
        print(f"AD: {grad_ad[0]}")
        print(f"PMWD: {grad_ad_pmwd[0]}")


if __name__ == "__main__":
    test_scatter()
