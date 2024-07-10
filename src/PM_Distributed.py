"""
Author: Nicolas Payot
Date: 06/19/2024

This program uses Python's Jax library to implement distributed simulation of ordinary differential equations (ODE)
representing the cosmic large-scale structure dynamics on multiple GPUs. The script mainly contains two functions:
'make_neural_ode_fn_sharded()' and 'warmup()'. The 'make_neural_ode_fn_sharded()' function returns another function
that can perform simulations of an N-body system, containing sharded computations. It is augmented
with a neural network learning process. The 'warmup()' function initializes and prepares the system for simulation
by running a simulation step once on each GPU. Thus, it compiles necessary functions on each GPU.
The program also employs the Fast Fourier Transform, mesh painting techniques and gradient calculation for simulating
the large scale structure of the universe.
"""

import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding, SingleDeviceSharding
from jax.experimental.ode import odeint
import jax.numpy as jnp

from typing import List, Callable

import jax_cosmo as jc

from jaxpm.kernels import fftk, gradient_kernel, laplace_kernel, longrange_kernel
from jaxpm.painting import cic_paint, cic_read

from src.FFT_distributed import create_ffts, distribute_array_on_gpus


def make_neural_ode_fn_sharded(model, mesh_shape, compute_mesh, super_res):
    """
    Returns a function that handles the sharded (across multiple GPUs) simulation of an ordinary differential
    equations (ODE) system representing the cosmic large-scale structure dynamics. This function describes the
    equations of motion for an N-body system. Additionally, it incorporates a neural network learning process.

    Args:
        model: The trained model.
        mesh_shape: The shape of the density mesh.
        compute_mesh: The compute mesh that defines the layout of devices for distributed computation.
        super_res: The scaling factor of the mesh compared to the number of particles.

    Returns:
        A function which performs the simulation of a neural ordinary
        differential equation system on multiple GPUs.
    """

    # get fast fourier transform functions from create_ffts (FFTs are used in the simulation process)
    rfftn_jit, irfftn_jit, _, _ = create_ffts(compute_mesh, mesh_shape[0])

    # create handles to the CIC painting functions
    sharded_cic_paint_jit = jax.jit(cic_paint, in_shardings=(
        NamedSharding(compute_mesh, P(None, None, "gpus")), NamedSharding(compute_mesh, P("gpus", None))),
                                    out_shardings=NamedSharding(compute_mesh, P(None, "gpus", None)))
    sharded_cic_read_jit = jax.jit(cic_read, in_shardings=(
        NamedSharding(compute_mesh, P(None, "gpus", None)), NamedSharding(compute_mesh, P("gpus", None))),
                                   out_shardings=NamedSharding(compute_mesh, P("gpus")))

    # initialise an array across multiple GPUs
    # TODO: extract `distribute_array_on_gpus` calls from the `make_neural_ode_fn_sharded` func
    init_mesh = distribute_array_on_gpus(jnp.zeros(mesh_shape), compute_mesh, P(None, None, "gpus"))

    # pre-compute some values on cpu
    kvec = fftk(mesh_shape)
    kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
    laplace_kernel_computed = laplace_kernel(kvec)
    longrange_kernel_computed = longrange_kernel(kvec, r_split=0)
    scalor = super_res ** 3
    sqrt_res = jnp.sqrt(super_res)

    # distribute pre-computed arrays across GPUs
    # TODO: extract `distribute_array_on_gpus` calls from the `make_neural_ode_fn_sharded` func
    kk = distribute_array_on_gpus(kk, compute_mesh, P("gpus", None, None))
    laplace_kernel_computed = distribute_array_on_gpus(laplace_kernel_computed, compute_mesh, P("gpus", None, None))

    def neural_nbody_ode(state, a, cosmo, params):
        """
        Defines a simulation step in a N-body system using ordinary differential equations (ODE). It describes
        the equations of motion of the N-body system and applies a neural network model function to modify the
        gravitational potentials. The simulation step includes drift and kick operations according
        to the leapfrog integrator scheme, which is commonly used in N-body simulations.

        Args:
            state (tuple): A tuple containing the current state of the N-body system,
                           composed of position ('pos') and velocity ('vel') arrays.

            a (float): The scale factor representing the time evolution of the Universe.

            cosmo (jax_cosmo.cosmology): A JAX-Cosmo cosmology object encapsulating
                                          the cosmological parameters and functions.

            params (list): The parameters of the model.

        Returns:
            tuple: Changes in position and velocity resulting from drift and kick operations, respectively.

        Note:
            The returned values should be used to update the position and velocity of each
            particle in the N-body system during a simulation timestep.
        """

        pos, vel = state

        # jax.debug.print("scale_factor = {x}", x=a)

        delta = sharded_cic_paint_jit(init_mesh.copy(), pos)
        delta_k = rfftn_jit(delta)

        pot_k = delta_k * laplace_kernel_computed * longrange_kernel_computed * scalor

        c = jnp.array([cosmo.tree_flatten()[0][0], cosmo.tree_flatten()[0][4]])
        pot_k = pot_k * (1. + model.apply(params, kk, jnp.atleast_1d(a), c) * sqrt_res)

        forces = jnp.stack(
            [sharded_cic_read_jit(irfftn_jit(gradient_kernel(kvec, i) * pot_k), pos) for i in range(3)], axis=-1)

        forces = forces * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a ** 3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a ** 2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return neural_nbody_ode


def warmup(state, cosmo, params, model, mesh_shape, compute_mesh):
    """
    Executes a warmup function which prepares the machine for simulation by running the simulation
    step once. This is done to initialize and compile the necessary functions on each GPU.

    Args:
        state: The state of simulation.
        cosmo: The cosmology object from 'jax_cosmo' package.
        params: The parameters of the model.
        model: The trained model.
        mesh_shape: The shape of the density mesh.
        compute_mesh: The compute mesh that defines the layout of devices for distributed computation.

    Returns:
        Changes in position and velocity based on forces calculated
        by the model.
    """
    # get fast fourier transform functions from create_ffts (FFTs are used in the simulation process)
    rfftn_jit, irfftn_jit, _, _ = create_ffts(compute_mesh, mesh_shape[0])

    # create handles to the CIC painting functions
    sharded_cic_paint_jit = jax.jit(cic_paint, in_shardings=(
        NamedSharding(compute_mesh, P(None, None, "gpus")), NamedSharding(compute_mesh, P("gpus", None))),
                                    out_shardings=NamedSharding(compute_mesh, P(None, "gpus", None)))
    sharded_cic_read_jit = jax.jit(cic_read, in_shardings=(
        NamedSharding(compute_mesh, P(None, "gpus", None)), NamedSharding(compute_mesh, P("gpus", None))),
                                   out_shardings=NamedSharding(compute_mesh, P("gpus")))

    # initialise an array across multiple GPUs
    init_mesh = distribute_array_on_gpus(jnp.zeros(mesh_shape), compute_mesh, P(None, None, "gpus"))

    # pre-compute some values on cpu
    kvec = fftk(mesh_shape)
    kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
    laplace_kernel_computed = laplace_kernel(kvec)
    longrange_kernel_computed = longrange_kernel(kvec, r_split=0)

    # distribute pre-computed arrays across GPUs
    kk = distribute_array_on_gpus(kk, compute_mesh, P("gpus", None, None))
    laplace_kernel_computed = distribute_array_on_gpus(laplace_kernel_computed, compute_mesh, P("gpus", None, None))

    a = 127.0
    pos, vel = state

    delta = sharded_cic_paint_jit(init_mesh.copy(), pos)
    delta_k = rfftn_jit(delta)

    pot_k = delta_k * laplace_kernel_computed * longrange_kernel_computed

    c = jnp.array([cosmo.tree_flatten()[0][0], cosmo.tree_flatten()[0][4]])
    model_results = 1. + model.apply(params, kk, jnp.atleast_1d(a), c)
    pot_k_model = pot_k * model_results

    forces = jnp.stack(
        [sharded_cic_read_jit(irfftn_jit(gradient_kernel(kvec, i) * pot_k_model), pos) for i in range(3)], axis=-1)
    forces = forces * 1.5 * cosmo.Omega_m

    # Computes the update of position (drift)
    dpos = 1. / (a ** 3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

    # Computes the update of velocity (kick)
    dvel = 1. / (a ** 2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

    return dpos, dvel


def make_ode_fn_sharded(mesh_shape, compute_mesh, super_res):
    """
    Returns a function that handles the sharded (across multiple GPUs) simulation of an ordinary differential
    equations (ODE) system representing the cosmic large-scale structure dynamics. This function describes the
    equations of motion for an N-body system.

    Args:
        mesh_shape: The shape of the density mesh.
        compute_mesh: The compute mesh that defines the layout of devices for distributed computation.
        super_res: The scaling factor of the mesh compared to the number of particles.

    Returns:
        A function which performs the simulation of a neural ordinary
        differential equation system on multiple GPUs.
    """

    # get fast fourier transform functions from create_ffts (FFTs are used in the simulation process)
    rfftn_jit, irfftn_jit, _, _ = create_ffts(compute_mesh, mesh_shape[0])

    # create handles to the CIC painting functions
    sharded_cic_paint_jit = jax.jit(cic_paint, in_shardings=(
        NamedSharding(compute_mesh, P(None, None, "gpus")), NamedSharding(compute_mesh, P("gpus", None))),
                                    out_shardings=NamedSharding(compute_mesh, P(None, "gpus", None)))
    sharded_cic_read_jit = jax.jit(cic_read, in_shardings=(
        NamedSharding(compute_mesh, P(None, "gpus", None)), NamedSharding(compute_mesh, P("gpus", None))),
                                   out_shardings=NamedSharding(compute_mesh, P("gpus")))

    # initialise an array across multiple GPUs
    init_mesh = distribute_array_on_gpus(jnp.zeros(mesh_shape), compute_mesh, P(None, None, "gpus"))

    # pre-compute some values on cpu
    kvec = fftk(mesh_shape)
    laplace_kernel_computed = laplace_kernel(kvec)
    longrange_kernel_computed = longrange_kernel(kvec, r_split=0)
    scalor = super_res ** 3

    # distribute pre-computed arrays across GPUs
    laplace_kernel_computed = distribute_array_on_gpus(laplace_kernel_computed, compute_mesh, P("gpus", None, None))

    def nbody_ode(state, a, cosmo):
        """
        Defines a simulation step in a N-body system using ordinary differential equations (ODE). It describes
        the equations of motion of the N-body system and applies a neural network model function to modify the
        gravitational potentials. The simulation step includes drift and kick operations according
        to the leapfrog integrator scheme, which is commonly used in N-body simulations.

        Args:
            state (tuple): A tuple containing the current state of the N-body system,
                           composed of position ('pos') and velocity ('vel') arrays.

            a (float): The scale factor representing the time evolution of the Universe.

            cosmo (jax_cosmo.cosmology): A JAX-Cosmo cosmology object encapsulating
                                          the cosmological parameters and functions.

        Returns:
            tuple: Changes in position and velocity resulting from drift and kick operations, respectively.

        Note:
            The returned values should be used to update the position and velocity of each
            particle in the N-body system during a simulation timestep.
        """

        pos, vel = state

        # jax.debug.print("scale_factor = {x}", x=a)

        delta = sharded_cic_paint_jit(init_mesh.copy(), pos)
        delta_k = rfftn_jit(delta)

        pot_k = delta_k * laplace_kernel_computed * longrange_kernel_computed * scalor

        forces = jnp.stack(
            [sharded_cic_read_jit(irfftn_jit(gradient_kernel(kvec, i) * pot_k), pos) for i in range(3)], axis=-1)
        forces = forces * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a ** 3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a ** 2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return nbody_ode


def run_pm_with_correction_sharded(pos: jnp.ndarray, vels: jnp.ndarray, scale_factors: jnp.ndarray, cosmo: jc.Cosmology,
                                   n_mesh: int, model: Callable, params: List, compute_mesh: Mesh, super_res: int = 1,
                                   rtol: float = 1e-5, atol: float = 1e-5, mxstep: int = jnp.inf,
                                   previous_sharding=False):
    """
    Runs a Particle-Mesh (PM) simulation with model correction using sharded distributed computation. This function
    executes the simulation on multiple GPUs, distributing the system's particles across the available GPUs.

    Parameters:
        pos (jnp.ndarray): Initial positions of the particles.
        vels (jnp.ndarray): Initial velocities of the particles.
        scale_factors (jnp.ndarray): Array of scale factor (i.e., time evolution parameter) values where the
                                      state of the simulation (particle positions and velocities) will be computed.
        cosmo (jc.Cosmology): JAX-Cosmo Cosmology object, defining the cosmological parameters.
        n_mesh (int): Number of mesh cells per dimension for the simulation. The total size of the mesh used in the
                      simulation will be (n_mesh, n_mesh, n_mesh).
        model (Callable): Callable model function to apply the neural network based corrections.
        params (List): List of parameters to pass to the model when applying it.
        compute_mesh (Mesh): Compute mesh for the distributed computation across multiple GPUs.
        super_res (int, optional): Scaler of the mesh size during the computation. Default is 1.
        rtol (float, optional): The relative tolerance parameter for the ODE solver. Defaults to 1e-5.
        atol (float, optional): The absolute tolerance parameter for the ODE solver. Defaults to 1e-5.
        mxstep (int, optional): Maximum number of steps taken by the solver in total. Defaults to infinite.
        previous_sharding (bool, optional): If the initial sharding of the data should be preserved. Defaults to False.

    Returns:
        jnp.ndarray: Array containing the positions and velocities of the particles at each scale factor specified,
                     computed using the PM simulation with model corrections.
    """
    with compute_mesh:
        if not previous_sharding and (type(pos.sharding) == SingleDeviceSharding or pos.sharding.mesh != compute_mesh):
            pos = distribute_array_on_gpus(pos, compute_mesh, P("gpus", None))
        if not previous_sharding and (
                type(vels.sharding) == SingleDeviceSharding or vels.sharding.mesh != compute_mesh):
            vels = distribute_array_on_gpus(vels, compute_mesh, P("gpus", None))

        pos *= super_res
        vels *= super_res
        n_mesh *= super_res

        mesh_shape = [n_mesh, n_mesh, n_mesh]

        # warmup([pos, vels], cosmo, params, model, mesh_shape, compute_mesh)
        results = odeint(make_neural_ode_fn_sharded(model, mesh_shape, compute_mesh, super_res), [pos, vels],
                         scale_factors, cosmo, params, rtol=rtol, atol=atol, mxstep=mxstep)
        results[0] /= super_res
        results[1] /= super_res
        return results


def run_pm_sharded(pos: jnp.ndarray, vels: jnp.ndarray, scale_factors: jnp.ndarray, cosmo: jc.Cosmology, n_mesh: int,
                   compute_mesh: Mesh, super_res: int = 1, rtol: float = 1e-5, atol: float = 1e-5,
                   mxstep: int = jnp.inf):
    """
    Runs a Particle-Mesh (PM) simulation using sharded distributed computation across multiple GPUs.

    Parameters:
        pos (jnp.ndarray): Initial positions of the particles.
        vels (jnp.ndarray): Initial velocities of the particles.
        scale_factors (jnp.ndarray): Array of scale factor values where the state of the simulation (particle
                                      positions and velocities) will be computed.
        cosmo (jc.Cosmology): JAX-Cosmo Cosmology object, defining the cosmological parameters.
        n_mesh (int): Number of mesh cells per dimension for the simulation. The total size of the mesh used in the
                      simulation will be (n_mesh, n_mesh, n_mesh).
        compute_mesh (Mesh): Compute mesh for the distributed computation across multiple GPUs.
        super_res (int, optional): Scaler of the mesh size during the computation. Default is 1.
        rtol (float, optional): The relative tolerance parameter for the ODE solver. Defaults to 1e-5.
        atol (float, optional): The absolute tolerance parameter for the ODE solver. Defaults to 1e-5.
        mxstep (int, optional): Maximum number of steps taken by the solver in total. Defaults to infinite.

    Returns:
        jnp.ndarray: Array containing the positions and velocities of the particles at each scale factor specified,
                     computed using the basic PM simulation.
    """
    with compute_mesh:
        pos = distribute_array_on_gpus(pos * super_res, compute_mesh, P("gpus", None))
        vels = distribute_array_on_gpus(vels * super_res, compute_mesh, P("gpus", None))

        n_mesh *= super_res
        mesh_shape = [n_mesh, n_mesh, n_mesh]

        results = odeint(make_ode_fn_sharded(mesh_shape, compute_mesh, super_res), [pos, vels], scale_factors, cosmo,
                         rtol=rtol, atol=atol, mxstep=mxstep)
        results[0] /= super_res
        results[1] /= super_res
        return results
