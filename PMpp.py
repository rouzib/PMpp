"""
Date: 03/10/2024

This script is used mainly to simulate the evolution of dark matter particles
in the universe using the Particle-Mesh (PM) method. It provides options to
simulate the evolution of the cosmos with or without corrections from the
neural network models.

The script is structured around three functions 'load_lh', 'run_pm', and 'run_pm_model'.

The 'load_lh' function is used to load Latin Hypercube (LH) simulations
with specified indexes and returns the simulation particles' positions,
velocities, redshifts, and cosmological information.

The 'run_pm' function is used to run the PM simulation with initial
conditions and returns the simulation result.

The 'run_pm_model' is similar to 'run_pm' but accepts a neural network
model and parameters as inputs, which are used to adjust the simulation.

Another function 'run_sim_with_model' is available which runs the simulation
with PM correction.

In the main block of the script, it starts with loading the LH simulations and
initial conditions for the universe simulations (both with and without
corrections). It generates and displays the density fields of simulations.

It finally stores the simulated velocities result in a file.
"""

import os
from functools import partial
from typing import List, Tuple, Union, Callable, Optional, Any

import numpy as np
import pickle
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax import Array
from jax.experimental.ode import odeint
from jax_cosmo import Cosmology
from matplotlib import pyplot as plt

from jaxpm.painting import cic_paint
from jaxpm.pm import make_ode_fn, make_neural_ode_fn_multiple, make_neural_ode_fn
from camels_utils import normalize_by_mesh
from Models import initialize_model

# Avoiding preallocation for Python's XLA client
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Defining the devices for computation
cpus = jax.devices("cpu")
gpus = jax.devices("gpu")


def load_lh(indexes: List[int], box_size: List[float], n_mesh: int, path: str = "CamelsSims", normalize: bool = True,
            cpu_memory: bool = False, debug: bool = False) -> tuple[Any, Any, list[Any] | Array, list[Any]]:
    """
    Load LH data from files.

    :param indexes: List of LH indices to load data for.
    :param box_size: Box size of the simulation.
    :param n_mesh: Number of mesh points.
    :param path: Path to the directory containing the LH data files. Default is "CamelsSims".
    :param normalize: Boolean indicating whether to normalize by the mesh the positions and velocities. Default is True.
    :param cpu_memory: Boolean indicating whether to keep the data in cpu memory. Default is False.
    :param debug: Boolean indicating whether to print debug statements. Default is False.

    :return: Tuple containing the loaded LH data: (p, v, z, cosmo).
            - p: Normalized target positions as a NumPy array.
            - v: Normalized target velocities as a NumPy array.
            - z: Redshift values as a NumPy array.
            - cosmo: List of cosmologies as Jax cosmology objects.
    """

    def read_single_cosmo(c):
        # Creates a jax cosmology object with planck15 parameters
        jax_cosmology = jc.Planck15(Omega_c=float(c[0]) - 0.049, Omega_b=0.049, n_s=0.9624, h=0.671,
                                    sigma8=float(c[1]))
        return jax_cosmology

    # Initialize data arrays
    p, v, z, cosmo = [], [], [], []
    # Iterate over indexes to load simulation data
    for i in indexes:
        try:
            if debug:
                print(f"loading LH_{i}")
            # Load data
            target_pos = jnp.load(f"{path}/LH_{i}_pos_{n_mesh}.npy")
            target_vel = jnp.load(f"{path}/LH_{i}_vel_{n_mesh}.npy")
            z = jnp.load(f"{path}/LH_{i}_z_{n_mesh}.npy")
            planck_cosmology = read_single_cosmo(jnp.load(f"{path}/LH_{i}_cosmo.npy"))

            # Move the data to CPU
            target_pos = jax.device_put(target_pos, cpus[0])
            target_vel = jax.device_put(target_vel, cpus[0])

            # Append to lists
            p.append(target_pos)
            v.append(target_vel)
            cosmo.append(planck_cosmology)

            """
            Some code to use jnp.ndarray instead of lists to avoid copying twice the data in memory
            
            if p is None:
                p = jnp.expand_dims(target_pos, axis=0).copy()
            else:
                p = jnp.append(p, jnp.expand_dims(target_pos, axis=0), axis=0)
            if v is None:
                v = jnp.expand_dims(target_vel, axis=0).copy()
            else:
                v = jnp.append(v, jnp.expand_dims(target_vel, axis=0), axis=0)
            """
        except Exception as e:
            print(e)

    # Normalize the positions and velocities by mesh
    if normalize:
        p, v = normalize_by_mesh(jnp.array(p), jnp.array(v), box_size[0], n_mesh)

    # Move the data to GPU or CPU
    if not cpu_memory:
        p, v = jax.device_put(p, gpus[0]), jax.device_put(v, gpus[0])
    return p, v, z, cosmo


@partial(jax.jit, static_argnames=["n_mesh", "use_redshifts"])
def run_pm(initial_pos: jnp.ndarray, initial_vel: jnp.ndarray, redshifts: List[float], cosmo: Union[dict, 'Cosmology'],
           n_mesh: int, use_redshifts: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """

    Run the PM simulation with initial conditions pos and vel

    :param initial_pos: The initial position of the particles. This is a 1D array of shape (n_particles, 3), where
     n_particles is the number of particles and 3 denotes the x, y, and z coordinates respectively.
    :param initial_vel: The initial velocity of the particles. This is a 1D array of shape (n_particles, 3), where
     n_particles is the number of particles and 3 denotes the x, y, and z components of the velocity respectively.
    :param redshifts: The redshift values at which the simulation is run. This is a 1D array of shape (n_redshifts,),
     where n_redshifts is the number of desired redshifts.
    :param cosmo: The cosmological parameters used in the simulation. This can be a dictionary or an object containing
     the necessary cosmological information.
    :param n_mesh: The number of mesh cells along each axis in the simulation. This determines the resolution of the
     simulation.
    :param use_redshifts: Boolean indicating whether redshifts or scale factors are used.

    :return: The final positions and velocities of the particles at each redshift specified. This is a 2D array of
     shape (n_particles, 2, n_redshifts), where n_particles is the number of particles, 2 indicates the positions and
      velocities, and n_redshifts is the number of redshifts specified.
    """
    # Converts redshifts to scale factors
    if use_redshifts:
        scale_factors = 1 / (1 + jnp.array(redshifts))
    else:
        scale_factors = redshifts

    mesh_shape = [n_mesh, n_mesh, n_mesh]
    return odeint(make_ode_fn(mesh_shape), [initial_pos, initial_vel], jnp.array(scale_factors), cosmo,
                  rtol=1e-5, atol=1e-5, )


@partial(jax.jit, static_argnames=["n_mesh", "model"])
def run_pm_model(pos: jnp.ndarray, vels: jnp.ndarray, scale_factors: jnp.ndarray, cosmo: Union[dict, "Cosmology"],
                 n_mesh: int, model: Callable, params: List, weights: Optional[jnp.array] = None):
    """
    Run the PM simulation with initial conditions pos and vel with the NN passed in model and params. This code handles
    multiple models with a list of parameters.

    :param pos: The initial positions of the particles.
    :param vels: The initial velocities of the particles.
    :param scale_factors: The scale factors at which the solution is desired.
    :param cosmo: The cosmological parameters.
    :param n_mesh: The number of grid cells per dimension.
    :param model: The neural network model for the PM model.
    :param params: The model parameters.
    :param weights: The weights for the neural network model (optional).

    :return: The solution obtained by integrating the PM model using the provided parameters.
    :rtype: numpy.ndarray

    """
    mesh_shape = [n_mesh, n_mesh, n_mesh]
    return odeint(make_neural_ode_fn_multiple(model, mesh_shape, weights=weights), [pos, vels],
                  jnp.array(scale_factors), cosmo, params, rtol=1e-5, atol=1e-5)


@partial(jax.jit, static_argnames=["n_mesh", "model"])
def run_pm_with_correction(pos: jnp.ndarray, vels: jnp.ndarray, scale_factors: jnp.ndarray,
                           cosmo: Union[dict, "Cosmology"], n_mesh: int, model: Callable, params: List):
    """
    Run the PM simulation with initial conditions pos and vel with the NN passed in model and params. This code handles
    a single model.

    :param pos: The initial positions of the particles.
    :param vels: The initial velocities of the particles.
    :param scale_factors: The scale factors at which the solution is desired.
    :param cosmo: The cosmological parameters.
    :param n_mesh: The number of grid cells per dimension.
    :param model: The neural network model for the PM model.
    :param params: The model parameters.
    :param weights: The weights for the neural network model (optional).

    :return: The solution obtained by integrating the PM model using the provided parameters.
    :rtype: numpy.ndarray

    """
    mesh_shape = [n_mesh, n_mesh, n_mesh]
    return odeint(make_neural_ode_fn(model, mesh_shape), [pos, vels], jnp.array(scale_factors), cosmo, params,
                  rtol=1e-5, atol=1e-5)


def run_sim_with_model(initial_pos: jnp.ndarray, initial_vel: jnp.ndarray, redshifts: List[float], cosmo: Cosmology,
                       n_mesh: int = 64, debug: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Runs a simulation with a given model.

    :param initial_pos: The initial position of particles in the simulation.
    :param initial_vel: The initial velocity of particles in the simulation.
    :param redshifts: List of redshifts at which the simulation should be run.
    :param cosmo: The cosmology for the simulation.
    :param n_mesh: The number of mesh points used in the simulation. Default is 64.
    :param debug: A flag indicating whether to run the simulation in debug mode. Default is False.
    :return: A tuple containing the corrected positions and velocities after simulation.
    """
    # Load model parameters based on the mesh size
    if n_mesh >= 64:
        amountZeroPercent = 0.35
    else:
        amountZeroPercent = 0.4

    numberOfK = int(n_mesh / 2) + 1
    amountZero = int(np.floor(numberOfK * amountZeroPercent))
    weights = jnp.array([jnp.concatenate((jnp.zeros(numberOfK - amountZero), jnp.ones(amountZero))),
                         jnp.concatenate((jnp.ones(numberOfK - amountZero), jnp.zeros(amountZero)))])

    # Initialize model
    modelName = "Default"
    nKnots = 32
    latentSize = 64
    model, _ = initialize_model(n_mesh=n_mesh, model_name=modelName, n_knots=nKnots, latent_size=latentSize)

    # Load model parameters
    modelPaths = ["Model/MyModel_nMesh64_LH100-149_Lr0.001_regularization/model499.pkl",
                  "Model/MyModel_nMesh32_LH100-499_Lr0.001_regularization_vel/model395.pkl"]
    paramsList = []
    for modelPath in modelPaths:
        with open(modelPath, 'rb') as file:
            params = pickle.load(file)
        paramsList.append(params)

    scale_factors = 1 / (1 + jnp.array(redshifts))  # Convert redshift to scale factor
    # Run the PM simulation with the model
    if debug:
        print("Running PM corrected Simulation")
    posPmCorr, velPmCorr = run_pm_model(pos=initial_pos, vels=initial_vel, scale_factors=scale_factors, cosmo=cosmo,
                                        n_mesh=n_mesh, model=model, params=paramsList, weights=weights)

    return posPmCorr, velPmCorr


def run_and_plot_simulation(initial_pos: np.ndarray, initial_vel: np.ndarray, redshifts: List[float], cosmo: dict,
                            n_mesh: int, run_func: Callable, plot_title: str, start: int, end: int,
                            display_onez: bool, new_nmesh: int, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param initial_pos: Initial positions of particles in simulation.
    :param initial_vel: Initial velocities of particles in simulation.
    :param redshifts: List of redshifts at which the simulation is run.
    :param cosmo: Cosmology parameters for the simulation.
    :param n_mesh: Number of mesh points for the simulation.
    :param run_func: Function that runs the simulation.
    :param plot_title: Title for the plot.
    :param start: Starting index of the redshifts to be displayed.
    :param end: Ending index of the redshifts to be displayed.
    :param display_onez: Boolean flag indicating whether to display only one redshift.
    :param new_nmesh: Number of mesh points for plot of higher resolution.
    :param cutoff: Value used to avoid zero value when taking logarithm image.
    :return: Tuple of the position and velocity arrays from the simulation.

    """
    pm_pos_temp, pm_vel_temp = run_func(initial_pos=initial_pos, initial_vel=initial_vel, redshifts=redshifts,
                                        cosmo=cosmo, n_mesh=n_mesh)

    plt.figure(figsize=[5, 5])
    plt.suptitle(plot_title)
    for i in range(start, end):
        if not display_onez:
            plt.subplot(4, 4, i + 1)
        plt.imshow(np.log(cic_paint(jnp.zeros([new_nmesh] * 3), pm_pos_temp[::2][i] * new_nmesh / n_mesh).sum(axis=0)
                          + cutoff), cmap="cividis")
    plt.show()
    return pm_pos_temp, pm_vel_temp


if __name__ == '__main__':
    sim = 0  # Simulation index
    nMesh = 32  # Number of mesh points
    boxSize = [25.0, 25.0, 25.0]  # Simulation box size

    # Load Latin Hypercube simulation data
    targetP, targetV, z, cosmology = load_lh([sim], boxSize, nMesh, path="CamelsSims", debug=True)
    targetP, targetV = targetP[0], targetV[0]  # Get target positions and velocities
    initialP, initialV = targetP[0], targetV[0]  # Initializing position & velocity
    cosmology = cosmology[0]  # Get the cosmology

    # Plotting parameters
    display_onez = True
    new_nMesh = 256  # New density resolution
    start, end = (-1, 0) if display_onez else (0, 16)  # Range of redshifts to display
    cutoff = 1e-1  # Avoiding zero-value denominator

    # Run and plot the simulation with the model applied and without it respectively, and store the resulting
    # particle velocities.
    # First run simulation with correction
    pm_pos_corr, pm_vel_corr = run_and_plot_simulation(initial_pos=initialP, initial_vel=initialV, redshifts=z,
                                                       cosmo=cosmology,
                                                       n_mesh=nMesh, run_func=run_sim_with_model,
                                                       plot_title="PM_corrected simulation",
                                                       start=start, end=end, display_onez=display_onez,
                                                       new_nmesh=new_nMesh, cutoff=cutoff)

    # Now, run simulation without correction
    pm_pos, pm_vel = run_and_plot_simulation(initial_pos=initialP, initial_vel=initialV, redshifts=z, cosmo=cosmology,
                                             n_mesh=nMesh, run_func=run_pm, plot_title="PM simulation",
                                             start=start, end=end, display_onez=display_onez, new_nmesh=new_nMesh,
                                             cutoff=cutoff)

    # Store the simulation velocities result in a file
    with open("Pk/" + f"vels_LH{sim}_{nMesh}.pkl", 'wb') as file:
        # We perform the dumping of the 'velocities' outcomes in a file, so that we can retrieve it conveniently.
        pickle.dump({f"LH_{sim}": [targetV, pm_vel, pm_vel_corr]}, file)
