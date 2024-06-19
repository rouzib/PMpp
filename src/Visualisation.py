"""
Date: 03/10/2024

This script is designed for running and visualizing n-body simulations and particle-mesh simulations in the context
of computational cosmology. The two main functions, 'visualize' and 'visualize_with_correction', provide a
comprehensive view of particle distributions and residuals between different types of simulations.

The 'visualize' function plots the n-body simulation results and particle mesh simulation for a given timestep. Under
the hood, it uses helper functions to generate a 4x4 grid of subplots, each displaying the density of particles in a
slice of the simulation volume.

The 'visualize_with_correction' function extends this functionality by incorporating a model to simulate and
visualize results with a correction derived from the model. It calculates both the Particle-Mesh (PM) simulation with
and without correction and further represents residuals between these two simulations and the original n-body
simulation.

Each function uses the cloud-in-cell (CIC) method for interpolating particle densities at each mesh point. Providing
a detailed view of how the given system evolves under specified cosmology and particle dynamics. Moreover,
'visualize_with_correction' introduces the ability to examine the impact of modifications on these factors.
"""
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.Models import initialize_model
from src.PMpp import run_pm_with_correction, run_pm, load_lh
from jaxpm.painting import cic_paint


# @partial(jax.jit, static_argnames=["n_mesh", "sim_index", "path"])
def visualize(target_pos, target_vel, cosmo, sim_index, n_mesh, new_nmesh, scale_factors, save=False, path: str = ""):
    """
    Plot 4x4 images of the simulation given by indices sim_index

    :param target_pos: ndarray, shape (..., 3, n_particles), positional data of target at each time step
    :param target_vel: ndarray, shape (..., 3, n_particles), velocity data of target at each time step
    :param cosmo: ndarray, shape (...,), cosmology data for each time step
    :param sim_index: int, index of the time step to visualize
    :param n_mesh: int, number of mesh points in each dimension
    :param new_nmesh: Number of mesh points for plot of higher resolution.
    :param scale_factors: ndarray, scale factors at which the solution is desired.

    :param save: bool, flag indicating whether to save the plots
    :param path: str, path to save the plots
    :return: None

    This method visualizes the n-body simulation and particle mesh simulation for a given time step.

    The method first plots the n-body simulation results. It creates a 4x4 grid of subplots, where each subplot shows
    the density of particles in a slice of the simulation volume. The plot uses the cloud-in-cell (CIC) interpolation
    to estimate the density at each mesh point. The plot is displayed or saved based on the `save` parameter.

    Next, the method runs the particle mesh simulation using the position and velocity data of the target at the
    given time step. It plots the results similar to the n-body simulation, but in this case, it also calculates the
    maximum density from the n-body simulation and uses it to set the maximum value for the color map. Again,
    the plot is displayed or saved based on the `save` parameter.

    Finally, the method plots the residuals between the particle mesh simulation and the n-body simulation. It
    follows a similar process as the previous plots but calculates the difference in density between the two
    simulations. The maximum density from the n-body simulation is used as the maximum value for the color map. The
    plot is displayed or saved based on the `save` parameter.
    """

    def plot_nbody_simulation(dataPos, save, path, n_mesh, new_nmesh):
        """
        Helper function to draw and save/display n-body simulation plot.

        :param dataPos: ndarray, positional data of target
        :param save: bool, flag indicating whether to save the plot
        :param path: str, path to save the plot
        :param n_mesh: int, number of mesh points in each dimension
        :param new_nmesh: Number of mesh points for plot of higher resolution.
        :return: None
        """

        plt.figure(figsize=[10, 10])  # Set up the figure
        plt.suptitle("Nbody")  # Add a title to the figure
        for i in range(16):  # Loop over subplots
            plt.subplot(4, 4, i + 1)  # Layout for subplots
            plt.imshow(cic_paint(jnp.zeros([new_nmesh] * 3), dataPos[::2][i] * new_nmesh / n_mesh).sum(axis=0),
                       cmap='gist_stern', vmin=0)
        nbody_image_path = f"{path}/nbody_{sim_index}.png"
        if save:  # Save or display the figure based on 'save' value
            plt.savefig(nbody_image_path)
        else:
            plt.show()

    def plot_pm_simulation(dataPos, pos_pm, n_mesh, save, path, new_nmesh):
        """
        Helper function to draw and save/display particle mesh simulation plot.

        :param dataPos: ndarray, positional data of target
        :param pos_pm: ndarray, positional data from particle mesh simulation
        :param n_mesh: int, number of mesh points in each dimension
        :param save: bool, flag indicating whether to save the plot
        :param path: str, path to save the plot
        :param new_nmesh: Number of mesh points for plot of higher resolution.
        :return: None
        """

        plt.figure(figsize=[10, 10])  # Set up the figure
        plt.suptitle("Particle mesh")  # Add a title to the figure
        for i in range(16):  # Loop over subplots
            plt.subplot(4, 4, i + 1)  # Layout for subplots
            plt.imshow(cic_paint(jnp.zeros([new_nmesh] * 3), pos_pm[::2][i] * new_nmesh / n_mesh).sum(axis=0),
                       cmap='gist_stern',
                       vmax=cic_paint(jnp.zeros([new_nmesh] * 3), dataPos[::2][i] * new_nmesh / n_mesh).sum(
                           axis=0).temp_min(), vmin=0)
        pm_image_path = f"{path}/pm_{sim_index}.png"
        if save:  # Save or display the figure based on 'save' value
            plt.savefig(pm_image_path)
        else:
            plt.show()

    def plot_residuals_simulation(dataPos, pos_pm, n_mesh, save, path, new_nmesh):
        """
        Helper function to draw and save/display residuals simulation plot.

        :param dataPos: ndarray, positional data of target
        :param pos_pm: ndarray, positional data from particle mesh simulation
        :param n_mesh: int, number of mesh points in each dimension
        :param save: bool, flag indicating whether to save the plot
        :param path: str, path to save the plot
        :param new_nmesh: Number of mesh points for plot of higher resolution.
        :return: None
        """

        plt.figure(figsize=[10, 10])  # Set up the figure
        plt.suptitle("PM - nbody residuals")  # Add a title to the figure
        for i in range(16):  # Loop over subplots
            plt.subplot(4, 4, i + 1)  # Layout for subplots
            plt.imshow(
                cic_paint(jnp.zeros([new_nmesh] * 3), pos_pm[::2][i] * new_nmesh / n_mesh).sum(axis=0) - cic_paint(
                    jnp.zeros([new_nmesh] * 3), dataPos[::2][i] * new_nmesh / n_mesh).sum(axis=0), cmap='gist_stern',
                vmax=cic_paint(jnp.zeros([new_nmesh] * 3), dataPos[::2][i] * new_nmesh / n_mesh).sum(axis=0).temp_min(),
                vmin=0)
        residuals_image_path = f"{path}/pm_nbody_residuals_{sim_index}.png"
        if save:  # Save or display the figure based on 'save' value
            plt.savefig(residuals_image_path)
        else:
            plt.show()

    # Extract positional and velocity data for a specific simulation
    dataPos = target_pos[sim_index]
    dataVel = target_vel[sim_index]
    cosmo = cosmo[sim_index]

    if save and not os.path.exists(path):  # Checks if directory exists
        os.makedirs(path)  # Creates directory if it doesn't exist

    # Call helper functions to draw and save/display plots for nbody, particle mesh and residuals
    plot_nbody_simulation(dataPos, save, path, n_mesh, new_nmesh)
    pos_pm, vel_pm = run_pm(initial_pos=dataPos[0], initial_vel=dataVel[0], redshifts=scale_factors, cosmo=cosmo,
                            n_mesh=n_mesh, use_redshifts=False)
    plot_pm_simulation(dataPos, pos_pm, n_mesh, save, path, new_nmesh)
    plot_residuals_simulation(dataPos, pos_pm, n_mesh, save, path, new_nmesh)


def visualize_with_correction(target_pos, target_vel, cosmo, sim_index, n_mesh, new_nmesh, scale_factors, model, params,
                              save=False, path=""):
    """
    Function to visualize the n-body simulation and particle mesh simulation results.

    :param target_pos: ndarray, positional data of target at each time step
    :param target_vel: ndarray, velocity data of target at each time step
    :param cosmo: ndarray, cosmology data for each time step
    :param sim_index: index of the time step to visualize
    :param n_mesh: The number of bins to use in 3D histogramming (cic_paint)
    :param new_nmesh: Number of mesh points for plot of higher resolution.
    :param scale_factors: ndarray, scale factors at which the solution is desired
    :param model: Callable, function reference for the model, e.g., neural_nbody_ode
    :param params: List, list of parameters for the model
    :param save: bool, if True, save the plots to the path, else show the plot
    :param path: str, path to save the plots
    :return: None
    """

    def prepare_simulation_data(target_pos, target_vel, cosmo, sim_index):
        """
        This function is used to prepare the simulation data.

        :param target_pos: Expected to be position data for the target.
        :param target_vel: Expected to be velocity data for the target.
        :param cosmo: A parameter related to the cosmology of the simulation.
        :param sim_index: The index of the simulation in the ensemble of simulations.
        :return: Position and velocity data placed on the GPU and cosmological parameters related to the simulation.
        """
        gpus = jax.devices("gpu")  # Getting all available GPUs
        dataPos = jax.device_put(target_pos[sim_index], gpus[0])  # Placing position data to the GPU
        dataVel = jax.device_put(target_vel[sim_index], gpus[0])  # Placing velocity data to the GPU
        cosmo = cosmo[sim_index]  # Selecting cosmological data based on the simulation index
        return dataPos, dataVel, cosmo

    def save_or_show_plot(save, path, file_name):
        """
        This function manages the saving or showing of the plot.

        :param save: A boolean indicating whether to save the figure.
        :param path: The location to save the figure, if save is True.
        :param file_name: The name of the file to save the figure as.
        :return: None
        """
        # file_name = f"Results/{path.split('/')[1]}/{file_name}_{path.split('/')[2].split('.')[0]}.png"  # Defining the full file path
        file_path = path + "/" + file_name
        if save and not os.path.exists(path):  # Checks if directory exists
            os.makedirs(path)  # Creates directory if it doesn't exist
        if save:
            plt.savefig(file_path)  # Conditionally save the figure
        else:
            plt.show()

    def plot_image_plt(image_data, vmax, save, path, file_name):
        """
        This function is used to create a 4x4 grid of plots.

        :param image_data: The data to plot.
        :param vmax: The maximum data to plot
        :param save: A boolean indicating whether to save the figure.
        :param path: The location to save the figure, if save is True.
        :param file_name: The name of the file to save the figure as.
        :return: None
        """
        plt.figure(figsize=[10, 10])  # Preparing a 4x4 grid of plots
        plt.suptitle(" ".join(file_name.split("_")[:-1]))  # Add a title to the figure
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(image_data[i], vmax=vmax[i], vmin=0, cmap='gist_stern')  # Painting the image with received data
        save_or_show_plot(save, path, file_name)  # Save or show the plot

    # Prepare data for the simulation
    # It will transfer data to GPU and select cosmological data for the current simulation index
    dataPos, dataVel, cosmo = prepare_simulation_data(target_pos, target_vel, cosmo, sim_index)

    # Creating n-body simulation plot
    vmax = [cic_paint(jnp.zeros([new_nmesh] * 3), dataPos[::2][i] * new_nmesh / n_mesh).sum(axis=0).temp_min() for i in
            range(16)]  # Get max value for color normalization
    image_data = [cic_paint(jnp.zeros([new_nmesh] * 3), dataPos[::2][i] * new_nmesh / n_mesh).sum(axis=0) for i in
                  range(16)]  # Painting the image with received data
    plot_image_plt(image_data, vmax, save, path, f"nbody_{sim_index}.png")

    # Run and plot Particle-Mesh simulation with correction and without correction
    pos_pmCorrected, vel_pmCorrected = run_pm_with_correction(pos=dataPos[0], vels=dataVel[0],
                                                              scale_factors=scale_factors, cosmo=cosmo,
                                                              n_mesh=n_mesh, model=model, params=params)
    pos_pm, vel_pm = run_pm(initial_pos=dataPos[0], initial_vel=dataVel[0], redshifts=scale_factors, cosmo=cosmo,
                            n_mesh=n_mesh, use_redshifts=False)

    # Creating Particle-Mesh simulation plot
    image_data = [cic_paint(jnp.zeros([new_nmesh] * 3), pos_pm[::2][i] * new_nmesh / n_mesh).sum(axis=0) for i in
                  range(16)]  # Painting the image with received data
    plot_image_plt(image_data, vmax, save, path, f"pm_{sim_index}.png")

    # Plot residual of PM simulation and n-body simulation
    image_data = [cic_paint(jnp.zeros([new_nmesh] * 3), pos_pm[::2][i] * new_nmesh / n_mesh).sum(axis=0) - cic_paint(
        jnp.zeros([new_nmesh] * 3), dataPos[::2][i] * new_nmesh / n_mesh).sum(axis=0) for i in
                  range(16)]  # Painting the image with received data
    plot_image_plt(image_data, vmax, save, path, f"pm_nbody_residuals_{sim_index}.png")

    # Creating PM simulation with correction plot
    image_data = [cic_paint(jnp.zeros([new_nmesh] * 3), pos_pmCorrected[::2][i] * new_nmesh / n_mesh).sum(axis=0) for i
                  in
                  range(16)]  # Painting the image with received data
    plot_image_plt(image_data, vmax, save, path, f"pm_corrected_{sim_index}.png")

    # Plot residual of PM simulation with correction and n-body simulation
    image_data = [
        cic_paint(jnp.zeros([new_nmesh] * 3), pos_pmCorrected[::2][i] * new_nmesh / n_mesh).sum(axis=0) - cic_paint(
            jnp.zeros([new_nmesh] * 3), dataPos[::2][i] * new_nmesh / n_mesh).sum(axis=0) for i in
        range(16)]  # Painting the image with received data
    plot_image_plt(image_data, vmax, save, path, f"pm_corrected_nbody_residuals_{sim_index}.png")

    # Plot residual of PM simulation with correction and PM simulation without correction
    image_data = [
        cic_paint(jnp.zeros([new_nmesh] * 3), pos_pmCorrected[::2][i] * new_nmesh / n_mesh).sum(axis=0) - cic_paint(
            jnp.zeros([new_nmesh] * 3), pos_pm[::2][i] * new_nmesh / n_mesh).sum(axis=0) for i in
        range(16)]  # Painting the image with received data
    plot_image_plt(image_data, vmax, save, path, f"pm_corrected_pm_residuals_{sim_index}.png")


"""
Example usage
"""
if __name__ == "__main__":
    # Initialize parameters
    sim = 0  # Simulation index
    nMesh = 32  # Number of mesh points
    boxSize = [25.0, 25.0, 25.0]  # Simulation box size

    # Initialize model
    model, _ = initialize_model(n_mesh=nMesh, model_name="Default", n_knots=32, latent_size=64)
    modelPath = "../Model/MyModel_nMesh32_LH100-499_Lr0.001_regularization_vel/model395.pkl"
    with open(modelPath, 'rb') as file:
        params = pickle.load(file)

    # Load Latin Hypercube simulation data
    targetP, targetV, z, cosmology = load_lh([sim], boxSize, nMesh, path="../CamelsSims", debug=True)
    scale_factors = 1 / (1 + jnp.array(z))

    # Visualize PM data without correction
    visualize(targetP, targetV, cosmology, sim, nMesh, 256, scale_factors, save=True, path="../Results")

    # Visualize PM data with correction
    visualize_with_correction(targetP, targetV, cosmology, sim, nMesh, 256, scale_factors, model, params,
                              save=True, path="../Results")
