"""
Date: 03/10/2024

This Python file is primarily used for calculation and simulation in cosmology. The functionalities include
calculating power spectrums, and cross correlations. The code also normalizes, initializes, and carries out
manipulation and compensation of data in a mesh system. Provision for running both the Particle-Mesh simulations with
or without model correction is also available. The main function orchestrates the entire processing sequence,
from initialization to result plotting.
"""

import os
import pickle

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from PMpp import load_lh, run_sim_with_model, run_pm
from camels_utils import normalize_by_mesh
from jaxpm.painting import cic_paint, compensate_cic
from jaxpm.utils import power_spectrum, cross_correlation_coefficients

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def read_pks(sims: range, n_mesh: int = 64):
    """
    The keys are of the type jax_cosmo.core.Cosmology (transform this object as a tuple: cosmology.tree_flatten()[0])
    The values are the corresponding [k, pk_nbody, pk_pm, pk_pm_corrected]
    Ex:
    array of all cosmologies:
        list(pks.keys())
    array of all k:
        np.array(list(pks.values()))[:, 0]
    array of all pk_nbody:
        np.array(list(pks.values()))[:, 1]

    :param sims: A range of simulation numbers.
    :param n_mesh: The mesh size (default is set to 64).
    :return: A dictionary containing the power spectra and the simulation indexes.

    """
    pks = {}
    simulations = []
    for i in sims:
        if os.path.exists("Pk/" + f"pk_LH{i}_{n_mesh}.pkl"):
            with open("Pk/" + f"pk_LH{i}_{n_mesh}.pkl", 'rb') as file:
                pks = {**pks, **pickle.load(file)}
            simulations.append(i)
        else:
            print("Pk/" + f"pk_LH{i}_{n_mesh}.pkl does not exist.")
    return pks, simulations


def read_ccs(sims: range, n_mesh: int = 64):
    """
    Reads cross correlations (ccs) from pickle files for a given range of simulations and mesh size.

    :param sims: A range of simulation numbers.
    :param n_mesh: The mesh size (default is set to 64).
    :return: A dictionary containing the cross correlations.
    """
    ccs = {}
    for i in sims:
        if os.path.exists("Pk/" + f"cc_LH{i}_{n_mesh}.pkl"):
            with open("Pk/" + f"cc_LH{i}_{n_mesh}.pkl", 'rb') as file:
                ccs = {**ccs, **pickle.load(file)}
        else:
            print("Pk/" + f"cc_LH{i}_{n_mesh}.pkl does not exist.")
    return ccs


def main(i):
    """
    :param i: The parameter `i` is an input indicating the specific value to be used in the method.
    :return: There is no explicit return value for this method.

    This method `main` performs a series of operations to run a simulation using the JaxPM library. It takes the
    parameter `i` as an input and performs the following steps:

    1. Initialize the mesh size (`nMesh`) and box size (`boxSize`).
    2. Load the training data using the `load_lh` function with the specified `i`, `boxSize`, `nMesh`, and other optional parameters.
    3. Normalize the position and velocity of the target data using the `normalize_by_mesh` function.
    4. Set the initial position and velocity based on the normalized target data.
    5. Perform a simulation with position and velocity correction using the `run_sim_with_model` function.
    6. Perform a simulation without position and velocity correction using the `run_pm` function.
    7. Calculate the power spectrum for each case (N-body, PM, and PM with correction) using the `power_spectrum` function.
    8. Compute the cross-correlations for PM and PM with correction using the `cross_correlation_coefficients` function.
    9. Plot the power spectrum results using matplotlib.
    10. Print a message indicating the completion of processing for the specified `i`.

    Note: There are commented out lines of code that save the power spectra and cross-correlations to pickle files."""
    # Initialize Mesh size and boxSize
    nMesh = 32
    boxSize = [25.0, 25.0, 25.0]

    # Load training data
    targetP, targetV, z, cosmology = load_lh([i], boxSize, nMesh, path="CamelsSims", normalize=False, debug=True)
    # Normalize position and velocity by mesh size
    targetP, targetV = targetP[0], targetV[0]
    targetP, targetV = normalize_by_mesh(targetP, targetV, boxSize[0], nMesh)
    # Set initial position and velocity
    initialP, initialV = targetP[0], targetV[0]
    cosmology = cosmology[0]

    """
    This is more memory efficient, but takes longer to run
    
    targetP, targetV, z, cosmology = load_lh([i], boxSize, 256, path="CamelsSims", normalize=False, debug=True)
    cosmology = cosmology[0]
    print("Loaded")
    targetP, targetV = targetP[0], targetV[0]
    downsampling_factor = len(targetP[0]) // nMesh ** 3
    key = jax.random.PRNGKey(0)
    permuted_indices = jax.random.permutation(key, len(targetP[0]))
    selected_indices = permuted_indices[: len(targetP[0]) // downsampling_factor]
    targetP = jnp.take(targetP, selected_indices, axis=1)
    testVel = jnp.take(targetV, selected_indices, axis=1)
    targetP, testVel = normalize_by_mesh(targetP, testVel, boxSize[0], nMesh)
    targetP, testVel = targetP[1:], testVel[1:]
    initialP, initialV = targetP[0], testVel[0]"""

    # ----- PM WITH CORRECTION -----
    # Run PM with correction
    pmPosCorr, pmVelCorr = run_sim_with_model(initial_pos=initialP, initial_vel=initialV, redshifts=z,
                                              cosmo=cosmology,
                                              n_mesh=nMesh)

    # ----- PM WITHOUT CORRECTION -----
    # Run PM without correction
    pmPos, pmVel = run_pm(initial_pos=initialP, initial_vel=initialV, redshifts=z, cosmo=cosmology, n_mesh=nMesh)

    # Calculate the power spectrum for each case (N-body, PM, and PM with correction)
    k, pk_nbody = power_spectrum(
        compensate_cic(cic_paint(jnp.zeros([nMesh, nMesh, nMesh]), targetP[-1])),
        boxsize=np.array([25.0] * 3), kmin=np.pi / 25.0, dk=2 * np.pi / 25.0)

    k, pk_pm = power_spectrum(compensate_cic(cic_paint(jnp.zeros([nMesh, nMesh, nMesh]), pmPos[-1])),
                              boxsize=np.array([25.0] * 3), kmin=np.pi / 25.0, dk=2 * np.pi / 25.0)

    k, pk_pm_corr = power_spectrum(
        compensate_cic(cic_paint(jnp.zeros([nMesh, nMesh, nMesh]), pmPosCorr[-1])),
        boxsize=np.array([25.0] * 3), kmin=np.pi / 25.0, dk=2 * np.pi / 25.0)

    # Compute the cross correlations for PM and PM with correction
    _, cross_c_corr = cross_correlation_coefficients(
        compensate_cic(cic_paint(jnp.zeros([nMesh, nMesh, nMesh]), targetP[-1])),
        compensate_cic(cic_paint(jnp.zeros([nMesh, nMesh, nMesh]), pmPosCorr[-1])),
        boxsize=np.array([25.] * 3),
        kmin=np.pi / 25.,
        dk=2 * np.pi / 25.)

    k, cross_c_pm = cross_correlation_coefficients(
        compensate_cic(cic_paint(jnp.zeros([nMesh, nMesh, nMesh]), targetP[-1])),
        compensate_cic(cic_paint(jnp.zeros([nMesh] * 3), pmPos[-1])),
        boxsize=np.array([25.] * 3), kmin=np.pi / 25., dk=2 * np.pi / 25.)

    """
    Save the power spectra. The power spectra and cross correlations can be retrieved with the previous functions
    
    with open("Pk/" + f"pk_LH{i}_{nMesh}.pkl", 'wb') as file:
        pickle.dump({cosmology: [k, pk_nbody, pk_pm, pk_pm_corr]}, file)

    with open("Pk/" + f"cc_LH{i}_{nMesh}.pkl", 'wb') as file:
        pickle.dump({cosmology: [k, cross_c_pm, cross_c_corr]}, file)"""

    # Plot the power spectrum results
    plt.loglog(k, pk_nbody, label="N-body")
    plt.loglog(k, pk_pm, label="JaxPM w/o correction")
    plt.loglog(k, pk_pm_corr, label="JaxPM w correction")
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.title(r"LH "f"{i}")
    plt.legend()
    plt.show()

    print(f"Finished processing {i}")


if __name__ == '__main__':
    main(0)
