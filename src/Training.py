"""
Date: 03/15/2023

This Python script is designed for deep learning training in the field of cosmology. It uses the JAX library for creating
and training a machine learning model, designed to predict the particle positions in a cosmological simulation based on
the starting conditions and cosmological parameters.

The script is designed to be run as a standalone and configured with constants in the global scope. It works by loading
simulation snapshots from a directory ("CamelsSims" by default) and using these snapshots to define the loss function
for the training. With the loss function defined, it proceeds to train the model, by default, for 1000 steps.

The script has been optimised for GPU computation and supports optional loss calculations such as velocity error and
Power Spectrum loss. Additionally, it carries out validation over the course of training and saves the model
parameters after every iteration. It also provides loss visualization by plotting training and validation losses
against epochs and saves it into PNG file.
"""

import os
import pickle
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.experimental.ode import odeint
from tqdm import tqdm

from src.Models import initialize_model
from src.PMpp import load_lh
from jaxpm.painting import cic_paint, compensate_cic
from jaxpm.pm import make_neural_ode_fn
from jaxpm.utils import power_spectrum

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

# Defined Constants
LOCAL = True
N_MESH = 32
BOX_SIZE = [25.0, 25.0, 25.0]
N_KNOTS = 16
LATENT_SIZE = 32
LEARNING_RATE = 0.001
N_STEPS = 1000
VELOCITY_LOSS = False
PK_LOSS = True
REGULARIZATION = False
MODEL_NAME = "Default"
STARTS_WITH_PATH = ""
SNAPSHOT_LIST = range(34)
TEST_IDX = jnp.array(list(range(0, 1)))
VAL_IDX = jnp.array(list(range(100, 101)))
DEBUG = True


def form_file_path(base_path, file_prefix, lh_index, n_mesh):
    """
    Form a file path based on the given parameters.

    :param base_path: The base path for the file.
    :param file_prefix: The file prefix for the file.
    :param lh_index: The lh index for the file.
    :param n_mesh: The n mesh for the file.
    :return: The formed file path.
    """
    return f"{base_path}/{file_prefix}_{lh_index}_{n_mesh}.npy"


def check_for_pk(lh_index: int, n_mesh: int, data_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate and return the power spectrum of a simulation of  a given index.

    This function will attempt to load the power spectrum data from disk using the provided `lh_index`
    and `n_mesh`. If the data is not found, it will calculate the power spectrum from `data_pos` or load
    the position data from file if `data_pos` is None.

    :param lh_index: The index of the LH (Latin Hypercube) for which to check the power spectrum.
    :param n_mesh: The number of mesh points along each dimension.
    :param data_pos: Optional. The position data of the LH. If not provided, it will be loaded from a file using the `lh_index`.
    :return: The wave numbers (`k`) and the power spectrum (`pk_nbody`) of the LH.

    """
    # Defining the base path for data and create it if necessary
    BASE_PATH = "../pk" if LOCAL else "/home/rouzib/scratch"
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH, exist_ok=True)

    # Defining wave number parameter
    K_MIN = np.pi / BOX_SIZE[0]
    DK = 2 * K_MIN

    # Variable to check if data is found or not
    not_found = False

    # Defining the file path
    k_path = form_file_path(BASE_PATH, "k_LH", lh_index, n_mesh)
    pk_nbody_path = form_file_path(BASE_PATH, "PK_LH", lh_index, n_mesh)

    # Try to load data
    try:
        k = jnp.load(k_path)
        pk_nbody = jnp.load(pk_nbody_path)
    except:  # If files not found, print error and set not_found to True
        print(f"{k_path} or {pk_nbody_path} could not be found")
        not_found = True

    # If data not found
    if not_found:
        # If position data not provided, load it from file
        if data_pos is None:
            print(f"Loaded {lh_index} because it was not passed")
            data_pos = load_lh([lh_index], BOX_SIZE, n_mesh, path="../CamelsSims" if LOCAL else "/home/rouzib/scratch",
                               cpu_memory=True, debug=False)[0][0]

        # Transfer data to GPU
        data_pos = jax.device_put(data_pos, gpus[0])

        # Calculate the power spectrum
        k, pk_nbody = power_spectrum(
            compensate_cic(cic_paint(jnp.zeros([n_mesh, n_mesh, n_mesh]), data_pos[-1])),
            boxsize=np.array(BOX_SIZE), kmin=K_MIN, dk=DK)

        # Save the calculated power spectrum to file for future use
        jnp.save(pk_nbody_path, pk_nbody)
        jnp.save(k_path, k)

        # Delete the position data to free up memory
        del data_pos

    # Return the wave numbers and the power spectrum
    return k, pk_nbody


@partial(jax.jit, static_argnames=["model", "n_mesh", "velocity_loss", "pk_loss", "regularization"])
def loss_fn(params, cosmology, target_pos, target_vel, scales, pks, box_size, n_mesh, model, velocity_loss=False,
            pk_loss=False, regularization=False):
    """
    Compute the loss function for a given set of input parameters.

    :param params: A dictionary of neural network parameters.
    :param cosmology: A list of cosmological parameters.
    :param target_pos: The target position.
    :param target_vel: The target velocity.
    :param scales: The scales.
    :param pks: The list of power spectra.
    :param box_size: The size of the box.
    :param n_mesh: The number of meshes.
    :param model: The neural network model.
    :param velocity_loss: Boolean indicating whether to include velocity loss. Default is False.
    :param pk_loss: Boolean indicating whether to include power spectrum loss. Default is False.
    :param regularization: Boolean indicating whether to include regularization loss. Default is False.
    :return: The computed loss value.
    """
    vel_contribution = 1e-2
    regularization_contribution = 1e-1
    pk_contribution = 1

    # Exception handling: Checking for empty inputs for target position and velocity
    if len(target_pos) == 0 or len(target_vel) == 0:
        raise Exception(f"No data given: len(target_pos) = {len(target_pos)}, len(target_vel) = {len(target_vel)}")

    # Initializing Mean Squared Error (MSE)
    mse = 0

    # Calculate loss if pk_loss flag is set to True
    if pk_loss:
        # Iterating over each element of target position, target velocity, cosmology and power spectrum
        for pos, vel, cosmo, target_pk in zip(target_pos, target_vel, cosmology, pks):
            # Setting initial conditions for the ODE solver
            pos_pm, vel_pm = odeint(make_neural_ode_fn(model, [n_mesh, n_mesh, n_mesh]),
                                    [pos[0], vel[0]], scales, cosmo, params, rtol=1e-5, atol=1e-5)

            # Applying modulo operation to position prediction with number of mesh points to keep the positions on the
            # mesh
            pos_pm %= n_mesh

            # Calculating difference between predicted and true position
            dx = pos_pm - pos

            # Reflect the particle across the boundary if it goes beyond it.
            dx = dx - n_mesh * jnp.round(dx / n_mesh)

            # Calculate the squared differences
            sim_mse = jnp.sum(dx ** 2, axis=-1)

            # If velocity_loss flag is set, include the loss due to velocity prediction error
            if velocity_loss:
                dv = vel_pm - vel
                sim_mse += vel_contribution * jnp.sum(dv ** 2, axis=-1)

            # Add all the calculated losses to get the total mse
            mse += jnp.mean(sim_mse)

            # Calculation of power spectrum loss
            pk = jax.vmap(lambda x: power_spectrum(compensate_cic(cic_paint(jnp.zeros([n_mesh, n_mesh, n_mesh]), x)),
                                                   boxsize=np.array([25.] * 3),
                                                   kmin=np.pi / 25., dk=2 * np.pi / 25.)[1])(pos_pm)
            mse += pk_contribution * jnp.mean(jnp.sum((pk / target_pk - 1) ** 2, axis=-1))
    else:
        # Calculate only the mean square error loss without considering the power spectrum loss
        for pos, vel, cosmo in zip(target_pos, target_vel, cosmology):
            # Setting initial conditions for the ODE solver
            pos_pm, vel_pm = odeint(make_neural_ode_fn(model, [n_mesh, n_mesh, n_mesh]),
                                    [pos[0], vel[0]], scales, cosmo, params, rtol=1e-5, atol=1e-5)

            # Applying modulo operation to position prediction with number of mesh points to keep the positions on the
            # mesh
            pos_pm %= n_mesh

            # Calculating difference between predicted and true position
            dx = pos_pm - pos

            # Reflect the particle across the boundary if it goes beyond it.
            dx = dx - n_mesh * jnp.round(dx / n_mesh)

            # Calculate the squared differences
            sim_mse = jnp.sum(dx ** 2, axis=-1)

            # If velocity_loss flag is set, include the loss due to velocity prediction error
            if velocity_loss:
                dv = vel_pm - vel
                sim_mse += vel_contribution * jnp.sum(dv ** 2, axis=-1)

            # Add all the calculated losses to get the total mse
            mse += jnp.mean(sim_mse)

    # If regularization flag is set, add regularization loss to the final mse
    if regularization:
        regularizationTerm = 0
        for i in params.values():
            regularizationTerm += jnp.sum(jnp.array(list(i.values())[0]) ** 2)
        mse += regularization_contribution * regularizationTerm

    return mse / len(target_pos)  # Return mean mse by averaging over all targets


def get_model(model_name, model_path=""):
    """
    Get the initialized model and parameters.

    :param model_name: Name of the model.
    :param model_path: (optional) Path to the saved model.
    :type model_name: str
    :type model_path: str
    :return: Initialized model, parameters, and start epoch.
    :rtype: tuple
    """
    if DEBUG:
        print("Initializing model")
    model, params = initialize_model(n_mesh=N_MESH, model_name=model_name, n_knots=N_KNOTS, latent_size=LATENT_SIZE)

    start_epoch = 0
    if model_path != "":
        with open(model_path, 'rb') as file:
            params = pickle.load(file)

        start_epoch = int(model_path.split(".")[0].split("model")[1])

    return model, params, start_epoch


if __name__ == '__main__':
    model_path = f"../Model/{MODEL_NAME}_nMesh{N_MESH}_LH{TEST_IDX[0]}-{TEST_IDX[-1]}" \
                 f"_Lr{LEARNING_RATE}_nKnots{N_KNOTS}_ls{LATENT_SIZE}{'_regularization' if REGULARIZATION else ''}" \
                 f"{'_vel' if VELOCITY_LOSS else ''}{'_pk' if PK_LOSS else ''}/"

    # load LH sims
    target_pos, target_vel, z, planck_cosmology = load_lh(TEST_IDX, BOX_SIZE, N_MESH,
                                                          path="../CamelsSims" if LOCAL else "/home/rouzib/scratch",
                                                          normalize=True, cpu_memory=True, debug=DEBUG)

    scale_factors = 1 / (1 + jnp.array(z))

    # Get power spectra for all the training simulations
    if PK_LOSS:
        if DEBUG:
            print("Loading pk")
        target_pks = []
        for i in range(len(TEST_IDX)):
            _, target_pk = check_for_pk(TEST_IDX[i], N_MESH, target_pos[i])
            target_pk = jax.device_put(target_pk, cpus[0])
            target_pks.append(target_pk)
        target_pks = jnp.array(target_pks, copy=False)

        val_pks = []
        for i in range(len(VAL_IDX)):
            _, val_pk = check_for_pk(VAL_IDX[i], N_MESH, target_pos[i])
            val_pk = jax.device_put(val_pk, cpus[0])
            val_pks.append(val_pk)
        val_pks = jnp.array(val_pks, copy=False)

    model, params, start_epoch = get_model(MODEL_NAME, STARTS_WITH_PATH)

    os.makedirs(model_path, exist_ok=True)

    # Put all the data on GPU if needed
    """print(target_pos.device_buffer.device())  
    target_pos = jax.device_put(target_pos, gpus[0])
    target_vel = jax.device_put(target_vel, gpus[0])
    target_pks = jax.device_put(target_pks, gpus[0])
    print(target_pos.device_buffer.device())"""

    if DEBUG:
        print("Starting training")
    batchSize = 2
    numBatch = len(TEST_IDX) // batchSize
    if numBatch == 0:
        numBatch = 1
        batchSize = len(TEST_IDX)
    if numBatch * batchSize != len(TEST_IDX):
        # left over sims
        numBatch += 1

    if DEBUG:
        print(f"BatchSize = {batchSize} and numBatch = {numBatch}")

    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(10.0),
                    optax.adam(LEARNING_RATE)),
        N_STEPS * numBatch)
    opt_state = optimizer.init(params)

    losses = []
    val_losses = []

    loss_fn_compiled = jax.value_and_grad(loss_fn)

    pbar = tqdm(range(start_epoch, N_STEPS))
    for step in pbar:
        tot_loss = 0
        tot_val_loss = 0

        # shuffle the data for each batch
        key = jax.random.PRNGKey(step * 10)
        indexes = jnp.array(range(len(TEST_IDX)))
        shuffledIndexes = jax.random.permutation(key, indexes, independent=True)

        # training loop
        for i in range(numBatch):
            # handle the ids in the batch and the case len(batch) == 0
            startIdx = i * batchSize
            endIdx = min((i + 1) * batchSize, len(TEST_IDX))
            if startIdx == endIdx:
                if DEBUG:
                    print(f"startIdx = endIdx: {startIdx} = {endIdx}")
                continue

            # load the data for each epoch
            if DEBUG:
                print(shuffledIndexes[startIdx:endIdx])
            target_pos, target_vel, z, planck_cosmology = load_lh(TEST_IDX[shuffledIndexes[startIdx:endIdx]], BOX_SIZE,
                                                                  N_MESH,
                                                                  path="../CamelsSims" if LOCAL else "/home/rouzib/scratch",
                                                                  normalize=True, cpu_memory=True,
                                                                  debug=DEBUG)
            tempPlanckCosmology = planck_cosmology
            tempPos = jax.device_put(target_pos, gpus[0])
            tempVel = jax.device_put(target_vel, gpus[0])

            """
            This code selects samples in the training dataset if all of the samples are loaded in planck_cosmology,
             target_pos and target_vel
            
            tempPlanckCosmology = [planck_cosmology[j] for j in shuffledIndexes[startIdx:endIdx]]

            # tempPlanckCosmology = planck_cosmology[shuffledIndexes[startIdx:endIdx]]
            tempPos = jax.device_put(target_pos[shuffledIndexes[startIdx:endIdx]], gpus[0])
            tempVel = jax.device_put(target_vel[shuffledIndexes[startIdx:endIdx]], gpus[0])"""

            # load the power spectra onto the gpu if necessary
            if PK_LOSS:
                tempPks = jax.device_put(target_pks[startIdx:endIdx], gpus[0])
            else:
                tempPks = jnp.array([])

            # compute loss
            loss, grads = loss_fn_compiled(params, tempPlanckCosmology, tempPos, tempVel, scale_factors,
                                           tempPks, BOX_SIZE[0], N_MESH, model, VELOCITY_LOSS, PK_LOSS, REGULARIZATION)

            if DEBUG:
                print(f"{shuffledIndexes[startIdx:endIdx]} loss is {loss}")

            # update the model
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            tot_loss += loss

        # totLoss /= numBatch

        # validation loop
        for i in range(len(VAL_IDX)):
            # load validation sim
            tempPos, tempVel, z, tempPlanckCosmology = load_lh([VAL_IDX[i]], BOX_SIZE, N_MESH,
                                                               path="../CamelsSims" if LOCAL else "/home/rouzib/scratch",
                                                               normalize=True, cpu_memory=False, debug=DEBUG)

            # load the power spectra onto the gpu if necessary
            if PK_LOSS:
                tempPks = jax.device_put(val_pks[i], gpus[0])
            else:
                tempPks = jnp.array([])

            # compute validation loss
            val_loss, _ = loss_fn_compiled(params, tempPlanckCosmology, tempPos, tempVel, scale_factors, tempPks,
                                           BOX_SIZE[0],
                                           N_MESH, model, VELOCITY_LOSS, PK_LOSS, REGULARIZATION)

            tot_val_loss += val_loss

        # update tqdm progress bar
        pbar.set_postfix({"Step": step, "Loss": tot_loss, "Val Loss": tot_val_loss})
        losses.append(tot_loss)
        val_losses.append(tot_val_loss)

        # save model every X step
        if step % 1 == 0:
            with open(model_path + f"model{step}.pkl", 'wb') as file:
                pickle.dump(params, file)

        # plot losses every X step
        if step % 5 == 0:
            plt.plot(range(len(losses)), losses, label="Training Loss")
            plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
            plt.title(f"Loss over {len(losses)} epochs")
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.savefig(model_path + f"/loss_{step}.png")
            plt.show()
            plt.clf()
