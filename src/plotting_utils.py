import jaxlib
import jax.numpy as jnp
from jax.experimental import io_callback
from matplotlib import pyplot as plt


def plot_particle_distribution_on_gpus(particles, force_mGPU=False):
    """
    Plot the x-axis position distribution of particles across GPUs.
    Particles are split evenly across GPUs, and unused particles are removed.

    Parameters:
    - particles: An instance of the Particles class.
    """
    # Retrieve positions, device count, and unused indices
    positions = particles.pos(jnp.float32, wrap=True)
    unused_index = particles.unused_index
    config = particles.conf

    # Ensure devices() exist
    devices = positions.devices() if hasattr(positions, "devices") else []
    num_devices = len(devices)

    if force_mGPU:
        num_devices = config.num_devices

    if num_devices == 0:
        raise ValueError("No devices detected for splitting particles.")

    bins = jnp.linspace(0, config.box_size[0], num=config.mesh_shape[0] + 1)

    total_count = 0

    # Plot distributions grouped by GPU
    plt.figure(figsize=(10, 6))
    if num_devices == 1:
        counts, _ = jnp.histogram(positions[:, 0], bins=bins)
        plt.bar(bins[:-1], counts, width=bins[1] - bins[0], alpha=0.5, label=f'{0}', align="edge")
        total_count += int(jnp.sum(counts))

    if num_devices > 1:
        start_idx = 0
        max_particles = positions.shape[0] // num_devices
        for i in range(num_devices):
            temp = positions[start_idx:start_idx + max_particles]
            unused_index_temp = unused_index[start_idx:start_idx + max_particles]
            start_idx += max_particles

            temp = temp[~unused_index_temp]

            counts, _ = jnp.histogram(temp[:, 0], bins=bins)
            plt.bar(bins[:-1], counts, width=bins[1] - bins[0], alpha=0.5, label=f'{i}', align="edge")
            total_count += int(jnp.sum(counts))

    # Beautify the plot
    plt.title(f"Particle X-Axis Distribution by GPU (n={total_count})", fontsize=16)
    plt.xlabel("X Position", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="GPU Index")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    xticks = jnp.linspace(0, config.box_size[0], num=8, endpoint=False)
    plt.xticks(xticks)

    # Display the plot
    plt.show()


def plot_pos_distribution(positions, config):
    bins = jnp.linspace(0, config.mesh_shape[0], num=config.mesh_shape[0] + 1)

    plt.figure(figsize=(10, 6))
    counts, _ = jnp.histogram(positions[:, 0], bins=bins)
    plt.bar(bins[:-1], counts, width=bins[1] - bins[0], alpha=0.5, align="edge")

    step = config.mesh_shape[0] // 8
    xticks = list(range(0, config.mesh_shape[0] + step, step))
    plt.xticks(xticks)

    # Beautify the plot
    plt.title(f"Particle X-Axis Distribution(n={int(jnp.sum(counts))})", fontsize=16)
    plt.xlabel("X Position", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_particle_bins(pos, nMesh, title=None, mask=None):
    """
    Plots the distribution of particles along the x-axis using histogram bins. The particle
    positions are calculated modulo the number of mesh bins to determine their placement
    within the defined range. Additionally, this function filters out particles with zero
    positions, calculates their counts within specific bins, and visualizes the data using
    a bar plot. The title of the plot can be customized or defaults to reflect the total
    number of particles.

    :param pos: 2D array containing the positions of the particles. Each row
                represents a particle, with the first column corresponding to
                the x-coordinate.
    :type pos: jax.numpy.ndarray
    :param nMesh: Number of global mesh bins used to define the range for the
                  histogram plot.
    :type nMesh: int
    :param title: Title of the plot. If not provided, defaults to show the total
                  number of particles in the title.
    :type title: str or None
    :return: None
    """
    if mask is not None:
        particles = pos[mask]
    else:
        particles = pos
    if type(title) is jaxlib.xla_extension.ArrayImpl:
        title = resolve_title(title.item())
    elif type(title) is not str:
        title = resolve_title(title)

    bins = jnp.linspace(0, nMesh, num=nMesh + 1)  # Define global bins (adjust as needed)
    total_count = 0
    particles = particles[jnp.any(particles != 0.0, axis=1)]
    particles_x = particles[:, 0] % nMesh
    counts, _ = jnp.histogram(particles_x, bins=bins)
    plt.bar(bins[:-1], counts, width=bins[1] - bins[0], alpha=0.5, align="edge")
    total_count += int(jnp.sum(counts))
    plt.xlabel("X position")
    plt.ylabel("Number of Particles")
    plt.title(f"Particle Distribution (N={total_count})" if title is None else title)
    step = nMesh // 8
    xticks = list(range(0, nMesh + step, step))
    plt.xticks(xticks)
    plt.show()


def plot_particle_bins_callback(pos, mask, nMesh, title_idx=None):
    """
    Wrapper for plotting particle bins in JIT context using host_callback.
    Args:
        pos: Positions of particles.
        nMesh: Mesh size.
        title_idx: Numeric index representing the title (instead of a string).
    """
    # Default to title_idx=0 if not provided
    title_idx = title_idx if title_idx is not None else 0

    # Use io_callback with JAX-compatible types only
    io_callback(
        lambda pos, nMesh, idx, mask: plot_particle_bins(pos, nMesh, idx, mask),
        (),
        pos, nMesh, title_idx, mask
    )


# Function to resolve numeric title index back into a string
def resolve_title(title_idx):
    title_map = {
        0: "All particles",
        1: "Particles in halo",
        2: "Particles that exited halo_slice",
        3: "Particles that entered halo_slice",
        4: "Particles that stayed in halo_slice",
        5: "In GPU slice",
        6: "Out of GPU slice and not in halo",
        7: "Particles that need to be removed",
        8: "Particles that need to be shared",
        9: "Particles that need to be shared to the left slice",
        10: "Particles that need to be shared to the right slice",
    }
    return title_map.get(title_idx, "Unknown Title")