# PM++: Multi-GPU Particle Mesh Cosmological Simulations

PM++ is an enhanced, multi-GPU version of the PMWD framework, designed for large-scale cosmological N-body simulations. This project extends the capabilities of PMWD to efficiently utilize multiple GPUs for simulating the evolution of dark matter structures in the universe.

## Key Features

- **Multi-GPU Support**: Efficiently distributes particle simulations across multiple GPUs using JAX
- **Memory Optimization**: Smart particle distribution and memory management for large-scale simulations
- **Lagrangian Perturbation Theory**: Includes LPT for generating realistic initial conditions
- **JIT Compilation**: Leverages JAX's just-in-time compilation for optimal performance

## Performance

PM++ can handle simulations with:
- **640³ particles** on 4 H100 GPUs (~27 GB memory per GPU)
- **1024³ particles** and beyond with appropriate hardware
- Significant per-device memory savings compared to single-GPU PMWD (4×27GB vs 70GB for equivalent simulations) enabling scaling of PM simulations.

## Project Structure

```
PM++_v2/
├── pmwd/                   # Original PMWD framework
│   ├── __init__.py         # Core PMWD imports
│   ├── particles.py        # Particle management
│   ├── nbody.py            # N-body simulation engine
│   ├── cosmology.py        # Cosmological models
│   └── ...                 # Other PMWD modules
├── src/                    # PM++ multi-GPU implementation
│   ├── particles.py        # Multi-GPU particle handling
│   ├── nbody.py            # Multi-GPU N-body simulations
│   ├── configuration.py    # PM++ configuration
│   └── ...                 # Enhanced PM++ modules
├── notebooks/              # Example notebooks
│   ├── pmpp_showcase.ipynb # Comprehensive usage examples
│   └── mGPU_pmwd_local.ipynb
├── CAMELS/                 # CAMELS simulation data
├── rewuirements.txt        # PM++ requirements (excludes notebook requirements)
└── README.md               # This file
```

## Installation

### Requirements

- Python 3.8+
- JAX with GPU support
- mcfit>=0.0.18 (with JAX support)
- Nvidia GPUs for NVCC backend

For the notebooks:
- NumPy
- Matplotlib
- h5py and hdf5plugin (for CAMELS data)
- readgadget (optional, for reading Gadget snapshots)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd PM++_v2
```

2. Install dependencies:
```bash
pip install jax[cuda] numpy matplotlib h5py hdf5plugin
```

3. For CAMELS data support, install readgadget:
```bash
pip install readgadget
```

## Quick Start

### Basic Multi-GPU Simulation

```python
import jax
from src.configuration import Configuration
from src.particles import Particles
from src.nbody import nbody
from src.cosmo import SimpleLCDM
from src.utils import create_compute_mesh

# Setup simulation parameters
num_ptcl = 256
box_size = 25.0  # Mpc/h
ptcl_grid_shape = (num_ptcl,) * 3
ptcl_spacing = box_size / num_ptcl

# Configure multi-GPU setup
num_devices = jax.device_count()
compute_mesh = create_compute_mesh(jax.devices()[:num_devices])

# Create configuration
conf = Configuration(
    ptcl_spacing=ptcl_spacing,
    ptcl_grid_shape=ptcl_grid_shape,
    compute_mesh=compute_mesh,
    max_ptcl_per_slice=int(num_ptcl**3 / num_devices * 1.8),
    max_share_ptcl=50000,
    max_share_gather_ptcl=120000
)

# Setup cosmology
cosmo = SimpleLCDM(conf)

# Initialize particles and run simulation
ptcl = Particles.from_pos_sharded(conf, positions, velocities)
nbody_jitted = jax.jit(nbody, static_argnames=("conf", "reverse"))
ptcl_final = nbody_jitted(ptcl, cosmo, conf)
```

### Working with CAMELS Data

```python
import readgadget
from pmwd import Particles, Configuration, SimpleLCDM, nbody

# Load CAMELS initial conditions
snapshot_filename = "CAMELS/ICs/ics"
header = readgadget.header(snapshot_filename)
pos = readgadget.read_block(snapshot_filename, "POS ", [1]) / 1e3  # Convert to Mpc/h
vel = readgadget.read_block(snapshot_filename, "VEL ", [1])

...see previous code

ptcl = Particles.from_pos_sharded(conf, pos, vel)
```

## Configuration Parameters

PM++ requires several key parameters for multi-GPU operation:

- **`max_ptcl_per_slice`**: Maximum particles per GPU device
- **`max_share_ptcl`**: Maximum particles shared between GPUs per step
- **`max_share_gather_ptcl`**: Maximum particles exchanged during gravity computation

These parameters should be tuned based on your simulation size and available GPU memory.

## Examples

See the `notebooks/` directory for comprehensive examples:

- **`pmpp_showcase.ipynb`**: Complete tutorial showing PM++ vs PMWD comparisons, CAMELS data loading, and large-scale simulation examples
- **`mGPU_pmwd_local.ipynb`**: Additional multi-GPU examples and benchmarks

## Credits and Acknowledgments

This project is based on **PMWD**. PM++ extends PMWD's capabilities to multi-GPU environments while maintaining the accuracy and reliability of the original framework.

## License

See **PMWD** license. 