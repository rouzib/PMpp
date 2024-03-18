# PMpp

PMpp is a Python-based deep learning project designed for training in the field of cosmology. It uses a modified version of [JaxPM](https://github.com/DifferentiableUniverseInitiative/JaxPM), a library for efficient particle mesh computations. 

This method is derived from the routines found in the [jaxpm-paper](https://github.com/DifferentiableUniverseInitiative/jaxpm-paper/tree/v_icml).

The project simulates the evolution of dark matter particles in the universe using the Particle-Mesh (PM) method and follows the implementation as outlined in this [paper](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_177.pdf). In other words, it's the code accompanying the paper. 

The project implements a neural network using the JAX library capable of predicting particle positions in cosmological simulations based on starting conditions and cosmological parameters.## 🔧 Installation

Before starting, ensure that you have [Python 3.10](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/) installed.

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/YourGithubUsername/PMpp.git
    ```

2. Navigate to the project directory.

    ```bash
    cd PMpp
    ```

3. Install the necessary packages using the provided `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

The main script used in this tool is `PMpp.py`, designed to train the model and save parameters after every iteration.

Before executing, ensure that the necessary constants are initialized correctly in the section clearly labelled "Defined Constants".

To execute:

```bash
python PMpp.py
```

## 📚 Dependencies

This project depends on the following libraries:

- [jax](https://github.com/google/jax)
- [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo)
- [matplotlib](https://matplotlib.org)
- [numpy](https://numpy.org)
- [optax](https://github.com/deepmind/optax)
- [tqdm](https://github.com/tqdm/tqdm)