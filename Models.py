"""
Date: 03/10/2024

This script contains the implementation of two primary classes:
`NeuralSplineFourierFilter` and `Model1`. The classes represent rotationally invariant filters,
parameterized by a b-spline with parameters specified by a small neural network.

The file also includes the implementations of helpful utility functions like `fftk` which computes
the discrete Fast Fourier Transform of shape and `_deBoorVectorized` which computes De Boor's algorithm
for B-splines in a vectorized manner.

The script is primarily used for handling and processing image data in Fourier space.
"""
import jax.numpy as jnp
from jaxpm.kernels import fftk
import haiku as hk
import jax


def initialize_model(n_mesh: int, model_name: str, n_knots: int = 16, latent_size: int = 32):
    """
    Initiliaze the model with a random initial state

    :param n_mesh: The size of the mesh to be used for the model.
    :param model_name: The name of the model to be initialized.
    :param n_knots: The number of knots to be used in the model. Defaults to 16.
    :param latent_size: The size of the latent space for the model. Defaults to 32.
    :return: A tuple containing the initialized model and its parameters.

    This method initializes a model by using the specified parameters. It creates a Model object of a specific type specified by the modelName. The model is initialized using the n_knot
    *s and latent_size parameters. The model is then transformed using the hk.transform() function and stored in the 'model' variable.

    A PRNGSequence object is created using the hk.PRNGSequence() function with a seed value of 1. This is used to generate random numbers for the model initialization process.

    An array 'kvec' is generated using the fftk() function with the size of the mesh. This array is used to calculate the 'kk' parameter using mathematical operations.

    The model is then initialized using the init() method of the 'model' object. The parameters are generated using the next() function on the rng_seq object, 'kk', and arrays of ones.

    The initialized model and its parameters are returned as a tuple.
    """
    model = hk.without_apply_rng(
        hk.transform(
            lambda x, a, c: models[model_name](
                n_knots=n_knots, latent_size=latent_size
            )(x, a, c)
        ),
    )
    rng_seq = hk.PRNGSequence(1)

    kvec = fftk([n_mesh, n_mesh, n_mesh])
    kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
    params = model.init(
        next(rng_seq), kk, jnp.ones([1]), jnp.ones([2]) * 0.3
    )
    return model, params


def _deBoorVectorized(x, t, c, p):
    """
    :param x: Input vector of values.
    :param t: Knot vector defining the parameter space.
    :param c: Control points defining the curve.
    :param p: Degree of the B-spline curve.
    :return: The de Boor vectorized algorithm evaluates the B-spline curve at the given values `x`, using the knot vector `t`, control points `c`, and curve degree `p`.

    The algorithm calculates the B-spline curve values using the de Boor algorithm in a vectorized manner. It iteratively interpolates the control points based on the given values `x`, returning
    * the curve values at the specified parameter values.
    """
    k = jnp.digitize(x, t) - 1

    d = [c[j + k - p] for j in range(0, p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


class Model1(hk.Module):
    """A rotationally invariant filter parameterized by
    a b-spline with parameters specified by a small NN."""

    def __init__(self, n_knots=8, latent_size=16, name=None):
        """
        n_knots: number of control points for the spline
        """
        super().__init__(name=name)
        self.n_knots = n_knots
        self.latent_size = latent_size

    def __call__(self, x, a, cosmo):
        """
        x: array, scale, normalized to fftfreq default
        a: scalar, scale factor
        """
        a = jnp.concatenate([jnp.array(cosmo), a])

        net = jnp.sin(hk.Linear(self.latent_size)(jnp.atleast_1d(a)))
        net = jnp.sin(hk.Linear(self.latent_size)(net))

        net = jnp.sin(hk.Linear(self.latent_size)(net))
        net = jnp.sin(hk.Linear(self.latent_size)(net))

        w = hk.Linear(self.latent_size)(net)
        k = hk.Linear(self.latent_size)(net)

        w = hk.Linear(self.n_knots + 1)(w)
        k = hk.Linear(self.n_knots - 1)(k)

        # make sure the knots sum to 1 and are in the interval 0,1
        k = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k))])

        w = jnp.concatenate([jnp.zeros((1,)), w])

        # Augment with repeating points
        ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

        return _deBoorVectorized(jnp.clip(x / jnp.sqrt(3), 0, 1 - 1e-4), ak, w, 3)


class NeuralSplineFourierFilter(hk.Module):
    """A rotationally invariant filter parameterized by
    a b-spline with parameters specified by a small NN."""

    def __init__(self, n_knots=8, latent_size=16, name=None):
        """
        n_knots: number of control points for the spline
        """
        super().__init__(name=name)
        self.n_knots = n_knots
        self.latent_size = latent_size

    def __call__(self, x, a, cosmo):
        """
        x: array, scale, normalized to fftfreq default
        a: scalar, scale factor
        """
        a = jnp.concatenate([jnp.array(cosmo), a])

        net = jnp.sin(hk.Linear(self.latent_size)(jnp.atleast_1d(a)))
        net = jnp.sin(hk.Linear(self.latent_size)(net))

        net = jnp.sin(hk.Linear(self.latent_size)(net))
        net = jnp.sin(hk.Linear(self.latent_size)(net))

        """net = jnp.sin(hk.Linear(self.latent_size)(net))
        net = jnp.sin(hk.Linear(self.latent_size)(net))"""

        w = hk.Linear(self.latent_size)(net)
        k = hk.Linear(self.latent_size)(net)

        w = hk.Linear(self.n_knots + 1)(w)
        k = hk.Linear(self.n_knots - 1)(k)

        """w = hk.Linear(self.n_knots + 1)(net)
        k = hk.Linear(self.n_knots - 1)(net)"""

        # make sure the knots sum to 1 and are in the interval 0,1
        k = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k))])

        w = jnp.concatenate([jnp.zeros((1,)), w])

        # Augment with repeating points
        ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

        return _deBoorVectorized(jnp.clip(x / jnp.sqrt(3), 0, 1 - 1e-4), ak, w, 3)


models = {"Default": NeuralSplineFourierFilter,
          "Model": Model1
          }
