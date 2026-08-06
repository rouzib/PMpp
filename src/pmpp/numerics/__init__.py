"""Reusable numerical primitives used by the PM++ solver."""

from .._api import install_lazy_api

_EXPORTS = {
    "fftfreq": (".fft", "fftfreq"),
    "fftfwd": (".fft", "fftfwd"),
    "fftinv": (".fft", "fftinv"),
    "odeint": (".ode", "odeint"),
}

install_lazy_api(__name__, _EXPORTS)
