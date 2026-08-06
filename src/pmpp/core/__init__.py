"""Shared configuration and low-level PM++ utilities."""

from .._api import install_lazy_api

_EXPORTS = {
    "AXIS_NAME": (".utils", "AXIS_NAME"),
    "Configuration": (".configuration", "Configuration"),
    "create_compute_mesh": (".utils", "create_compute_mesh"),
    "pmid_to_idx": (".utils", "pmid_to_idx"),
    "raise_error": (".utils", "raise_error"),
}

install_lazy_api(__name__, _EXPORTS)
