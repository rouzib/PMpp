"""Cloud-in-cell scatter and gather operators."""

from .._api import install_lazy_api

_EXPORTS = {
    "_gather": (".gather", "_gather"),
    "_scatter": (".scatter", "_scatter"),
    "gather": (".gather", "gather"),
    "pallas_cic_supported": (".pallas", "pallas_cic_supported"),
    "scatter": (".scatter", "scatter"),
}

install_lazy_api(__name__, _EXPORTS)
