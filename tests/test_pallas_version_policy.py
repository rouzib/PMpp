"""Version-policy checks for the optional Pallas CIC implementation."""

import pytest

from pmpp.cic import pallas as pallas_module


@pytest.mark.parametrize(("version", "expected"), [("0.5.3", False), ("0.6.0", True), ("0.9.1", True),
                                                   ("0.10.2+computecanada", True), ("1.0.0", True),
                                                   ("development", False), ],
                         )
def test_pallas_cic_jax_version_policy(monkeypatch, version, expected):
    monkeypatch.setattr(pallas_module.jax, "__version__", version)
    assert pallas_module._supported_pallas_jax_version() is expected
