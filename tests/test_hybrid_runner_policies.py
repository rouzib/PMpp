"""Command-line routing policy contract for the standalone forward runner."""

import sys

import pytest

from scripts.run_hybrid_256_box5 import arguments


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], ("hybrid", "neighbor_only")),
        (["--policy", "hybrid"], ("hybrid", "hybrid")),
        (["--policy", "neighbor_only"], ("neighbor_only", "neighbor_only")),
        (["--policy", "hybrid", "--nbody-policy", "neighbor_only"],
         ("hybrid", "neighbor_only")),
        (["--lpt-policy", "neighbor_only", "--nbody-policy", "hybrid"],
         ("neighbor_only", "hybrid")),
    ],
)
def test_phase_policies_resolve_independently(monkeypatch, flags, expected):
    monkeypatch.setattr(sys, "argv", ["runner", "--output", "result.json", *flags])
    args = arguments()
    assert (args.lpt_policy, args.nbody_policy) == expected


def test_hybrid_lpt_rejects_cpu_even_with_strict_nbody(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "runner", "--output", "result.json", "--platform", "cpu",
        "--lpt-policy", "hybrid", "--nbody-policy", "neighbor_only",
    ])
    with pytest.raises(SystemExit, match="2"):
        arguments()
