#!/usr/bin/env python3
"""Plot three mean-density projections from a saved PM++ density cube."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_hybrid_256_box5 import plot_density_projections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("density", type=Path, help="Saved *_density.npy file")
    parser.add_argument("--box-size", type=float, default=5.0, help="Box length in Mpc/h")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.box_size <= 0:
        parser.error("box-size must be positive")

    density = np.load(args.density, mmap_mode="r", allow_pickle=False)
    if density.ndim != 3 or len(set(density.shape)) != 1:
        raise ValueError(f"expected a cubic 3D density, got {density.shape}")
    projections = {
        axis: density.mean(axis=index, dtype=np.float64).astype(np.float32)
        for index, axis in enumerate("xyz")
    }
    if any(not np.isfinite(value).all() for value in projections.values()):
        raise ValueError("density projections contain nonfinite values")
    stem = args.density.stem.removesuffix("_density")
    output = args.output_dir / f"{stem}_projections.png"
    plot_density_projections(projections, args.box_size, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
