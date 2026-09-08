"""Summarize matched benchmark JSON and draw runtime/buffer comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    assert baseline["size"] == candidate["size"]
    assert baseline["devices"] == candidate["devices"]
    rows = {}
    for name, old in baseline["phases"].items():
        new = candidate["phases"][name]
        if "median_seconds" not in old:
            rows[name] = {"baseline": old, "candidate": new}
            continue
        rows[name] = {
            "baseline_ms": old["median_seconds"] * 1000,
            "candidate_ms": new["median_seconds"] * 1000,
            "runtime_reduction_percent": (1 - new["median_seconds"] / old["median_seconds"]) * 100,
            "baseline_temporary_bytes": old["temporary_bytes"],
            "candidate_temporary_bytes": new["temporary_bytes"],
            "temporary_reduction_percent": (1 - new["temporary_bytes"] / old["temporary_bytes"]) * 100,
            "relative_norm_difference": abs(new["l2"] / old["l2"] - 1),
            "baseline_samples": len(old["seconds"]),
            "candidate_samples": len(new["seconds"]),
            "baseline_p10_p90_ms": (np.percentile(old["seconds"], [10, 90]) * 1000).tolist(),
            "candidate_p10_p90_ms": (np.percentile(new["seconds"], [10, 90]) * 1000).tolist(),
        }
    output = {
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "baseline_file": args.baseline.name,
        "candidate_file": args.candidate.name,
        "size": baseline["size"],
        "device_count": len(baseline["devices"]),
        "jax": baseline["jax"],
        "phases": rows,
        "candidate_validation": candidate.get("final_validation")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "| Phase | Baseline (ms) | Candidate (ms) | Time reduction | Temporary-buffer reduction |",
        "| --- | ---: | ---: | ---: | ---: |"
    ]
    for name, row in rows.items():
        if "baseline_ms" in row:
            lines.append(
                f'| {name} | {row["baseline_ms"]:.2f} | {row["candidate_ms"]:.2f} | '
                f'{row["runtime_reduction_percent"]:.1f}% | {row["temporary_reduction_percent"]:.1f}% |'
            )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {
        "fft_pullback": "Real FFT pullback",
        "force": "Force",
        "step": "Initialized step",
        "nbody": "Four-step N-body",
        "nbody_grad": "N-body gradient",
        "streaming_2lpt": "Streamed 2LPT"
    }
    names = list(labels)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, title in zip(
        axes, ("runtime_reduction_percent", "temporary_reduction_percent"),
        ("Runtime reduction", "Compiler temporary-buffer reduction")
    ):
        values = [rows[name][key] for name in names]
        ax.barh(list(labels.values()), values, color=["#217c80" if v >= 0 else "#c76248" for v in values])
        ax.invert_yaxis()
        ax.axvline(0, color="#666", lw=.8)
        ax.set_xlabel("Percent; positive means lower")
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.annotate(
                f"{v:.1f}%", (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points", va="center",
                ha="left" if v >= 0 else "right", fontsize=9
            )
        ax.margins(x=.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PM++ · 256³ · 2 × RTX 3090 · JAX 0.9.1", fontsize=14)
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    fig.savefig(args.output_dir / "comparison.pdf")
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
