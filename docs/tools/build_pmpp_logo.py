#!/usr/bin/env python
"""Build the PM++ hybrid SVG logo from a Gadget HDF5 snapshot.

The generated SVG keeps the circle, partition dividers, rim, and PM++ wordmark
as vector artwork.  A full-box particle-count projection is stored once as an
embedded grayscale PNG and recolored into four bands by SVG filters.

Examples
--------
CAMELS (single-file snapshot, the default)::

    python docs/tools/build_pmpp_logo.py

Quijote (sharded snapshot)::

    python docs/tools/build_pmpp_logo.py \
        --snapshot 'Quijote/0/snapdir_004/snap_004.*.hdf5'
"""

from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = REPO_ROOT / "CAMELS" / "snapshot_090.hdf5"
DEFAULT_SVG = REPO_ROOT / "docs" / "source" / "_static" / "pmpp-logo.svg"
DEFAULT_PNG = REPO_ROOT / "docs" / "source" / "_static" / "pmpp-logo.png"

VIEWBOX_SIZE = 1024
CIRCLE_CENTER = 512
CIRCLE_RADIUS = 488
PARTITION_EDGES = (0, 256, 512, 768, 1024)
PARTITION_COLORS = ("#0090ff", "#00dfa0", "#ffe000", "#ff7a00")
DIVIDER_COLORS = ("#00c7ca", "#a7df00", "#ffb500")
WORDMARK_CENTER_Y = 520.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        nargs="+",
        default=[str(DEFAULT_SNAPSHOT)],
        help=(
            "Snapshot HDF5 file(s), directories, or quoted glob patterns. "
            "All matching shards are streamed and combined."
        ),
    )
    parser.add_argument(
        "--projection-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Axis integrated out of the full-box projection (default: z).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=2048,
        help="Pixel resolution of the embedded density projection (default: 2048).",
    )
    parser.add_argument(
        "--svg-out",
        type=Path,
        default=DEFAULT_SVG,
        help="Self-contained SVG output path.",
    )
    parser.add_argument(
        "--png-out",
        type=Path,
        default=DEFAULT_PNG,
        help="PNG fallback output path, rasterized from the completed SVG.",
    )
    parser.add_argument(
        "--png-size",
        type=int,
        default=1280,
        help="Width and height of the PNG fallback (default: 1280).",
    )
    return parser.parse_args()


def resolve_snapshot_paths(specs: list[str]) -> list[Path]:
    """Expand files, directories, and shell-independent glob patterns."""
    resolved: list[Path] = []
    for raw_spec in specs:
        expanded = os.path.expandvars(os.path.expanduser(raw_spec))
        candidate = Path(expanded)
        if candidate.is_dir():
            matches = sorted(candidate.glob("*.hdf5"))
        elif candidate.is_file():
            matches = [candidate]
        else:
            matches = [Path(path) for path in sorted(glob.glob(expanded))]
        if not matches:
            raise FileNotFoundError(f"No HDF5 snapshots matched {raw_spec!r}.")
        resolved.extend(path.resolve() for path in matches)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Snapshot path is not HDF5: {path}")
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def projection_axes(axis: str) -> tuple[int, int]:
    """Return horizontal and vertical coordinate columns for a projection."""
    return {
        "x": (1, 2),  # y-z plane
        "y": (0, 2),  # x-z plane
        "z": (0, 1),  # x-y plane
    }[axis]


def load_projection(
    snapshot_paths: list[Path],
    projection_axis: str,
    bins: int,
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Stream particle coordinates and construct a full-box 2D histogram."""
    if bins < 64:
        raise ValueError("--bins must be at least 64.")

    horizontal_axis, vertical_axis = projection_axes(projection_axis)
    counts = np.zeros((bins, bins), dtype=np.uint32)
    total_particles = 0
    expected_particles: int | None = None
    box_size: float | None = None
    redshift: float | None = None

    for snapshot_path in snapshot_paths:
        with h5py.File(snapshot_path, "r") as handle:
            if "PartType1/Coordinates" not in handle:
                raise KeyError(f"Missing PartType1/Coordinates in {snapshot_path}")
            if "Header" not in handle:
                raise KeyError(f"Missing Header in {snapshot_path}")

            header = handle["Header"].attrs
            shard_box_size = float(header["BoxSize"])
            shard_redshift = float(header.get("Redshift", np.nan))
            shard_expected = int(np.asarray(header["NumPart_Total"])[1])

            if box_size is None:
                box_size = shard_box_size
                redshift = shard_redshift
                expected_particles = shard_expected
            elif not np.isclose(shard_box_size, box_size, rtol=0, atol=1e-6):
                raise ValueError(
                    f"Inconsistent BoxSize: {snapshot_path} has {shard_box_size}, "
                    f"expected {box_size}."
                )

            coordinates = handle["PartType1/Coordinates"]
            for start in range(0, coordinates.shape[0], chunk_size):
                chunk = np.asarray(
                    coordinates[start : start + chunk_size, [horizontal_axis, vertical_axis]],
                    dtype=np.float32,
                )
                shard_counts, _, _ = np.histogram2d(
                    chunk[:, 1],
                    chunk[:, 0],
                    bins=bins,
                    range=((0.0, box_size), (0.0, box_size)),
                )
                counts += shard_counts.astype(np.uint32)
                total_particles += int(chunk.shape[0])

    histogram_particles = int(counts.sum(dtype=np.uint64))
    if histogram_particles != total_particles:
        raise RuntimeError(
            "Projection lost particles outside the declared full-box bounds: "
            f"loaded={total_particles}, histogram={histogram_particles}."
        )
    if expected_particles is not None and total_particles != expected_particles:
        raise RuntimeError(
            "Snapshot shard set is incomplete or duplicated: "
            f"header expects {expected_particles}, loaded {total_particles}."
        )

    def display_path(path: Path) -> str:
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(path)

    metadata: dict[str, object] = {
        "snapshot_files": [display_path(path) for path in snapshot_paths],
        "projection_axis": projection_axis,
        "projection_plane": "".join("xyz"[index] for index in (horizontal_axis, vertical_axis)),
        "bins": bins,
        "box_size": box_size,
        "redshift": redshift,
        "particles": total_particles,
    }
    return counts, metadata


def normalize_projection(counts: np.ndarray) -> tuple[Image.Image, dict[str, float]]:
    """Apply a deterministic log stretch and a small screen-style glow."""
    log_counts = np.log1p(counts.astype(np.float32))
    positive = log_counts[counts > 0]
    if positive.size == 0:
        raise ValueError("The projected particle histogram is empty.")

    low, high = np.percentile(positive, (1.0, 99.9))
    if not high > low:
        raise ValueError("The projected particle histogram has no usable contrast.")

    normalized = np.clip((log_counts - low) / (high - low), 0.0, 1.0)
    normalized = np.power(normalized, 1.35, dtype=np.float32)
    base = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L")

    glow_radius = 1.5 * counts.shape[0] / 2048.0
    blurred = np.asarray(base.filter(ImageFilter.GaussianBlur(radius=glow_radius)), dtype=np.float32) / 255.0
    base_float = np.asarray(base, dtype=np.float32) / 255.0
    mixed = 1.0 - (1.0 - base_float) * (1.0 - 0.42 * blurred)
    alpha = np.rint(np.clip(mixed, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.full((*alpha.shape, 4), 255, dtype=np.uint8)
    rgba[..., 3] = alpha
    image = Image.fromarray(rgba, mode="RGBA")

    return image, {
        "normalization": "log1p",
        "nonzero_percentile_low": float(low),
        "nonzero_percentile_high": float(high),
        "gamma": 1.35,
        "glow_mix": 0.42,
        "glow_radius_pixels": float(glow_radius),
    }


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=9)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def svg_number(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def text_path_data(text: str) -> tuple[str, str]:
    """Convert a bold DejaVu Sans wordmark into portable SVG path data."""
    font = FontProperties(family="DejaVu Sans", weight="bold")
    font_path = findfont(font, fallback_to_default=False)
    path = TextPath((0.0, 0.0), text, size=1.0, prop=font, usetex=False)
    bounds = path.get_extents()

    target_width = 800.0
    scale = target_width / bounds.width
    x_offset = (VIEWBOX_SIZE - target_width) / 2.0 - bounds.xmin * scale
    source_center_y = (bounds.ymin + bounds.ymax) / 2.0

    def transform_pairs(values: np.ndarray) -> list[float]:
        transformed: list[float] = []
        for index in range(0, len(values), 2):
            x_value = x_offset + float(values[index]) * scale
            y_value = WORDMARK_CENTER_Y - (float(values[index + 1]) - source_center_y) * scale
            transformed.extend((x_value, y_value))
        return transformed

    commands: list[str] = []
    for vertices, code in path.iter_segments(curves=True, simplify=False):
        if code == MplPath.MOVETO:
            x_value, y_value = transform_pairs(vertices)
            commands.append(f"M {svg_number(x_value)} {svg_number(y_value)}")
        elif code == MplPath.LINETO:
            x_value, y_value = transform_pairs(vertices)
            commands.append(f"L {svg_number(x_value)} {svg_number(y_value)}")
        elif code == MplPath.CURVE3:
            control_x, control_y, end_x, end_y = transform_pairs(vertices)
            commands.append(
                f"Q {svg_number(control_x)} {svg_number(control_y)} "
                f"{svg_number(end_x)} {svg_number(end_y)}"
            )
        elif code == MplPath.CURVE4:
            values = transform_pairs(vertices)
            commands.append("C " + " ".join(svg_number(value) for value in values))
        elif code == MplPath.CLOSEPOLY:
            commands.append("Z")
        else:  # pragma: no cover - TextPath currently uses only these codes.
            raise ValueError(f"Unsupported matplotlib path code: {code}")

    return " ".join(commands), font_path


def color_matrix(hex_color: str) -> str:
    red = int(hex_color[1:3], 16) / 255.0
    green = int(hex_color[3:5], 16) / 255.0
    blue = int(hex_color[5:7], 16) / 255.0
    return (
        f"{red:.6f} 0 0 0 0 "
        f"{green:.6f} 0 0 0 0 "
        f"{blue:.6f} 0 0 0 0 "
        "0 0 0 1 0"
    )


def source_label(snapshot_paths: list[Path], redshift: object) -> str:
    joined = " ".join(str(path).lower() for path in snapshot_paths)
    if redshift is None or not np.isfinite(float(redshift)):
        redshift_text = "unknown"
    else:
        redshift_value = float(redshift)
        redshift_text = "0" if abs(redshift_value) < 5e-7 else f"{redshift_value:.6g}"
    if "camels" in joined:
        return f"CAMELS dark-matter snapshot (z={redshift_text})"
    if "quijote" in joined:
        return f"Quijote dark-matter snapshot (z={redshift_text})"
    return f"Gadget dark-matter snapshot (z={redshift_text})"


def build_svg(image: Image.Image, metadata: dict[str, object]) -> tuple[str, dict[str, object]]:
    embedded_png = encode_png(image)
    wordmark_path, font_path = text_path_data("PM++")
    label = source_label(
        [Path(path) for path in metadata["snapshot_files"]],
        metadata.get("redshift"),
    )

    complete_metadata = {
        "asset": "PM++ hybrid SVG logo",
        "source": label,
        **metadata,
        "normalization": metadata["normalization"],
        "partition_colors": list(PARTITION_COLORS),
        "partition_edges": list(PARTITION_EDGES),
        "wordmark": "PM++",
        "wordmark_center_y": WORDMARK_CENTER_Y,
        "wordmark_font_source": Path(font_path).name,
        "transparent_background": True,
    }
    metadata_text = html.escape(json.dumps(complete_metadata, indent=2, sort_keys=True))

    clip_paths = []
    filters = []
    image_uses = []
    for index, color in enumerate(PARTITION_COLORS):
        x_start = PARTITION_EDGES[index]
        width = PARTITION_EDGES[index + 1] - x_start
        clip_paths.append(
            f'<clipPath id="partition-{index}"><rect x="{x_start}" y="0" '
            f'width="{width}" height="{VIEWBOX_SIZE}"/></clipPath>'
        )
        filters.append(
            f'<filter id="tint-{index}" x="0" y="0" width="100%" height="100%" '
            'color-interpolation-filters="sRGB">'
            f'<feColorMatrix type="matrix" values="{color_matrix(color)}"/>'
            "</filter>"
        )
        image_uses.append(
            f'<use href="#density-field" clip-path="url(#partition-{index})" '
            f'filter="url(#tint-{index})"/>'
        )

    divider_lines = []
    for x_value, color in zip(PARTITION_EDGES[1:-1], DIVIDER_COLORS):
        divider_lines.append(
            f'<line x1="{x_value}" y1="0" x2="{x_value}" y2="{VIEWBOX_SIZE}" '
            f'stroke="{color}" stroke-width="3" opacity="0.82"/>'
        )

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{VIEWBOX_SIZE}" height="{VIEWBOX_SIZE}" viewBox="0 0 {VIEWBOX_SIZE} {VIEWBOX_SIZE}" role="img" aria-labelledby="pmpp-logo-title pmpp-logo-description">
  <title id="pmpp-logo-title">PM++ multi-GPU cosmology logo</title>
  <desc id="pmpp-logo-description">The PM++ wordmark over a real full-box cosmological density projection divided into four blue, green, yellow, and orange compute partitions.</desc>
  <metadata>{metadata_text}</metadata>
  <defs>
    <clipPath id="disc-clip"><circle cx="{CIRCLE_CENTER}" cy="{CIRCLE_CENTER}" r="{CIRCLE_RADIUS}"/></clipPath>
    {''.join(clip_paths)}
    {''.join(filters)}
    <linearGradient id="brand-gradient" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{PARTITION_COLORS[0]}"/>
      <stop offset="0.35" stop-color="{PARTITION_COLORS[1]}"/>
      <stop offset="0.65" stop-color="{PARTITION_COLORS[2]}"/>
      <stop offset="1" stop-color="{PARTITION_COLORS[3]}"/>
    </linearGradient>
    <image id="density-field" x="0" y="0" width="{VIEWBOX_SIZE}" height="{VIEWBOX_SIZE}" preserveAspectRatio="none" href="data:image/png;base64,{embedded_png}"/>
  </defs>
  <g clip-path="url(#disc-clip)">
    {''.join(image_uses)}
    {''.join(divider_lines)}
  </g>
  <circle cx="{CIRCLE_CENTER}" cy="{CIRCLE_CENTER}" r="{CIRCLE_RADIUS}" fill="none" stroke="url(#brand-gradient)" stroke-width="10"/>
  <path d="{wordmark_path}" fill="url(#brand-gradient)" stroke="#fff" stroke-opacity="0.22" stroke-width="5" stroke-linejoin="round" paint-order="stroke fill"/>
  <path d="{wordmark_path}" fill="none" stroke="#fff" stroke-opacity="0.16" stroke-width="1.5"/>
</svg>
'''
    return svg, complete_metadata


def find_browser() -> Path:
    command_names = ("google-chrome", "chrome", "chromium", "chromium-browser", "msedge")
    for command_name in command_names:
        command = shutil.which(command_name)
        if command:
            return Path(command)

    candidates = (
        Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find Chrome, Chromium, or Edge for SVG rasterization.")


def windows_path(path: Path) -> str:
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def browser_path_and_url(path: Path, browser: Path) -> tuple[str, str]:
    is_windows_browser = browser.suffix.lower() == ".exe" and platform.system() != "Windows"
    if is_windows_browser:
        converted = windows_path(path.resolve())
        file_url = "file:///" + quote(converted.replace("\\", "/"), safe="/:~")
        return converted, file_url
    return str(path.resolve()), path.resolve().as_uri()


def rasterize_svg(svg_path: Path, png_path: Path, size: int) -> Path:
    if size < 16:
        raise ValueError("--png-size must be at least 16.")
    browser = find_browser()
    # Chromium enforces a minimum viewport width on some platforms.  Render
    # small icons at 512 px and downsample so the SVG is never edge-cropped.
    render_size = max(size, 512)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    output_argument, _ = browser_path_and_url(png_path, browser)
    _, svg_url = browser_path_and_url(svg_path, browser)

    with tempfile.TemporaryDirectory(prefix="pmpp-logo-browser-", dir=png_path.parent) as profile_dir:
        wrapper_path = Path(profile_dir) / "render.html"
        wrapper_path.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:transparent}"
            "img{display:block;width:100vw;height:100vh}</style></head>"
            f"<body><img src=\"{html.escape(svg_url, quote=True)}\"></body></html>",
            encoding="utf-8",
        )
        profile_argument, _ = browser_path_and_url(Path(profile_dir), browser)
        _, wrapper_url = browser_path_and_url(wrapper_path, browser)
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--default-background-color=00000000",
            f"--window-size={render_size},{render_size}",
            f"--user-data-dir={profile_argument}",
            f"--screenshot={output_argument}",
            wrapper_url,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode != 0:
            raise RuntimeError(
                "Browser SVG rasterization failed.\n"
                f"command: {command}\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
            )

    if not png_path.exists():
        raise RuntimeError(f"Browser did not create the expected PNG: {png_path}")
    if render_size != size:
        with Image.open(png_path) as rendered:
            resized = rendered.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            resized.save(png_path, format="PNG", optimize=True, compress_level=9)
    with Image.open(png_path) as image:
        if image.size != (size, size):
            raise RuntimeError(f"Expected a {size}x{size} PNG, got {image.size}.")
    return browser


def main() -> None:
    args = parse_args()
    snapshot_paths = resolve_snapshot_paths(args.snapshot)
    counts, projection_metadata = load_projection(
        snapshot_paths,
        projection_axis=args.projection_axis,
        bins=args.bins,
    )
    projection_image, normalization_metadata = normalize_projection(counts)
    projection_metadata.update(normalization_metadata)
    svg, complete_metadata = build_svg(projection_image, projection_metadata)

    args.svg_out.parent.mkdir(parents=True, exist_ok=True)
    args.svg_out.write_text(svg, encoding="utf-8", newline="\n")
    browser = rasterize_svg(args.svg_out, args.png_out, args.png_size)

    summary = {
        "svg": str(args.svg_out.resolve()),
        "svg_bytes": args.svg_out.stat().st_size,
        "png": str(args.png_out.resolve()),
        "png_bytes": args.png_out.stat().st_size,
        "png_size": args.png_size,
        "browser": str(browser),
        "projection": complete_metadata,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
