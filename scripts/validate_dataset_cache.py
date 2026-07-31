#!/usr/bin/env python3
"""Verify that an extracted frame cache still matches its source capture."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source capture not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _compare_source(
    expected: dict[str, Any],
    actual: dict[str, Any],
    label: str,
) -> None:
    mismatches = [
        key
        for key in ("path", "size_bytes", "mtime_ns")
        if expected.get(key) != actual[key]
    ]
    if mismatches:
        raise ValueError(
            f"{label} changed since frame extraction "
            f"({', '.join(mismatches)} mismatch)"
        )


def _validate_numbered_images(image_dir: Path, expected_count: int) -> None:
    if expected_count < 2:
        raise ValueError("cached frame metadata must describe at least two frames")
    if not image_dir.is_dir():
        raise ValueError(f"cached image directory is missing: {image_dir}")
    actual_names = sorted(
        image.name
        for image in image_dir.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
    )
    expected_names = [f"{index:06d}.jpg" for index in range(expected_count)]
    if actual_names != expected_names:
        raise ValueError(
            f"cached frame set is incomplete or unexpected in {image_dir}: "
            f"expected {expected_count}, found {len(actual_names)}"
        )


def validate_cache(
    dataset: Path,
    source: Path,
    sample_fps: float,
    max_frames: int,
) -> None:
    dataset = dataset.expanduser().resolve()
    metadata_path = dataset / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not math.isclose(
        float(metadata.get("sample_fps", -1)),
        sample_fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("cached sample_fps does not match this run")
    if metadata.get("max_frames") != max_frames:
        raise ValueError("cached max_frames does not match this run")

    actual_source = _source_identity(source)
    dataset_type = metadata.get("dataset_type")
    if dataset_type == "single_video":
        expected = {
            "path": metadata.get("source_video"),
            "size_bytes": metadata.get("source_size_bytes"),
            "mtime_ns": metadata.get("source_mtime_ns"),
        }
        _compare_source(expected, actual_source, "source capture")
        selected_indices = metadata.get("selected_indices")
        if not isinstance(selected_indices, list):
            raise ValueError("single-video metadata has no selected frame list")
        _validate_numbered_images(
            dataset / "images",
            len(selected_indices),
        )
    else:
        raise ValueError(
            "frame cache predates source-identity tracking; recreate it with "
            "--retry-incomplete"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--sample-fps", required=True, type=float)
    parser.add_argument("--max-frames", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_cache(
            args.dataset,
            args.source,
            args.sample_fps,
            args.max_frames,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
