#!/usr/bin/env python3
"""Verify that an extracted frame cache still matches its source capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


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
    if expected.get("size_bytes") != actual["size_bytes"]:
        raise ValueError(
            f"{label} changed since frame extraction (size_bytes mismatch)"
        )

    expected_digest = expected.get("sha256")
    if isinstance(expected_digest, str):
        # The common same-path/same-mtime case avoids re-reading a large video.
        # A move, restore, or touch falls back to content verification instead
        # of invalidating an otherwise identical cache.
        if (
            expected.get("path") == actual["path"]
            and expected.get("mtime_ns") == actual["mtime_ns"]
        ):
            return
        if _sha256(Path(actual["path"])) != expected_digest:
            raise ValueError(
                f"{label} changed since frame extraction (sha256 mismatch)"
            )
        return

    # Backward-compatible handling for caches created before content digests
    # were recorded. These retain the previous strict path/mtime behavior.
    mismatches = [
        key
        for key in ("path", "mtime_ns")
        if expected.get(key) != actual[key]
    ]
    if mismatches:
        raise ValueError(
            f"{label} changed since frame extraction "
            f"({', '.join(mismatches)} mismatch)"
        )


def _validate_image_names(image_dir: Path, expected_names: Sequence[str]) -> None:
    if not image_dir.is_dir():
        raise ValueError(f"cached image directory is missing: {image_dir}")
    actual_names = sorted(
        image.relative_to(image_dir).as_posix()
        for image in image_dir.rglob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
    )
    if actual_names != sorted(expected_names):
        raise ValueError(
            f"cached frame set is incomplete or unexpected in {image_dir}: "
            f"expected {len(expected_names)}, found {len(actual_names)}"
        )


def _validate_numbered_images(image_dir: Path, expected_count: int) -> None:
    if expected_count < 2:
        raise ValueError("cached frame metadata must describe at least two frames")
    expected_names = [f"{index:06d}.jpg" for index in range(expected_count)]
    _validate_image_names(image_dir, expected_names)


def _normalize_sources(source: Path | Sequence[Path]) -> list[Path]:
    sources = [source] if isinstance(source, Path) else list(source)
    if not sources:
        raise ValueError("at least one source capture is required")
    return [item.expanduser().resolve() for item in sources]


def _validate_video_set(
    dataset: Path,
    metadata: dict[str, Any],
    actual_sources: Sequence[Path],
) -> int:
    source_records = metadata.get("sources")
    if not isinstance(source_records, list) or len(source_records) < 2:
        raise ValueError("video-set metadata must contain at least two sources")
    if len(actual_sources) != len(source_records):
        raise ValueError(
            "source capture count does not match the ordered video set: "
            f"expected {len(source_records)}, found {len(actual_sources)}"
        )

    expected_names = []
    expected_dimensions: tuple[int, int] | None = None
    total_selected_frames = 0
    for index, (record, source_path) in enumerate(
        zip(source_records, actual_sources, strict=True)
    ):
        if not isinstance(record, dict):
            raise ValueError(f"video-set source record {index} is invalid")
        if (
            not isinstance(record.get("source_video"), str)
            or not isinstance(record.get("source_size_bytes"), int)
            or not isinstance(record.get("source_mtime_ns"), int)
            or not isinstance(record.get("source_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"]) is None
        ):
            raise ValueError(f"video-set source {index} has invalid identity")
        expected_source = {
            "path": record.get("source_video"),
            "size_bytes": record.get("source_size_bytes"),
            "mtime_ns": record.get("source_mtime_ns"),
            "sha256": record.get("source_sha256"),
        }
        _compare_source(
            expected_source,
            _source_identity(source_path),
            f"source capture {index + 1}",
        )

        image_prefix = record.get("image_prefix")
        if (
            not isinstance(image_prefix, str)
            or not image_prefix.startswith(f"{index:02d}_")
            or re.fullmatch(r"[0-9]+_[a-z0-9_]+_", image_prefix) is None
        ):
            raise ValueError(f"video-set source {index} has an invalid image prefix")
        selected_indices = record.get("selected_indices")
        if (
            not isinstance(selected_indices, list)
            or len(selected_indices) < 2
            or not all(isinstance(item, int) for item in selected_indices)
        ):
            raise ValueError(
                f"video-set source {index} has no valid selected frame list"
            )
        expected_names.extend(
            f"{image_prefix}{frame_index:06d}.jpg"
            for frame_index in range(len(selected_indices))
        )
        total_selected_frames += len(selected_indices)

        dimensions = (
            record.get("output_frame_width"),
            record.get("output_frame_height"),
        )
        if (
            not all(isinstance(value, int) and value > 0 for value in dimensions)
            or (expected_dimensions is not None and dimensions != expected_dimensions)
        ):
            raise ValueError("video-set sources do not share output dimensions")
        expected_dimensions = dimensions

    if metadata.get("total_selected_frames") != total_selected_frames:
        raise ValueError("video-set total frame count does not match source records")
    if expected_dimensions != (
        metadata.get("output_frame_width"),
        metadata.get("output_frame_height"),
    ):
        raise ValueError("video-set output dimensions do not match source records")
    _validate_image_names(dataset / "images", expected_names)
    return total_selected_frames


def _validate_source_registration(
    metadata: dict[str, Any],
    stats: dict[str, Any],
    minimum_ratio: float,
) -> None:
    if metadata.get("dataset_type") != "video_set":
        raise ValueError(
            "per-source registration validation requires a video-set cache"
        )
    source_records = metadata.get("sources")
    diagnostics = stats.get("source_registration")
    if (
        not isinstance(source_records, list)
        or not isinstance(diagnostics, list)
        or len(diagnostics) != len(source_records)
    ):
        raise ValueError("cached COLMAP per-source diagnostics are incomplete")

    for index, (record, diagnostic) in enumerate(
        zip(source_records, diagnostics, strict=True)
    ):
        if not isinstance(record, dict) or not isinstance(diagnostic, dict):
            raise ValueError("cached COLMAP per-source diagnostics are invalid")
        selected_indices = record.get("selected_indices")
        if not isinstance(selected_indices, list) or not selected_indices:
            raise ValueError("video-set source metadata is incomplete")
        expected_input_count = len(selected_indices)
        registered_count = diagnostic.get("num_registered_images")
        if (
            diagnostic.get("source_index") != index
            or diagnostic.get("image_prefix") != record.get("image_prefix")
            or diagnostic.get("num_input_images") != expected_input_count
            or not isinstance(registered_count, int)
            or isinstance(registered_count, bool)
            or not 0 <= registered_count <= expected_input_count
        ):
            raise ValueError("cached COLMAP per-source diagnostics are invalid")
        expected_ratio = registered_count / expected_input_count
        if not math.isclose(
            float(diagnostic.get("registration_ratio", -1)),
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(diagnostic.get("minimum_registration_ratio", -1)),
            minimum_ratio,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("cached COLMAP per-source diagnostics are invalid")
        expected_status = "passed" if expected_ratio >= minimum_ratio else "failed"
        if diagnostic.get("status") != expected_status:
            raise ValueError("cached COLMAP per-source diagnostics are invalid")

    if stats.get("status") == "passed" and any(
        diagnostic.get("status") != "passed" for diagnostic in diagnostics
    ):
        raise ValueError("cached COLMAP per-source quality gate is inconsistent")


def validate_cache(
    dataset: Path,
    source: Path | Sequence[Path],
    sample_fps: float,
    max_frames: int,
    camera_model: str | None = None,
    overlap: int | None = None,
    quadratic_overlap: bool | None = None,
    min_registration_ratio: float | None = None,
    require_colmap: bool = False,
    matching_strategy: str | None = None,
    min_source_registration_ratio: float | None = None,
) -> None:
    dataset = dataset.expanduser().resolve()
    metadata_path = dataset / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        uuid.UUID(str(metadata.get("dataset_instance_id")))
    except (ValueError, AttributeError):
        raise ValueError(
            "frame cache predates dataset-instance tracking; recreate it with "
            "--retry-incomplete"
        ) from None
    if not math.isclose(
        float(metadata.get("sample_fps", -1)),
        sample_fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("cached sample_fps does not match this run")
    if metadata.get("max_frames") != max_frames:
        raise ValueError("cached max_frames does not match this run")

    actual_sources = _normalize_sources(source)
    dataset_type = metadata.get("dataset_type")
    if dataset_type == "single_video":
        if len(actual_sources) != 1:
            raise ValueError(
                "single-video cache requires exactly one source capture"
            )
        expected = {
            "path": metadata.get("source_video"),
            "size_bytes": metadata.get("source_size_bytes"),
            "mtime_ns": metadata.get("source_mtime_ns"),
            "sha256": metadata.get("source_sha256"),
        }
        _compare_source(
            expected,
            _source_identity(actual_sources[0]),
            "source capture",
        )
        selected_indices = metadata.get("selected_indices")
        if not isinstance(selected_indices, list):
            raise ValueError("single-video metadata has no selected frame list")
        _validate_numbered_images(
            dataset / "images",
            len(selected_indices),
        )
        expected_input_images = len(selected_indices)
    elif dataset_type == "video_set":
        expected_input_images = _validate_video_set(
            dataset,
            metadata,
            actual_sources,
        )
    else:
        raise ValueError(
            "frame cache predates source-identity tracking; recreate it with "
            "--retry-incomplete"
        )

    expected_images_digest = metadata.get("images_sha256")
    if not isinstance(expected_images_digest, str):
        raise ValueError(
            "frame cache predates image-integrity tracking; recreate it "
            "with --retry-incomplete"
        )
    if _tree_sha256(dataset / "images") != expected_images_digest:
        raise ValueError("cached extracted-image content has changed")

    stats_path = dataset / "reconstruction_stats.json"
    if not stats_path.exists():
        if require_colmap:
            raise ValueError("COLMAP reconstruction stats are missing")
        return

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if require_colmap and stats.get("status") != "passed":
        raise ValueError("COLMAP reconstruction has not passed its quality gate")
    if camera_model is not None and stats.get("camera_model") != camera_model:
        raise ValueError("cached COLMAP camera_model does not match this run")
    if matching_strategy is not None:
        if matching_strategy not in {"sequential", "exhaustive"}:
            raise ValueError(
                "matching_strategy must be 'sequential' or 'exhaustive'"
            )
        recorded_matching_strategy = stats.get(
            "matching_strategy",
            "sequential",
        )
        if recorded_matching_strategy != matching_strategy:
            raise ValueError(
                "cached COLMAP matching_strategy does not match this run"
            )
    if overlap is not None and stats.get("matching_overlap") != overlap:
        raise ValueError("cached COLMAP overlap does not match this run")
    if (
        quadratic_overlap is not None
        and stats.get("matching_quadratic_overlap") is not quadratic_overlap
    ):
        raise ValueError(
            "cached COLMAP quadratic-overlap setting does not match this run"
        )
    if min_registration_ratio is not None and not math.isclose(
        float(stats.get("minimum_registration_ratio", -1)),
        min_registration_ratio,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "cached COLMAP registration quality gate does not match this run"
        )
    if min_source_registration_ratio is not None:
        if not 0 < min_source_registration_ratio <= 1:
            raise ValueError(
                "min_source_registration_ratio must be greater than 0 and at "
                "most 1"
            )
        if not math.isclose(
            float(stats.get("minimum_source_registration_ratio", -1)),
            min_source_registration_ratio,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "cached COLMAP per-source registration quality gate does not "
                "match this run"
            )
        _validate_source_registration(
            metadata,
            stats,
            min_source_registration_ratio,
        )
    if stats.get("num_input_images") != expected_input_images:
        raise ValueError("cached COLMAP input-image count does not match frame cache")
    if stats.get("selected_model_path") != "sparse/0":
        raise ValueError("cached COLMAP selected-model marker is invalid")
    if stats.get("status") == "passed":
        recorded_files = stats.get("selected_model_files")
        if not isinstance(recorded_files, dict):
            raise ValueError("cached COLMAP model-integrity record is missing")
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            path = dataset / "sparse" / "0" / name
            expected = recorded_files.get(name)
            if not path.is_file() or not isinstance(expected, dict):
                raise ValueError(f"cached COLMAP model file is missing: {name}")
            if expected.get("size_bytes") != path.stat().st_size:
                raise ValueError(f"cached COLMAP model file size changed: {name}")
            if expected.get("sha256") != _sha256(path):
                raise ValueError(f"cached COLMAP model file content changed: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", nargs="+", type=Path)
    parser.add_argument("--sample-fps", required=True, type=float)
    parser.add_argument("--max-frames", required=True, type=int)
    parser.add_argument("--camera-model")
    parser.add_argument(
        "--matching-strategy",
        choices=("sequential", "exhaustive"),
    )
    parser.add_argument("--overlap", type=int)
    parser.add_argument(
        "--quadratic-overlap",
        action="store_true",
        default=None,
    )
    parser.add_argument("--min-registration-ratio", type=float)
    parser.add_argument("--min-source-registration-ratio", type=float)
    parser.add_argument("--require-colmap", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_cache(
            args.dataset,
            args.source,
            args.sample_fps,
            args.max_frames,
            args.camera_model,
            args.overlap,
            args.quadratic_overlap,
            args.min_registration_ratio,
            args.require_colmap,
            args.matching_strategy,
            args.min_source_registration_ratio,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
