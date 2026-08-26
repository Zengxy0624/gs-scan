#!/usr/bin/env python3
"""Atomically merge existing single-video frame caches into one video set."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Sequence


SOURCE_RECORD_FIELDS = (
    "source_video",
    "source_size_bytes",
    "source_mtime_ns",
    "source_sha256",
    "source_fps",
    "total_source_frames",
    "duration_seconds",
    "selected_indices",
    "selected_timestamps_seconds",
    "source_frame_width",
    "source_frame_height",
    "output_frame_width",
    "output_frame_height",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_tree_digest(
    digest: hashlib._Hash,
    relative_name: str,
    file_sha256: str,
) -> None:
    digest.update(relative_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(file_sha256.encode("ascii"))
    digest.update(b"\0")


def _read_metadata(dataset: Path) -> dict[str, object]:
    metadata_path = dataset / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Frame metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Frame metadata must be a JSON object: {metadata_path}")
    return metadata


def _source_prefix(index: int, dataset: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", dataset.name.lower()).strip("_")
    return f"{index:02d}_{stem or 'video'}_"


def _validate_source_identity(metadata: dict[str, object], dataset: Path) -> None:
    source_value = metadata.get("source_video")
    if not isinstance(source_value, str):
        raise ValueError(f"Single-video source path is missing: {dataset}")
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")
    source_stat = source.stat()
    if metadata.get("source_size_bytes") != source_stat.st_size:
        raise ValueError(f"Source video size changed since extraction: {source}")
    recorded_digest = metadata.get("source_sha256")
    if not isinstance(recorded_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", recorded_digest
    ) is None:
        raise ValueError(f"Source video digest is invalid: {dataset}")
    recorded_path = Path(source_value).expanduser().resolve()
    if (
        recorded_path != source
        or metadata.get("source_mtime_ns") != source_stat.st_mtime_ns
    ) and _sha256(source) != recorded_digest:
        raise ValueError(f"Source video content changed since extraction: {source}")


def _link_or_copy(source: Path, target: Path, link_mode: str) -> None:
    if link_mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError as error:
        if link_mode != "auto" or error.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
        }:
            raise
        shutil.copy2(source, target)


def merge_frame_datasets(
    source_datasets: Sequence[Path],
    output_dataset: Path,
    link_mode: str = "hardlink",
) -> Path:
    """Merge validated frame caches with unique prefixes and no re-encoding."""
    if not source_datasets:
        raise ValueError("At least one source frame dataset is required")
    if link_mode not in {"hardlink", "auto", "copy"}:
        raise ValueError("link_mode must be 'hardlink', 'auto', or 'copy'")

    sources = [path.expanduser().resolve() for path in source_datasets]
    if len(set(sources)) != len(sources):
        raise ValueError("Source frame dataset paths must be unique")
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"Frame dataset not found: {source}")

    output = output_dataset.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output dataset already exists: {output}")
    if any(output == source or output.is_relative_to(source) for source in sources):
        raise ValueError("Output dataset must not be inside a source dataset")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        output_images = temporary / "images"
        output_images.mkdir()
        merged_digest = hashlib.sha256()
        source_records: list[dict[str, object]] = []
        total_selected_frames = 0
        expected_recipe: tuple[float, int, int] | None = None
        expected_dimensions: tuple[int, int] | None = None

        for index, source_dataset in enumerate(sources):
            metadata = _read_metadata(source_dataset)
            if metadata.get("dataset_type") != "single_video":
                raise ValueError(
                    f"Expected a single_video frame cache: {source_dataset}"
                )
            try:
                uuid.UUID(str(metadata.get("dataset_instance_id")))
            except (ValueError, AttributeError):
                raise ValueError(
                    f"Frame cache has no valid instance ID: {source_dataset}"
                ) from None
            _validate_source_identity(metadata, source_dataset)

            sample_fps = float(metadata.get("sample_fps", -1))
            max_frames = metadata.get("max_frames")
            jpeg_quality = metadata.get("jpeg_quality", 95)
            if (
                not math.isfinite(sample_fps)
                or sample_fps <= 0
                or not isinstance(max_frames, int)
                or max_frames < 2
                or not isinstance(jpeg_quality, int)
                or not 1 <= jpeg_quality <= 100
            ):
                raise ValueError(f"Invalid extraction recipe: {source_dataset}")
            recipe = (sample_fps, max_frames, jpeg_quality)
            if expected_recipe is None:
                expected_recipe = recipe
            elif recipe != expected_recipe:
                raise ValueError("All frame caches must use one extraction recipe")

            dimensions = (
                metadata.get("output_frame_width"),
                metadata.get("output_frame_height"),
            )
            if not all(
                isinstance(value, int) and value > 0 for value in dimensions
            ):
                raise ValueError(f"Invalid output dimensions: {source_dataset}")
            if expected_dimensions is None:
                expected_dimensions = dimensions
            elif dimensions != expected_dimensions:
                raise ValueError(
                    "All frame caches must have identical output dimensions"
                )

            selected_indices = metadata.get("selected_indices")
            if not isinstance(selected_indices, list) or len(selected_indices) < 2:
                raise ValueError(
                    f"Frame cache has no valid selected frame list: {source_dataset}"
                )
            expected_names = [
                f"{frame_index:06d}.jpg"
                for frame_index in range(len(selected_indices))
            ]
            source_images = source_dataset / "images"
            if not source_images.is_dir():
                raise FileNotFoundError(f"Image directory not found: {source_images}")
            actual_names = sorted(
                path.relative_to(source_images).as_posix()
                for path in source_images.rglob("*")
                if path.is_file()
            )
            if actual_names != expected_names:
                raise ValueError(
                    f"Frame set is incomplete or unexpected in {source_images}: "
                    f"expected {len(expected_names)}, found {len(actual_names)}"
                )

            prefix = _source_prefix(index, source_dataset)
            source_digest = hashlib.sha256()
            for source_name in expected_names:
                source_image = source_images / source_name
                file_digest = _sha256(source_image)
                _update_tree_digest(source_digest, source_name, file_digest)
                target_name = f"{prefix}{source_name}"
                _update_tree_digest(merged_digest, target_name, file_digest)
                _link_or_copy(
                    source_image,
                    output_images / target_name,
                    link_mode,
                )
            if metadata.get("images_sha256") != source_digest.hexdigest():
                raise ValueError(
                    f"Frame content no longer matches metadata: {source_dataset}"
                )

            missing_fields = [
                field for field in SOURCE_RECORD_FIELDS if field not in metadata
            ]
            if missing_fields:
                raise ValueError(
                    f"Frame metadata is missing {', '.join(missing_fields)}: "
                    f"{source_dataset}"
                )
            source_records.append(
                {
                    **{field: metadata[field] for field in SOURCE_RECORD_FIELDS},
                    "image_prefix": prefix,
                }
            )
            total_selected_frames += len(selected_indices)

        if expected_recipe is None or expected_dimensions is None:
            raise RuntimeError("No frame datasets were merged")
        merged_metadata = {
            "dataset_type": "video_set",
            "dataset_instance_id": str(uuid.uuid4()),
            "sources": source_records,
            "total_selected_frames": total_selected_frames,
            "output_frame_width": expected_dimensions[0],
            "output_frame_height": expected_dimensions[1],
            "images_sha256": merged_digest.hexdigest(),
            "sample_fps": expected_recipe[0],
            "max_frames": expected_recipe[1],
            "jpeg_quality": expected_recipe[2],
        }
        (temporary / "metadata.json").write_text(
            json.dumps(merged_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Merged {len(sources)} frame datasets into {output}")
    print(f"Frames: {total_selected_frames}")
    print(f"Images: {output / 'images'}")
    print(f"Metadata: {output / 'metadata.json'}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Fresh merged dataset path.")
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Ordered single-video frame dataset paths.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "auto", "copy"),
        default="hardlink",
        help=(
            "Reuse JPEG storage with hard links, fall back to copies, or "
            "always copy (default: hardlink)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        merge_frame_datasets(args.sources, args.output, args.link_mode)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
