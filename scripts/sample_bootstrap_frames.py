#!/usr/bin/env python3
"""Extract bootstrap candidate frames from a merged video-set dataset.

Three selection modes, combinable and deduplicated against each other and
against images already present in the base dataset (by output name):

- --all-frames-for-source PREFIX: every consecutive-pair midpoint of the
  source's selected base frames.
- --around-csv CSV --worst N: the midpoints neighboring the N worst-psnr
  parseable base-frame rows of a per-image metrics CSV; rows named like
  prior-round candidates (<prefix>m<index>.jpg) are skipped and counted.
- --dense-range PREFIX:START:END: every source-video frame index in
  [START, END] inclusive that is not already among the source's
  selected_indices.

At least one selection option must be given.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Sequence

try:
    from scripts.merge_frame_datasets import _validate_source_identity
except ImportError:  # Direct execution puts scripts/ itself on sys.path.
    from merge_frame_datasets import _validate_source_identity


_SAMPLE_NAME_PATTERN = re.compile(r"(?P<prefix>.+_)(?P<position>\d{6})\.jpg")


def consecutive_midpoints(selected_indices: Sequence[int]) -> list[int]:
    """Return floor midpoints of consecutive pairs that leave room for a frame."""
    midpoints: list[int] = []
    for first, second in zip(selected_indices, selected_indices[1:]):
        midpoint = (first + second) // 2
        if midpoint != first and midpoint != second:
            midpoints.append(midpoint)
    return midpoints


def _candidate_name(image_prefix: str, source_frame_index: int) -> str:
    return f"{image_prefix}m{source_frame_index:06d}.jpg"


def _read_metadata(dataset: Path) -> dict[str, object]:
    metadata_path = dataset / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {metadata_path}")
    return metadata


def _sources_by_prefix(
    metadata: dict[str, object],
    dataset: Path,
) -> dict[str, dict[str, object]]:
    sources = metadata.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Dataset metadata has no sources list: {dataset}")
    by_prefix: dict[str, dict[str, object]] = {}
    for record in sources:
        if not isinstance(record, dict):
            raise ValueError(f"Dataset source records must be objects: {dataset}")
        prefix = record.get("image_prefix")
        if not isinstance(prefix, str) or not prefix.endswith("_"):
            raise ValueError(f"Source record has no valid image_prefix: {dataset}")
        selected = record.get("selected_indices")
        if (
            not isinstance(selected, list)
            or len(selected) < 2
            or not all(isinstance(index, int) for index in selected)
        ):
            raise ValueError(
                f"Source {prefix} has no valid selected_indices: {dataset}"
            )
        source_fps = record.get("source_fps")
        if (
            not isinstance(source_fps, (int, float))
            or not math.isfinite(source_fps)
            or source_fps <= 0
        ):
            raise ValueError(f"Source {prefix} has no valid source_fps: {dataset}")
        if prefix in by_prefix:
            raise ValueError(f"Duplicate source image_prefix {prefix}: {dataset}")
        by_prefix[prefix] = record
    return by_prefix


def _parse_sample_name(name: str) -> tuple[str, int]:
    match = _SAMPLE_NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Image name is not a sampled base frame "
            f"(<prefix_><position %06d>.jpg): {name!r}"
        )
    return match.group("prefix"), int(match.group("position"))


def _parse_dense_range(argument: str) -> tuple[str, int, int]:
    """Parse one --dense-range PREFIX:START:END argument."""
    parts = argument.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"--dense-range must be PREFIX:START:END: {argument!r}"
        )
    prefix, start_text, end_text = parts
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError:
        raise ValueError(
            f"--dense-range START and END must be integers: {argument!r}"
        ) from None
    return prefix, start, end


def _worst_csv_rows(
    around_csv: Path,
    worst: int,
) -> tuple[list[dict[str, str]], int]:
    with around_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [column for column in ("name", "psnr") if column not in columns]
        if missing:
            raise ValueError(
                f"Metrics CSV is missing the {', '.join(missing)} "
                f"column(s): {around_csv}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Metrics CSV has no data rows: {around_csv}")

    # Rows whose name is not a sampled base frame are prior-round bootstrap
    # candidates (<prefix>m<index>.jpg); skip them before ranking so --worst
    # counts only parseable base-frame rows.
    parseable: list[dict[str, str]] = []
    skipped_unparseable = 0
    for row in rows:
        name = row.get("name")
        if isinstance(name, str) and _SAMPLE_NAME_PATTERN.fullmatch(name):
            parseable.append(row)
        else:
            skipped_unparseable += 1
    if not parseable:
        raise ValueError(
            f"Metrics CSV has no rows naming sampled base frames "
            f"(<prefix_><position %06d>.jpg): {around_csv}"
        )

    def row_psnr(row: dict[str, str]) -> float:
        try:
            value = float(row["psnr"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Metrics CSV has a non-numeric psnr for "
                f"{row.get('name')!r}: {around_csv}"
            ) from None
        if not math.isfinite(value):
            raise ValueError(
                f"Metrics CSV has a non-finite psnr for "
                f"{row.get('name')!r}: {around_csv}"
            )
        return value

    parseable.sort(key=row_psnr)
    return parseable[:worst], skipped_unparseable


def _select_candidates(
    sources: dict[str, dict[str, object]],
    all_frames_for_sources: Sequence[str],
    around_csv: Path | None,
    worst: int | None,
    dense_ranges: Sequence[tuple[str, int, int]],
) -> tuple[set[tuple[str, int]], int]:
    selections: set[tuple[str, int]] = set()
    skipped_unparseable_csv_rows = 0

    for prefix in all_frames_for_sources:
        record = sources.get(prefix)
        if record is None:
            raise ValueError(
                f"--all-frames-for-source prefix {prefix!r} is not in the base "
                f"dataset; known prefixes: {', '.join(sorted(sources))}"
            )
        selected = record["selected_indices"]
        assert isinstance(selected, list)
        for midpoint in consecutive_midpoints(selected):
            selections.add((prefix, midpoint))

    if around_csv is not None:
        assert worst is not None
        worst_rows, skipped_unparseable_csv_rows = _worst_csv_rows(
            around_csv,
            worst,
        )
        for row in worst_rows:
            prefix, position = _parse_sample_name(row["name"])
            record = sources.get(prefix)
            if record is None:
                raise ValueError(
                    f"Metrics CSV row {row['name']!r} does not match any source "
                    f"prefix; known prefixes: {', '.join(sorted(sources))}"
                )
            selected = record["selected_indices"]
            assert isinstance(selected, list)
            if position >= len(selected):
                raise ValueError(
                    f"Metrics CSV row {row['name']!r} is outside the source's "
                    f"{len(selected)} sampled frames"
                )
            for neighbor in (position - 1, position + 1):
                if not 0 <= neighbor < len(selected):
                    continue
                first, second = sorted((selected[position], selected[neighbor]))
                midpoint = (first + second) // 2
                if midpoint != first and midpoint != second:
                    selections.add((prefix, midpoint))

    for prefix, start, end in dense_ranges:
        record = sources.get(prefix)
        if record is None:
            raise ValueError(
                f"--dense-range prefix {prefix!r} is not in the base "
                f"dataset; known prefixes: {', '.join(sorted(sources))}"
            )
        total_frames = record.get("total_source_frames")
        if not isinstance(total_frames, int) or total_frames <= 0:
            raise ValueError(
                f"Source {prefix} has no valid total_source_frames"
            )
        if not 0 <= start <= end < total_frames:
            raise ValueError(
                f"--dense-range {prefix}:{start}:{end} must satisfy "
                f"0 <= START <= END < {total_frames} source frames"
            )
        selected = record["selected_indices"]
        assert isinstance(selected, list)
        used_indices = set(selected)
        for index in range(start, end + 1):
            if index not in used_indices:
                selections.add((prefix, index))

    return selections, skipped_unparseable_csv_rows


def _extract_wanted_frames(
    video_path: Path,
    targets: Sequence[tuple[int, Path]],
    jpeg_quality: int,
) -> None:
    """Decode the video once and write every wanted source frame."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    cursor = 0
    source_index = 0
    try:
        while cursor < len(targets):
            ok, frame = capture.read()
            if not ok:
                break
            if source_index == targets[cursor][0]:
                target = targets[cursor][1]
                saved = cv2.imwrite(
                    str(target),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not saved:
                    raise RuntimeError(f"Failed to write bootstrap frame: {target}")
                cursor += 1
            source_index += 1
    finally:
        capture.release()

    if cursor != len(targets):
        raise RuntimeError(
            f"Video decoding stopped early for {video_path}: "
            f"expected {len(targets)} bootstrap frames, wrote {cursor}."
        )


def sample_bootstrap_frames(
    base_dataset: Path,
    output_dir: Path,
    all_frames_for_sources: Sequence[str] = (),
    around_csv: Path | None = None,
    worst: int | None = None,
    dense_ranges: Sequence[str] = (),
    jpeg_quality: int | None = None,
) -> Path:
    """Extract deduplicated bootstrap candidate frames from source videos."""
    base = base_dataset.expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Base dataset not found: {base}")
    metadata = _read_metadata(base)
    if metadata.get("dataset_type") != "video_set":
        raise ValueError(f"Expected a video_set dataset: {base}")
    try:
        uuid.UUID(str(metadata.get("dataset_instance_id")))
    except (ValueError, AttributeError):
        raise ValueError(f"Base dataset has no valid instance ID: {base}") from None

    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Output directory already exists: {output}. "
            "Use a fresh versioned directory."
        )
    if output == base or output.is_relative_to(base):
        raise ValueError("Output directory must not be inside the base dataset")

    if not all_frames_for_sources and around_csv is None and not dense_ranges:
        raise ValueError(
            "Select candidates with --all-frames-for-source, --around-csv, "
            "and/or --dense-range"
        )
    parsed_dense_ranges = [
        _parse_dense_range(argument) for argument in dense_ranges
    ]
    if (around_csv is None) != (worst is None):
        raise ValueError("--around-csv and --worst must be used together")
    if worst is not None and worst < 1:
        raise ValueError("--worst must be at least 1")
    if around_csv is not None:
        around_csv = around_csv.expanduser().resolve()
        if not around_csv.is_file():
            raise FileNotFoundError(f"Metrics CSV not found: {around_csv}")

    if jpeg_quality is None:
        base_quality = metadata.get("jpeg_quality")
        if not isinstance(base_quality, int) or not 1 <= base_quality <= 100:
            raise ValueError(f"Base dataset has no valid jpeg_quality: {base}")
        jpeg_quality = base_quality
    elif not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    sources = _sources_by_prefix(metadata, base)
    selections, skipped_unparseable_csv_rows = _select_candidates(
        sources,
        all_frames_for_sources,
        around_csv,
        worst,
        parsed_dense_ranges,
    )

    base_images = base / "images"
    if not base_images.is_dir():
        raise FileNotFoundError(f"Base image directory not found: {base_images}")
    existing_names = {path.name for path in base_images.iterdir() if path.is_file()}

    skipped_name_collisions = 0
    candidates: list[tuple[str, int]] = []
    for prefix, index in sorted(selections):
        if _candidate_name(prefix, index) in existing_names:
            skipped_name_collisions += 1
        else:
            candidates.append((prefix, index))
    if not candidates:
        raise ValueError(
            f"No bootstrap candidates remain: {len(selections)} frames "
            f"selected, {skipped_name_collisions} skipped as name collisions"
        )

    indices_by_prefix: dict[str, list[int]] = {}
    for prefix, index in candidates:
        indices_by_prefix.setdefault(prefix, []).append(index)
    for prefix in indices_by_prefix:
        _validate_source_identity(sources[prefix], base)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        images_dir = temporary / "images"
        images_dir.mkdir()
        candidate_records: list[dict[str, object]] = []
        per_source_counts: dict[str, int] = {}
        for prefix, indices in indices_by_prefix.items():
            record = sources[prefix]
            source_fps = float(record["source_fps"])  # type: ignore[arg-type]
            targets = [
                (index, images_dir / _candidate_name(prefix, index))
                for index in indices
            ]
            _extract_wanted_frames(
                Path(str(record["source_video"])),
                targets,
                jpeg_quality,
            )
            per_source_counts[prefix] = len(indices)
            for index in indices:
                candidate_records.append(
                    {
                        "name": _candidate_name(prefix, index),
                        "image_prefix": prefix,
                        "source_frame_index": index,
                        "timestamp_seconds": index / source_fps,
                    }
                )

        candidate_names = sorted(
            str(record["name"]) for record in candidate_records
        )
        output_metadata = {
            "dataset_type": "bootstrap_candidates",
            "dataset_instance_id": str(uuid.uuid4()),
            "base_dataset": str(base),
            "base_dataset_instance_id": metadata["dataset_instance_id"],
            "jpeg_quality": jpeg_quality,
            "num_candidates": len(candidate_records),
            "skipped_name_collisions": skipped_name_collisions,
            "selection": {
                "all_frames_for_sources": list(all_frames_for_sources),
                "around_csv": str(around_csv) if around_csv is not None else None,
                "worst": worst,
                "dense_ranges": [
                    {"image_prefix": prefix, "start": start, "end": end}
                    for prefix, start, end in parsed_dense_ranges
                ],
                "skipped_unparseable_csv_rows": skipped_unparseable_csv_rows,
            },
            "candidates": candidate_records,
            "per_source_counts": per_source_counts,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(output_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "candidate_names.txt").write_text(
            "\n".join(candidate_names) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Wrote {len(candidate_records)} candidate frames to {output / 'images'}")
    print(f"Skipped name collisions: {skipped_name_collisions}")
    print(f"Candidate list: {output / 'candidate_names.txt'}")
    print(f"Metadata: {output / 'metadata.json'}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_dataset",
        type=Path,
        help="Merged video_set dataset directory.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Fresh output directory; images/, metadata.json, and "
            "candidate_names.txt are created here."
        ),
    )
    parser.add_argument(
        "--all-frames-for-source",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Select every consecutive-pair midpoint for the source whose "
            "image_prefix equals PREFIX. Repeatable."
        ),
    )
    parser.add_argument(
        "--around-csv",
        type=Path,
        help=(
            "Per-image metrics CSV with index,name,source,psnr,ssim,lpips "
            "columns; requires --worst."
        ),
    )
    parser.add_argument(
        "--worst",
        type=int,
        help=(
            "Number of lowest-psnr rows of --around-csv to sample the "
            "neighboring midpoints of."
        ),
    )
    parser.add_argument(
        "--dense-range",
        action="append",
        default=[],
        metavar="PREFIX:START:END",
        help=(
            "Select every source-video frame index in [START, END] inclusive "
            "for the source whose image_prefix equals PREFIX, excluding "
            "indices already in its selected_indices. Repeatable."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        help=(
            "OpenCV JPEG quality from 1 to 100 "
            "(default: the base dataset's jpeg_quality)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sample_bootstrap_frames(
            args.base_dataset,
            args.output_dir,
            args.all_frames_for_source,
            args.around_csv,
            args.worst,
            args.dense_range,
            args.jpeg_quality,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
