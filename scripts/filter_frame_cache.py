#!/usr/bin/env python3
"""Build a blur-filtered copy of an extracted single-video frame cache.

Reads the cache's ``sharpness_scores.csv`` (written by ``extract_frames.py
--sharpness-window``), drops every image whose ``ratio_local`` is below a
threshold (or whose ``raw`` score is below an optional floor), and writes a
NEW cache directory: surviving images are hard-linked and renumbered
contiguously, ``metadata.json`` keeps the source identity and extraction
recipe but lists only surviving frames, and both score CSVs are rewritten
with the new names. The input cache is never modified. The threshold is
recorded as ``sharpness_drop_ratio`` / ``sharpness_drop_raw_floor`` so the
merger refuses to mix filtered and unfiltered caches.

Example:
  python scripts/filter_frame_cache.py \\
      work/jump_center/JC_06_1257 work/jump_center/JC_06_1257_sharp070 \\
      --ratio-threshold 0.7
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Sequence

PER_FRAME_FIELDS = (
    "selected_indices",
    "selected_timestamps_seconds",
    "anchor_indices",
    "anchor_sharpness",
    "selected_sharpness",
)


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


def _load_scores(path: Path, expected_names: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Sharpness scores not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Sharpness scores are empty: {path}")
    missing = {"name", "raw", "ratio_local"} - set(rows[0])
    if missing:
        raise ValueError(f"Score file lacks columns {sorted(missing)}: {path}")
    if [row["name"] for row in rows] != list(expected_names):
        raise ValueError(
            f"Score rows do not match the cache's frame list: {path}"
        )
    return rows


def removal_positions(
    rows: Sequence[dict[str, str]],
    ratio_threshold: float,
    raw_floor: float,
) -> list[int]:
    """Positions (row indices) of frames failing the sharpness rule."""
    removed: list[int] = []
    for position, row in enumerate(rows):
        ratio = float(row["ratio_local"])
        raw = float(row["raw"])
        if not (math.isfinite(ratio) and math.isfinite(raw)):
            raise ValueError(f"Non-finite sharpness score for {row['name']}")
        if ratio < ratio_threshold or raw < raw_floor:
            removed.append(position)
    return removed


def filter_frame_cache(
    source: Path,
    output: Path,
    ratio_threshold: float = 0.7,
    raw_floor: float = 0.0,
    link_mode: str = "hardlink",
) -> dict[str, object]:
    """Write the filtered copy and return a summary dictionary."""
    if not (math.isfinite(ratio_threshold) and 0 <= ratio_threshold):
        raise ValueError("ratio_threshold must be zero or positive")
    if not (math.isfinite(raw_floor) and raw_floor >= 0):
        raise ValueError("raw_floor must be zero or positive")
    if link_mode not in {"hardlink", "auto", "copy"}:
        raise ValueError("link_mode must be 'hardlink', 'auto', or 'copy'")

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Frame metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_type") != "single_video":
        raise ValueError(f"Expected a single_video frame cache: {source}")
    if float(metadata.get("sharpness_window_seconds", 0.0)) <= 0:
        raise ValueError(
            f"Cache was extracted without --sharpness-window (no scores): {source}"
        )
    if float(metadata.get("sharpness_drop_ratio", 0.0)) > 0 or float(
        metadata.get("sharpness_drop_raw_floor", 0.0)
    ) > 0:
        raise ValueError(f"Cache is already a filtered copy: {source}")
    selected = metadata.get("selected_indices")
    if not isinstance(selected, list) or len(selected) < 2:
        raise ValueError(f"Cache has no valid selected frame list: {source}")
    for field in PER_FRAME_FIELDS:
        value = metadata.get(field)
        if not isinstance(value, list) or len(value) != len(selected):
            raise ValueError(f"Metadata field {field} is missing or inconsistent")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output dataset already exists: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Output dataset must not be inside the source cache")

    names = [f"{index:06d}.jpg" for index in range(len(selected))]
    images_dir = source / "images"
    for name in names:
        if not (images_dir / name).is_file():
            raise FileNotFoundError(f"Cache image missing: {images_dir / name}")
    if _tree_sha256(images_dir) != metadata.get("images_sha256"):
        raise ValueError(f"Cache images no longer match their metadata: {source}")

    rows = _load_scores(source / "sharpness_scores.csv", names)
    removed = set(removal_positions(rows, ratio_threshold, raw_floor))
    kept = [position for position in range(len(names)) if position not in removed]
    if len(kept) < 2:
        raise ValueError("Filtering would leave fewer than two frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        out_images = temporary / "images"
        out_images.mkdir()
        new_names: list[str] = []
        for new_position, old_position in enumerate(kept):
            new_name = f"{new_position:06d}.jpg"
            _link_or_copy(images_dir / names[old_position], out_images / new_name, link_mode)
            new_names.append(new_name)

        with (temporary / "sharpness_scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for new_name, old_position in zip(new_names, kept, strict=True):
                writer.writerow({**rows[old_position], "name": new_name})

        removed_source_indices = {int(rows[p]["source_index"]) for p in removed} if (
            "source_index" in rows[0]
        ) else set()
        all_frames_path = source / "sharpness_all_frames.csv"
        if all_frames_path.is_file():
            with all_frames_path.open(newline="", encoding="utf-8") as src, (
                temporary / "sharpness_all_frames.csv"
            ).open("w", newline="", encoding="utf-8") as dst:
                reader = csv.DictReader(src)
                writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
                writer.writeheader()
                for row in reader:
                    if (
                        "is_selected" in row
                        and int(row["source_index"]) in removed_source_indices
                    ):
                        row = {**row, "is_selected": "0"}
                    writer.writerow(row)

        filtered_metadata = dict(metadata)
        for field in PER_FRAME_FIELDS:
            filtered_metadata[field] = [metadata[field][p] for p in kept]
        filtered_metadata.update(
            {
                "dataset_instance_id": str(uuid.uuid4()),
                "images_sha256": _tree_sha256(out_images),
                "sharpness_drop_ratio": float(ratio_threshold),
                "sharpness_drop_raw_floor": float(raw_floor),
                "filtered_from": str(source),
                "filtered_from_instance_id": metadata.get("dataset_instance_id"),
                "removed_frame_count": len(removed),
                "removed_source_indices": sorted(removed_source_indices),
            }
        )
        (temporary / "metadata.json").write_text(
            json.dumps(filtered_metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    summary = {
        "source": str(source),
        "output": str(output),
        "input_frames": len(names),
        "kept_frames": len(kept),
        "removed_frames": len(removed),
        "ratio_threshold": ratio_threshold,
        "raw_floor": raw_floor,
    }
    print(
        f"Filtered {source.name}: kept {len(kept)} of {len(names)} frames "
        f"(removed {len(removed)}: ratio_local<{ratio_threshold:g}"
        + (f" or raw<{raw_floor:g}" if raw_floor > 0 else "")
        + f") -> {output}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="Extracted single-video frame cache.")
    parser.add_argument("output", type=Path, help="Fresh filtered cache path.")
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.7,
        help=(
            "Remove frames whose ratio_local (sharpness over the rolling "
            "median of neighboring frames) is below this (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--raw-floor",
        type=float,
        default=0.0,
        help=(
            "Also remove frames whose absolute variance-of-Laplacian is below "
            "this (default: 0 = disabled; absolute scores are lens- and "
            "scene-dependent)."
        ),
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "auto", "copy"),
        default="hardlink",
        help="How surviving JPEGs reuse the source files (default: hardlink).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        filter_frame_cache(
            args.source,
            args.output,
            ratio_threshold=args.ratio_threshold,
            raw_floor=args.raw_floor,
            link_mode=args.link_mode,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
