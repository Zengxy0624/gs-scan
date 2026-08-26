#!/usr/bin/env python3
"""Build a blur-filtered copy of a reconstructed dataset without re-running BA.

Frames failing the sharpness rule (relative ratio below a threshold OR absolute
variance-of-Laplacian below a floor) are excluded from the copied images/ and
deregistered from a copy of the COLMAP model. Poses of surviving frames are
untouched, so no new bundle adjustment is needed and the base dataset is never
written to. A protected-names list (e.g. bootstrap candidates that are not in
the score file) is always kept.

Example:
  python scripts/filter_sharp_dataset.py \
      work/ambulance/merged_gpu_r1 work/ambulance/merged_gpu_r1_sharp050a20 \
      --scores work/ambulance/merged_gpu_sharp050a20/sharpness_scores.csv \
      --ratio-threshold 0.50 --raw-floor 20 \
      --protect-names work/ambulance/merged_gpu_r1/candidate_names.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

try:
    from merge_frame_datasets import _link_or_copy
except ImportError:  # imported as scripts.filter_sharp_dataset
    from scripts.merge_frame_datasets import _link_or_copy


MODEL_FILE_NAMES = ("cameras.bin", "images.bin", "points3D.bin")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="Reconstructed dataset to filter.")
    parser.add_argument("output", type=Path, help="Fresh filtered dataset path.")
    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help=(
            "Sharpness CSV with name,raw,ratio_local columns (as written by "
            "the scoring step or a previously filtered dataset)."
        ),
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.50,
        help=(
            "Remove frames whose sharpness ratio against their temporal "
            "neighborhood is below this (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--raw-floor",
        type=float,
        default=20.0,
        help=(
            "Also remove frames whose absolute variance-of-Laplacian is below "
            "this, catching uniformly blurry segments (default: %(default)s; "
            "0 disables)."
        ),
    )
    parser.add_argument(
        "--protect-names",
        type=Path,
        help=(
            "Optional file with one image name per line that is always kept "
            "(e.g. bootstrap candidates absent from the score file)."
        ),
    )
    return parser


def _load_scores(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Score file not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Score file is empty: {path}")
    missing = {"name", "raw", "ratio_local"} - set(rows[0])
    if missing:
        raise ValueError(
            f"Score file lacks required columns {sorted(missing)}: {path}"
        )
    return rows


def _removal_set(
    rows: Sequence[dict[str, str]],
    ratio_threshold: float,
    raw_floor: float,
) -> set[str]:
    removed: set[str] = set()
    for row in rows:
        ratio = float(row["ratio_local"])
        raw = float(row["raw"])
        if not (math.isfinite(ratio) and math.isfinite(raw)):
            raise ValueError(f"Non-finite sharpness score for {row['name']}")
        if ratio < ratio_threshold or raw < raw_floor:
            removed.add(row["name"])
    return removed


def filter_sharp_dataset(
    base: Path,
    output: Path,
    scores: Path,
    ratio_threshold: float = 0.50,
    raw_floor: float = 20.0,
    protect_names: Path | None = None,
) -> Path:
    if not 0.0 < ratio_threshold < 1.5:
        raise ValueError("--ratio-threshold must be in (0, 1.5)")
    if raw_floor < 0.0 or not math.isfinite(raw_floor):
        raise ValueError("--raw-floor must be finite and non-negative")

    base = base.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output dataset already exists: {output}")
    if output == base or output.is_relative_to(base):
        raise ValueError("Output dataset must not be inside the base dataset")
    image_dir = base / "images"
    model_dir = base / "sparse" / "0"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    missing_model = [
        name for name in MODEL_FILE_NAMES if not (model_dir / name).is_file()
    ]
    if missing_model:
        raise FileNotFoundError(
            f"Base model incomplete at {model_dir}: missing "
            + ", ".join(missing_model)
        )

    rows = _load_scores(scores)
    scored_names = {row["name"] for row in rows}
    protected: set[str] = set()
    if protect_names is not None:
        if not protect_names.is_file():
            raise FileNotFoundError(f"Protect list not found: {protect_names}")
        protected = set(protect_names.read_text(encoding="utf-8").split())

    all_names = sorted(
        path.name for path in image_dir.iterdir() if path.is_file()
    )
    unaccounted = [
        name
        for name in all_names
        if name not in scored_names and name not in protected
    ]
    if unaccounted:
        raise ValueError(
            f"{len(unaccounted)} base images are neither scored nor protected "
            f"(first: {unaccounted[0]}); refusing to guess their sharpness"
        )

    removed = _removal_set(rows, ratio_threshold, raw_floor) - protected
    kept = [name for name in all_names if name not in removed]
    if not kept:
        raise ValueError("Filter would remove every image; aborting")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        staged_images = temporary / "images"
        staged_images.mkdir()
        for name in kept:
            _link_or_copy(image_dir / name, staged_images / name, "auto")

        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(model_dir))
        registered_before = reconstruction.num_reg_images()
        images_by_name = {
            image.name: image for image in reconstruction.images.values()
        }
        deregistered = 0
        for name in sorted(removed):
            image = images_by_name.get(name)
            if image is not None and image.has_pose:
                reconstruction.deregister_frame(image.frame_id)
                deregistered += 1
        staged_model = temporary / "sparse" / "0"
        staged_model.mkdir(parents=True)
        reconstruction.write(str(staged_model))

        filter_record = {
            "base_dataset": str(base),
            "scores_file": str(scores),
            "rule": (
                f"remove if ratio_local < {ratio_threshold} "
                f"or raw < {raw_floor}; protected names always kept"
            ),
            "ratio_threshold": ratio_threshold,
            "raw_floor": raw_floor,
            "num_removed": len(removed),
            "num_kept_images": len(kept),
            "num_protected": len(protected),
            "note": (
                "database.db intentionally omitted; use the base dataset for "
                "any re-matching"
            ),
        }
        metadata_path = base / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        metadata["sharpness_filter"] = filter_record
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        stats_path = base / "reconstruction_stats.json"
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["sharpness_filter"] = filter_record
            stats["num_input_images"] = len(kept)
            stats["num_registered_images"] = reconstruction.num_reg_images()
            stats["num_points3D"] = reconstruction.num_points3D()
            (temporary / "reconstruction_stats.json").write_text(
                json.dumps(stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        with (temporary / "sharpness_scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0].keys()) + ["removed"]
            )
            writer.writeheader()
            writer.writerows(
                {**row, "removed": row["name"] in removed} for row in rows
            )
        if protected:
            (temporary / "protected_names.txt").write_text(
                "\n".join(sorted(protected)) + "\n", encoding="utf-8"
            )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"{output}: kept {len(kept)}/{len(all_names)} images "
        f"(removed {len(removed)}, protected {len(protected)}); registered "
        f"{registered_before} -> {reconstruction.num_reg_images()} "
        f"(deregistered {deregistered})"
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        filter_sharp_dataset(
            args.base,
            args.output,
            args.scores,
            args.ratio_threshold,
            args.raw_floor,
            args.protect_names,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
