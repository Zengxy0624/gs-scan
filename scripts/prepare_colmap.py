"""Create the sparse COLMAP model expected by gsplat using PyCOLMAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CAMERA_MODEL = "OPENCV"


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate video-camera intrinsics, poses, and sparse points with "
            "PyCOLMAP for gsplat training."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset directory containing an images/ folder.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=15,
        help=(
            "Number of quadratic-neighbor levels: frame offsets "
            "1, 2, 4, ..., 2^(N-1) (default: 15)."
        ),
    )
    parser.add_argument(
        "--camera-model",
        default=DEFAULT_CAMERA_MODEL,
        choices=(
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "OPENCV_FISHEYE",
        ),
        help=(
            "Shared COLMAP camera model for all video frames "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=-1,
        help=(
            "Maximum CPU threads for feature extraction, matching, and mapping; "
            "-1 lets COLMAP use all available threads (default: -1)."
        ),
    )
    parser.add_argument(
        "--min-registration-ratio",
        type=float,
        default=0.5,
        help=(
            "Fail unless the largest sparse model registers at least this "
            "fraction of input images and at least 10 images when available "
            "(default: 0.5)."
        ),
    )
    return parser


def _validate_input(dataset: Path, overlap: int) -> tuple[Path, list[Path]]:
    if overlap < 1:
        raise ValueError("--overlap must be at least 1")

    image_dir = dataset / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < 2:
        raise ValueError(
            f"Need at least two images in {image_dir}; found {len(images)}"
        )

    database_path = dataset / "database.db"
    sparse_path = dataset / "sparse"
    if database_path.exists():
        raise FileExistsError(
            f"COLMAP database already exists: {database_path}. "
            "Use a fresh dataset output directory to rerun reconstruction."
        )
    if sparse_path.exists() and any(sparse_path.iterdir()):
        raise FileExistsError(
            f"Sparse output is not empty: {sparse_path}. "
            "Use a fresh dataset output directory to rerun reconstruction."
        )

    return image_dir, images


def _minimum_registered_images(
    total_images: int,
    min_registration_ratio: float,
) -> int:
    if not 0.0 < min_registration_ratio <= 1.0:
        raise ValueError(
            "--min-registration-ratio must be greater than 0 and at most 1"
        )
    return min(
        total_images,
        max(10, math.ceil(total_images * min_registration_ratio)),
    )


def _write_reconstruction_stats(dataset: Path, stats: dict[str, object]) -> Path:
    stats_path = dataset / "reconstruction_stats.json"
    temporary_path = dataset / f".{stats_path.name}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, indent=2, sort_keys=True)
            stats_file.write("\n")
            stats_file.flush()
            os.fsync(stats_file.fileno())
        os.replace(temporary_path, stats_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return stats_path


def reconstruct(
    dataset: Path,
    overlap: int,
    camera_model: str,
    num_threads: int = -1,
    min_registration_ratio: float = 0.5,
) -> Path:
    if num_threads == 0 or num_threads < -1:
        raise ValueError("--num-threads must be -1 or a positive integer")
    _minimum_registered_images(
        total_images=0,
        min_registration_ratio=min_registration_ratio,
    )

    import pycolmap

    dataset = dataset.expanduser().resolve()
    image_dir, images = _validate_input(dataset, overlap)
    minimum_registered = _minimum_registered_images(
        total_images=len(images),
        min_registration_ratio=min_registration_ratio,
    )
    database_path = dataset / "database.db"
    sparse_path = dataset / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    print(f"PyCOLMAP {pycolmap.__version__}")
    print(f"Extracting features from {len(images)} images in {image_dir}")
    reader_options = pycolmap.ImageReaderOptions(camera_model=camera_model)
    extraction_options = pycolmap.FeatureExtractionOptions(num_threads=num_threads)
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=camera_model,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cpu,
    )

    print(
        f"Sequentially matching {overlap} quadratic-neighbor levels "
        "at frame offsets 1, 2, 4, ..."
    )
    pairing_options = pycolmap.SequentialPairingOptions(
        overlap=overlap,
        quadratic_overlap=True,
    )
    matching_options = pycolmap.FeatureMatchingOptions(
        num_threads=num_threads,
        use_gpu=False,
    )
    pycolmap.match_sequential(
        database_path=database_path,
        pairing_options=pairing_options,
        matching_options=matching_options,
        device=pycolmap.Device.cpu,
    )

    print(f"Running incremental structure-from-motion with threads={num_threads}")
    mapping_options = pycolmap.IncrementalPipelineOptions(
        num_threads=num_threads,
    )
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_path,
        options=mapping_options,
    )
    if not reconstructions:
        raise RuntimeError(
            "COLMAP did not recover a model. Increase the video sampling rate, "
            "remove blurred frames, and rerun in a fresh output directory."
        )

    model_id, reconstruction = max(
        reconstructions.items(),
        key=lambda item: item[1].num_reg_images(),
    )

    registered = reconstruction.num_reg_images()
    points = reconstruction.num_points3D()
    registration_ratio = registered / len(images)
    passed_quality_gate = registered >= minimum_registered
    stats = {
        "camera_model": camera_model,
        "matching_overlap": overlap,
        "matching_quadratic_overlap": True,
        "minimum_registered_images": minimum_registered,
        "minimum_registration_ratio": min_registration_ratio,
        "num_input_images": len(images),
        "num_points3D": points,
        "num_registered_images": registered,
        "registration_ratio": registration_ratio,
        "selected_model_id": model_id,
        "selected_model_path": "sparse/0",
        "status": "passed" if passed_quality_gate else "failed",
    }

    print(reconstruction.summary())
    print(f"Selected largest model: {registered}/{len(images)} registered images")
    print(f"Sparse points: {points}")
    if not passed_quality_gate:
        stats_path = _write_reconstruction_stats(dataset, stats)
        print(f"Reconstruction stats: {stats_path}")
        raise RuntimeError(
            "COLMAP registration quality gate failed: "
            f"{registered}/{len(images)} images registered "
            f"({registration_ratio:.1%}); need at least {minimum_registered} "
            f"images ({min_registration_ratio:.1%} requested). "
            "Inspect capture overlap, blur, and camera settings, then rerun in "
            "a fresh output directory."
        )

    model_path = sparse_path / "0"
    model_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(model_path)
    required_model_files = (
        model_path / "cameras.bin",
        model_path / "images.bin",
        model_path / "points3D.bin",
    )
    incomplete_files = [
        str(path)
        for path in required_model_files
        if not path.is_file() or path.stat().st_size == 0
    ]
    if incomplete_files:
        raise RuntimeError(
            "Selected COLMAP model was not written completely: "
            + ", ".join(incomplete_files)
        )

    stats["selected_model_files"] = {
        path.name: _file_identity(path) for path in required_model_files
    }

    # A passed stats file is the final, atomic completion marker. Never write it
    # before the selected model is durably present and structurally complete.
    stats_path = _write_reconstruction_stats(dataset, stats)
    print(f"gsplat COLMAP model: {model_path}")
    print(f"Reconstruction stats: {stats_path}")
    return model_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reconstruct(
            args.dataset,
            args.overlap,
            args.camera_model,
            args.num_threads,
            args.min_registration_ratio,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
