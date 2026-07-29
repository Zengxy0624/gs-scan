"""Create the sparse COLMAP model expected by gsplat using PyCOLMAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


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
        help="Number of neighboring video frames to match in each direction.",
    )
    parser.add_argument(
        "--camera-model",
        default="SIMPLE_RADIAL",
        choices=(
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "OPENCV_FISHEYE",
        ),
        help="Shared COLMAP camera model for all video frames.",
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
        raise ValueError(f"Need at least two images in {image_dir}; found {len(images)}")

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


def reconstruct(dataset: Path, overlap: int, camera_model: str) -> Path:
    import pycolmap

    dataset = dataset.expanduser().resolve()
    image_dir, images = _validate_input(dataset, overlap)
    database_path = dataset / "database.db"
    sparse_path = dataset / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    print(f"PyCOLMAP {pycolmap.__version__}")
    print(f"Extracting features from {len(images)} images in {image_dir}")
    reader_options = pycolmap.ImageReaderOptions(camera_model=camera_model)
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )

    print(f"Sequentially matching with overlap={overlap}")
    pairing_options = pycolmap.SequentialPairingOptions(overlap=overlap)
    pycolmap.match_sequential(
        database_path=database_path,
        pairing_options=pairing_options,
    )

    print("Running incremental structure-from-motion")
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_path,
    )
    if not reconstructions:
        raise RuntimeError(
            "COLMAP did not recover a model. Increase the video sampling rate, "
            "remove blurred frames, and rerun in a fresh output directory."
        )

    reconstruction = max(
        reconstructions.values(), key=lambda model: model.num_reg_images()
    )
    model_path = sparse_path / "0"
    model_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(model_path)

    registered = reconstruction.num_reg_images()
    points = reconstruction.num_points3D()
    print(reconstruction.summary())
    print(f"Selected largest model: {registered}/{len(images)} registered images")
    print(f"Sparse points: {points}")
    print(f"gsplat COLMAP model: {model_path}")
    if registered < max(10, len(images) // 2):
        print(
            "Warning: relatively few frames registered. Inspect capture overlap "
            "before committing to a full gsplat training run."
        )
    return model_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reconstruct(args.dataset, args.overlap, args.camera_model)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
