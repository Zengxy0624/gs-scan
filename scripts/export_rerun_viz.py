#!/usr/bin/env python3
"""Export a COLMAP sparse model to a rerun .rrd file for shareable 3D viewing.

Logs the sparse SfM point cloud plus every registered camera center (colored
by source video) and a thinned set of pinhole frusta, so the spatial
distribution of the data is visible in the rerun viewer (desktop `rerun`
binary, or drag-drop at https://rerun.io/viewer).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


MODEL_FILE_NAMES = ("cameras.bin", "images.bin", "points3D.bin")

# Camera models whose parameter vector starts with fx fy cx cy.
FX_FY_CX_CY_MODELS = frozenset(
    {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "THIN_PRISM_FISHEYE",
    }
)

SUBSAMPLE_SEED = 0
POINT_RADIUS = 0.002
CAMERA_CENTER_RADIUS = 0.01

# matplotlib tab20 as RGB bytes (hardcoded to avoid a matplotlib dependency).
SOURCE_COLOR_TABLE = (
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
)


def source_prefix(image_name: str) -> str:
    """Return the source-video prefix shared by frames of one video."""
    return image_name.rsplit("_", 1)[0] + "_"


def camera_k_matrix(
    model_name: str,
    params: Sequence[float],
) -> list[list[float]]:
    """Build the 3x3 K matrix from an fx-fy-cx-cy-leading parameter vector."""
    if model_name not in FX_FY_CX_CY_MODELS:
        raise ValueError(
            f"Unsupported camera model for pinhole export: {model_name} "
            f"(expected one of {', '.join(sorted(FX_FY_CX_CY_MODELS))})"
        )
    if len(params) < 4:
        raise ValueError(
            f"Camera model {model_name} has {len(params)} parameters; "
            "expected at least fx fy cx cy"
        )
    fx, fy, cx, cy = (float(value) for value in params[:4])
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def export_rerun_viz(
    dataset: Path,
    output: Path,
    max_points: int = 1_000_000,
    every_nth_camera: int = 4,
    application_id: str = "3dgs-health-sfm",
) -> Path:
    """Write the sparse model at dataset/sparse/0 to a fresh .rrd file."""
    if max_points < 1:
        raise ValueError("--max-points must be a positive integer")
    if every_nth_camera < 1:
        raise ValueError("--every-nth-camera must be a positive integer")

    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")
    model_dir = dataset / "sparse" / "0"
    missing = [name for name in MODEL_FILE_NAMES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"COLMAP sparse model files missing under {model_dir}: "
            f"{', '.join(missing)} (expected a reconstructed dataset with "
            "sparse/0/cameras.bin, images.bin, points3D.bin)"
        )

    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pycolmap
    import rerun as rr

    reconstruction = pycolmap.Reconstruction(str(model_dir))

    points = list(reconstruction.points3D.values())
    total_points = len(points)
    positions = np.array(
        [point.xyz for point in points], dtype=np.float64
    ).reshape(total_points, 3)
    colors = np.array(
        [point.color for point in points], dtype=np.uint8
    ).reshape(total_points, 3)
    if total_points > max_points:
        chosen = np.random.default_rng(SUBSAMPLE_SEED).choice(
            total_points, size=max_points, replace=False
        )
        positions = positions[chosen]
        colors = colors[chosen]

    images_by_source: dict[str, list[object]] = {}
    for image in reconstruction.images.values():
        if not getattr(image, "has_pose", True):
            continue
        images_by_source.setdefault(source_prefix(image.name), []).append(image)
    source_colors = {
        source: SOURCE_COLOR_TABLE[index % len(SOURCE_COLOR_TABLE)]
        for index, source in enumerate(sorted(images_by_source))
    }

    rr.init(application_id, spawn=False)
    rr.save(str(output))

    if len(positions):
        rr.log(
            "points",
            rr.Points3D(positions, colors=colors, radii=POINT_RADIUS),
            static=True,
        )

    center_positions: list[object] = []
    center_colors: list[tuple[int, int, int]] = []
    frusta_logged = 0
    for source in sorted(images_by_source):
        members = sorted(images_by_source[source], key=lambda image: image.name)
        for index, image in enumerate(members):
            cam_from_world = image.cam_from_world()
            rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
            translation = np.asarray(cam_from_world.translation, dtype=np.float64)
            world_rotation = rotation.T
            center = -world_rotation @ translation
            center_positions.append(center)
            center_colors.append(source_colors[source])
            if index % every_nth_camera:
                continue
            camera = reconstruction.cameras[image.camera_id]
            model_name = getattr(camera.model, "name", str(camera.model))
            k_matrix = camera_k_matrix(model_name, camera.params)
            entity = f"cameras/{source}/{image.name}"
            rr.log(
                entity,
                rr.Transform3D(mat3x3=world_rotation, translation=center),
                static=True,
            )
            rr.log(
                entity,
                rr.Pinhole(
                    image_from_camera=k_matrix,
                    resolution=[int(camera.width), int(camera.height)],
                ),
                static=True,
            )
            frusta_logged += 1

    if center_positions:
        rr.log(
            "camera_centers",
            rr.Points3D(
                np.array(center_positions, dtype=np.float64),
                colors=np.array(center_colors, dtype=np.uint8),
                radii=CAMERA_CENTER_RADIUS,
            ),
            static=True,
        )

    recording = rr.get_global_data_recording()
    if recording is not None:
        recording.flush()
    rr.disconnect()

    size_bytes = output.stat().st_size
    print(
        f"Logged {len(positions)}/{total_points} sparse points, "
        f"{len(center_positions)} camera centers across "
        f"{len(images_by_source)} sources, {frusta_logged} camera frusta"
    )
    print(f"Wrote {output} ({size_bytes / 1_000_000:.1f} MB)")
    print(
        "View it in the rerun desktop viewer (`rerun <file>.rrd`) or "
        "drag-drop the file at https://rerun.io/viewer"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help=(
            "Reconstructed dataset directory containing "
            "sparse/0/{cameras,images,points3D}.bin."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh .rrd output path (must not already exist).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1_000_000,
        help=(
            "Deterministically subsample the point cloud beyond this many "
            "points (default: 1000000)."
        ),
    )
    parser.add_argument(
        "--every-nth-camera",
        type=int,
        default=4,
        help=(
            "Log a pinhole frustum for every Nth image per source video; "
            "all camera centers are always logged (default: 4)."
        ),
    )
    parser.add_argument(
        "--application-id",
        default="3dgs-health-sfm",
        help="Rerun application ID for the recording (default: 3dgs-health-sfm).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        export_rerun_viz(
            args.dataset,
            args.output,
            max_points=args.max_points,
            every_nth_camera=args.every_nth_camera,
            application_id=args.application_id,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
