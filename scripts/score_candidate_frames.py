#!/usr/bin/env python3
"""Rank registered candidate frames by rendering quality for hard-example mining.

Renders every registered candidate frame from a baked gaussian-splat PLY
export at its COLMAP pose, compares the render with the candidate's actual
photo, and writes a ranked CSV where the worst-reconstructed frames come
first — those are the highest-value additions for finetuning.

The candidate poses live in an *augmented* reconstruction (``--dataset``),
but the trained model lives in the normalized world of the *original*
training dataset (``--base-dataset``): the trainer's world-normalization
similarity depends on the image set, so it is recomputed here exactly as
training computed it (gsplat's ``datasets.colmap.Parser``) and applied to
the candidate poses.

Exposure differences between source videos must not pollute the ranking, so
frames are ranked by PSNR measured *after* a per-image, per-channel affine
color fit of the render to the photo (``psnr_affine``, ascending).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1] / "third-party" / "gsplat" / "examples"

CSV_FIELDS = (
    "rank",
    "name",
    "source",
    "registered",
    "psnr_raw",
    "psnr_affine",
    "affine_gain_r",
    "affine_gain_g",
    "affine_gain_b",
)

# DC spherical-harmonics basis constant used by 3DGS PLY exports.
SH_C0 = 0.28209479177387814


def source_prefix(image_name: str) -> str:
    """'05_a001_08071524_c009_m001142.jpg' -> '05_a001_08071524_c009_'."""
    stem = image_name.rsplit("_", 1)
    return stem[0] + "_" if len(stem) == 2 else image_name


def psnr(rendered: np.ndarray, reference: np.ndarray) -> float:
    """PSNR in dB between two arrays of values in [0, 1]."""
    rendered = np.asarray(rendered, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if rendered.shape != reference.shape:
        raise ValueError(
            f"PSNR inputs must have identical shapes: "
            f"{rendered.shape} vs {reference.shape}"
        )
    mse = float(np.mean((rendered - reference) ** 2))
    if mse == 0.0:
        return math.inf
    return -10.0 * math.log10(mse)


def affine_fit(rendered: np.ndarray, reference: np.ndarray) -> tuple[
    np.ndarray, np.ndarray
]:
    """Per-channel least-squares (gain, bias) mapping ``rendered`` onto ``reference``.

    Minimizes ``||gain * rendered + bias - reference||^2`` independently per
    color channel over all pixels, in closed form. Returns two ``(3,)`` arrays.
    """
    rendered = np.asarray(rendered, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if rendered.shape != reference.shape:
        raise ValueError(
            f"Affine-fit inputs must have identical shapes: "
            f"{rendered.shape} vs {reference.shape}"
        )
    if rendered.ndim < 1 or rendered.shape[-1] != 3:
        raise ValueError(
            f"Affine fit expects trailing RGB channels, got shape {rendered.shape}"
        )
    x = rendered.reshape(-1, 3)
    y = reference.reshape(-1, 3)
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    var_x = ((x - mean_x) ** 2).mean(axis=0)
    cov_xy = ((x - mean_x) * (y - mean_y)).mean(axis=0)
    gain = np.where(var_x > 1e-12, cov_xy / np.maximum(var_x, 1e-12), 1.0)
    bias = mean_y - gain * mean_x
    return gain, bias


def apply_affine(
    rendered: np.ndarray, gain: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Apply a per-channel affine correction and clamp the result to [0, 1]."""
    corrected = np.asarray(rendered, dtype=np.float64) * gain + bias
    return np.clip(corrected, 0.0, 1.0)


def apply_parser_transform(parser: Any, camtoworld: np.ndarray) -> np.ndarray:
    """Apply the training-time world normalization to one camera-to-world pose.

    gsplat's ``datasets.colmap.Parser`` stores the applied similarity as the
    ``parser.transform`` attribute (a 4x4 matrix) and applies it to poses via
    ``datasets.normalize.transform_cameras``: left-multiply the pose, then
    rescale the rotation block by the norm of its first row so it stays a
    rotation. This replicates that exact operation for a single pose.
    """
    matrix = np.asarray(parser.transform, dtype=np.float64)
    pose = np.asarray(camtoworld, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Parser transform must be 4x4, got {matrix.shape}")
    if pose.shape != (4, 4):
        raise ValueError(f"Camera-to-world pose must be 4x4, got {pose.shape}")
    transformed = matrix @ pose
    scaling = float(np.linalg.norm(transformed[0, :3]))
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("Parser transform produced a degenerate rotation")
    transformed[:3, :3] = transformed[:3, :3] / scaling
    return transformed


def score_candidates(
    names: Sequence[str],
    render_frame: Callable[[str], np.ndarray | None],
    load_frame: Callable[[str], np.ndarray],
) -> list[dict[str, object]]:
    """Score each candidate; ``render_frame`` returns None for unregistered names."""
    rows: list[dict[str, object]] = []
    for name in names:
        rendered = render_frame(name)
        if rendered is None:
            rows.append(
                {
                    "name": name,
                    "source": source_prefix(name),
                    "registered": False,
                    "psnr_raw": None,
                    "psnr_affine": None,
                    "affine_gain_r": None,
                    "affine_gain_g": None,
                    "affine_gain_b": None,
                }
            )
            continue
        reference = load_frame(name)
        if rendered.shape != reference.shape:
            raise ValueError(
                f"Render/photo shape mismatch for {name}: "
                f"{rendered.shape} vs {reference.shape}"
            )
        gain, bias = affine_fit(rendered, reference)
        corrected = apply_affine(rendered, gain, bias)
        rows.append(
            {
                "name": name,
                "source": source_prefix(name),
                "registered": True,
                "psnr_raw": psnr(rendered, reference),
                "psnr_affine": psnr(corrected, reference),
                "affine_gain_r": float(gain[0]),
                "affine_gain_g": float(gain[1]),
                "affine_gain_b": float(gain[2]),
            }
        )
    return rows


def rank_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Rank scored rows by psnr_affine ascending; unregistered rows go last."""
    scored = sorted(
        (row for row in rows if row["registered"]),
        key=lambda row: (row["psnr_affine"], row["name"]),
    )
    unregistered = [row for row in rows if not row["registered"]]
    ranked = [
        {**row, "rank": rank} for rank, row in enumerate(scored, start=1)
    ]
    ranked.extend({**row, "rank": None} for row in unregistered)
    return ranked


def _csv_record(row: dict[str, object]) -> list[str]:
    record: list[str] = []
    for field in CSV_FIELDS:
        value = row.get(field)
        if value is None:
            record.append("")
        elif field == "registered":
            record.append("true" if value else "false")
        elif field in {"psnr_raw", "psnr_affine"}:
            record.append(f"{value:.4f}")
        elif field.startswith("affine_gain_"):
            record.append(f"{value:.6f}")
        else:
            record.append(str(value))
    return record


def write_scores_csv(
    rows: Sequence[dict[str, object]], output: Path
) -> Path:
    """Atomically write ranked rows as a CSV file."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        staged = temporary / output.name
        with staged.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
            for row in rows:
                writer.writerow(_csv_record(row))
        staged.replace(output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return output


def print_summary(rows: Sequence[dict[str, object]]) -> None:
    scored = [row for row in rows if row["registered"]]
    unregistered = [row for row in rows if not row["registered"]]
    print(f"Scored {len(scored)} candidate frames, {len(unregistered)} unregistered")
    if not scored:
        return
    values = [float(row["psnr_affine"]) for row in scored]
    print(
        f"psnr_affine mean {float(np.mean(values)):.3f} dB, "
        f"min {min(values):.3f} dB"
    )
    print("Worst frames:")
    for row in scored[:10]:
        print(f"  {row['rank']:>4} {row['name']} {row['psnr_affine']:.3f} dB")
    by_source: dict[str, list[float]] = {}
    for row in scored:
        by_source.setdefault(str(row["source"]), []).append(
            float(row["psnr_affine"])
        )
    print("Per-source mean psnr_affine:")
    for source, source_values in sorted(
        by_source.items(), key=lambda item: float(np.mean(item[1]))
    ):
        print(
            f"  {source} {float(np.mean(source_values)):.3f} dB "
            f"(n={len(source_values)})"
        )


def _read_names(names_file: Path) -> list[str]:
    if not names_file.is_file():
        raise FileNotFoundError(f"Candidate name list not found: {names_file}")
    names = [
        line.strip()
        for line in names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"Candidate name list is empty: {names_file}")
    if len(set(names)) != len(names):
        raise ValueError(f"Candidate name list has duplicates: {names_file}")
    return names


def _load_gaussians(ply_path: Path) -> dict[str, np.ndarray]:
    """Load a baked 3DGS PLY export into activated numpy arrays."""
    from plyfile import PlyData

    vertex = PlyData.read(str(ply_path))["vertex"]

    def columns(field_names: Sequence[str]) -> np.ndarray:
        try:
            return np.stack(
                [np.asarray(vertex[field], dtype=np.float32) for field in field_names],
                axis=1,
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"PLY is missing gaussian-splat fields "
                f"{', '.join(field_names)}: {ply_path}"
            ) from error

    means = columns(("x", "y", "z"))
    quats = columns(("rot_0", "rot_1", "rot_2", "rot_3"))
    scales = np.exp(columns(("scale_0", "scale_1", "scale_2")))
    opacities = 1.0 / (
        1.0 + np.exp(-columns(("opacity",))[:, 0].astype(np.float64))
    )
    colors = np.clip(
        SH_C0 * columns(("f_dc_0", "f_dc_1", "f_dc_2")) + 0.5, 0.0, 1.0
    )
    if means.shape[0] == 0:
        raise ValueError(f"PLY contains no gaussians: {ply_path}")
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities.astype(np.float32),
        "colors": colors.astype(np.float32),
    }


def _make_renderer(
    gaussians: dict[str, np.ndarray], device: str
) -> Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]:
    """Build a render(camtoworld, K, width, height) callback on ``device``."""
    import torch
    from gsplat import rasterization

    tensors = {
        key: torch.from_numpy(value).to(device)
        for key, value in gaussians.items()
    }

    @torch.no_grad()
    def render(
        camtoworld: np.ndarray, K: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        viewmat = torch.linalg.inv(
            torch.from_numpy(np.asarray(camtoworld, dtype=np.float32))
        ).to(device)[None]
        Ks = torch.from_numpy(np.asarray(K, dtype=np.float32)).to(device)[None]
        colors, _, _ = rasterization(
            means=tensors["means"],
            quats=tensors["quats"],
            scales=tensors["scales"],
            opacities=tensors["opacities"],
            colors=tensors["colors"],
            viewmats=viewmat,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,
        )
        return np.clip(
            colors[0, ..., :3].detach().cpu().numpy().astype(np.float64),
            0.0,
            1.0,
        )

    return render


def _image_camtoworld(image: Any) -> np.ndarray:
    """Camera-to-world 4x4 from a pycolmap image (mirrors colmap.py)."""
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    w2c = np.eye(4)
    w2c[:3, :4] = np.asarray(cam_from_world.matrix(), dtype=np.float64)
    return np.linalg.inv(w2c)


def _shared_camera_state(parser: Any, downscale: int) -> dict[str, object]:
    """Extract the single shared camera's undistortion + downscaled intrinsics."""
    camera_ids = sorted(set(parser.camera_ids))
    if len(camera_ids) != 1:
        raise ValueError(
            f"Expected one shared camera in the base dataset, "
            f"found {len(camera_ids)}: {camera_ids}"
        )
    camera_id = camera_ids[0]
    K = np.asarray(parser.Ks_dict[camera_id], dtype=np.float64)
    width, height = parser.imsize_dict[camera_id]
    out_width = max(1, int(round(width / downscale)))
    out_height = max(1, int(round(height / downscale)))
    K_scaled = K.copy()
    K_scaled[0, :] *= out_width / width
    K_scaled[1, :] *= out_height / height
    return {
        "params": np.asarray(parser.params_dict[camera_id]),
        "mapx": parser.mapx_dict.get(camera_id),
        "mapy": parser.mapy_dict.get(camera_id),
        "roi_undist": parser.roi_undist_dict.get(camera_id),
        "K": K_scaled,
        "width": out_width,
        "height": out_height,
    }


def _load_candidate_image(
    image_path: Path, camera: dict[str, object]
) -> np.ndarray:
    """Read, undistort (parser maps), and downscale one candidate photo."""
    import cv2
    import imageio.v2 as imageio

    if not image_path.is_file():
        raise FileNotFoundError(f"Candidate image not found: {image_path}")
    image = imageio.imread(str(image_path))[..., :3]
    params = camera["params"]
    if params is not None and len(params) > 0:
        mapx, mapy = camera["mapx"], camera["mapy"]
        if image.shape[:2] != mapx.shape:
            raise ValueError(
                f"Candidate image size {image.shape[1]}x{image.shape[0]} does "
                f"not match the shared camera "
                f"{mapx.shape[1]}x{mapx.shape[0]}: {image_path}"
            )
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = camera["roi_undist"]
        image = image[y : y + h, x : x + w]
    width, height = camera["width"], camera["height"]
    if (image.shape[1], image.shape[0]) != (width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image.astype(np.float64) / 255.0


def score_candidate_frames(
    dataset: Path,
    base_dataset: Path,
    ply: Path,
    output: Path,
    names_file: Path | None = None,
    downscale: int = 4,
    device: str = "cuda",
    limit: int | None = None,
) -> Path:
    """Score candidate frames against the trained model; write a ranked CSV."""
    if downscale < 1:
        raise ValueError(f"--downscale must be a positive integer, got {downscale}")
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be a positive integer, got {limit}")
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"--device must be 'cuda' or 'cpu', got {device!r}")

    dataset = dataset.expanduser().resolve()
    base_dataset = base_dataset.expanduser().resolve()
    ply = ply.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Augmented dataset not found: {dataset}")
    sparse_dir = dataset / "sparse" / "0"
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"Augmented reconstruction not found: {sparse_dir}")
    images_dir = dataset / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Augmented image directory not found: {images_dir}")
    if not base_dataset.is_dir():
        raise FileNotFoundError(f"Base dataset not found: {base_dataset}")
    if not (base_dataset / "sparse" / "0").is_dir() and not (
        base_dataset / "sparse"
    ).is_dir():
        raise FileNotFoundError(
            f"Base reconstruction not found: {base_dataset / 'sparse'}"
        )
    if not ply.is_file():
        raise FileNotFoundError(f"Gaussian-splat PLY not found: {ply}")

    if names_file is None:
        names_file = dataset / "candidate_names.txt"
    names = _read_names(names_file.expanduser().resolve())
    if limit is not None:
        names = names[:limit]

    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available; use --device cpu")

    if str(_EXAMPLES) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES))
    from datasets.colmap import Parser  # noqa: E402 (needs examples on sys.path)

    import pycolmap

    # The base parser recomputes the training-time world normalization and the
    # shared camera's undistortion state exactly as the trainer built them.
    parser = Parser(
        data_dir=str(base_dataset), factor=1, normalize=True, test_every=8
    )
    camera = _shared_camera_state(parser, downscale)

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    registered_images = {
        reconstruction.images[image_id].name: reconstruction.images[image_id]
        for image_id in reconstruction.reg_image_ids()
    }

    renderer = _make_renderer(_load_gaussians(ply), device)

    def render_frame(name: str) -> np.ndarray | None:
        image = registered_images.get(name)
        if image is None:
            return None
        camtoworld = apply_parser_transform(parser, _image_camtoworld(image))
        return renderer(
            camtoworld, camera["K"], camera["width"], camera["height"]
        )

    def load_frame(name: str) -> np.ndarray:
        return _load_candidate_image(images_dir / name, camera)

    ranked = rank_rows(score_candidates(names, render_frame, load_frame))
    write_scores_csv(ranked, output)
    print_summary(ranked)
    print(f"Scores: {output.expanduser().resolve()}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Augmented dataset whose sparse/0 holds the registered candidates.",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        required=True,
        help=(
            "Original training dataset; the world normalization and shared "
            "camera come from it, exactly as training computed them."
        ),
    )
    parser.add_argument(
        "--ply",
        type=Path,
        required=True,
        help="Baked gaussian-splat PLY export of the trained model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ranked scores CSV path (written atomically).",
    )
    parser.add_argument(
        "--names",
        type=Path,
        default=None,
        help=(
            "File listing candidate image names to score, one per line "
            "(default: <dataset>/candidate_names.txt)."
        ),
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=4,
        help="Downscale factor for render and photo before scoring (default: 4).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Torch device used for rendering (default: cuda).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N listed names (default: all).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        score_candidate_frames(
            dataset=args.dataset,
            base_dataset=args.base_dataset,
            ply=args.ply,
            output=args.output,
            names_file=args.names,
            downscale=args.downscale,
            device=args.device,
            limit=args.limit,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
