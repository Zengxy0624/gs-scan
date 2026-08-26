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
DEFAULT_VOCAB_TREE_NUM_IMAGES = 100
DEFAULT_FEATURE_MAX_IMAGE_SIZE = 3200
DEFAULT_FEATURE_MAX_NUM_FEATURES = 8192
DEFAULT_INIT_MIN_TRI_ANGLE = 16.0
DEFAULT_BA_GLOBAL_FRAMES_RATIO = 1.1
DEFAULT_BA_GLOBAL_POINTS_RATIO = 1.1
DEFAULT_BA_GLOBAL_MAX_REFINEMENTS = 5
DEFAULT_BA_GLOBAL_MAX_NUM_ITERATIONS = 50


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
        "--matching",
        choices=("sequential", "exhaustive", "vocab_tree"),
        default="sequential",
        help=(
            "Feature-pairing strategy. Sequential uses the quadratic-neighbor "
            "overlap, exhaustive compares every image pair, and vocab_tree "
            "retrieves visually similar images (default: sequential)."
        ),
    )
    parser.add_argument(
        "--vocab-tree-num-images",
        type=int,
        default=DEFAULT_VOCAB_TREE_NUM_IMAGES,
        help=(
            "For vocabulary-tree matching, retrieve this many candidate images "
            "per query (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--vocab-tree",
        type=Path,
        help=(
            "Optional local COLMAP vocabulary-tree file. By default PyCOLMAP "
            "downloads and caches its version-compatible tree."
        ),
    )
    parser.add_argument(
        "--guided-matching",
        action="store_true",
        help="Enable guided feature matching for higher recall at extra cost.",
    )
    parser.add_argument(
        "--sift-affine-shape",
        action="store_true",
        help="Enable affine-shape estimation during SIFT extraction.",
    )
    parser.add_argument(
        "--sift-domain-size-pooling",
        action="store_true",
        help="Enable domain-size pooling during SIFT extraction.",
    )
    parser.add_argument(
        "--feature-max-image-size",
        type=int,
        help="Override COLMAP's maximum image dimension for feature extraction.",
    )
    parser.add_argument(
        "--feature-max-num-features",
        type=int,
        help="Override the maximum number of SIFT features per image.",
    )
    parser.add_argument(
        "--init-min-tri-angle",
        type=float,
        help=(
            "Override the mapper's minimum triangulation angle for the initial "
            "pair, in degrees."
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
    parser.add_argument(
        "--reuse-matches",
        action="store_true",
        help=(
            "Skip feature extraction and matching, mapping directly from the "
            "dataset's existing database.db. The caller must pass the same "
            "extraction and matching options that built the database, because "
            "they are recorded as this reconstruction's provenance."
        ),
    )
    parser.add_argument(
        "--ba-global-frames-ratio",
        type=float,
        help=(
            "Trigger a global bundle adjustment when the model grows by this "
            "frame factor (default: %.1f). Larger values run fewer, later "
            "adjustments and speed up large reconstructions."
            % DEFAULT_BA_GLOBAL_FRAMES_RATIO
        ),
    )
    parser.add_argument(
        "--ba-global-points-ratio",
        type=float,
        help=(
            "Trigger a global bundle adjustment when the model grows by this "
            "point factor (default: %.1f)." % DEFAULT_BA_GLOBAL_POINTS_RATIO
        ),
    )
    parser.add_argument(
        "--ba-global-max-refinements",
        type=int,
        help=(
            "Maximum refinement cycles per global bundle adjustment "
            "(default: %d)." % DEFAULT_BA_GLOBAL_MAX_REFINEMENTS
        ),
    )
    parser.add_argument(
        "--ba-global-max-num-iterations",
        type=int,
        help=(
            "Maximum solver iterations per global bundle adjustment "
            "(default: %d)." % DEFAULT_BA_GLOBAL_MAX_NUM_ITERATIONS
        ),
    )
    parser.add_argument(
        "--snapshot-frames-freq",
        type=int,
        help=(
            "Write a recovery snapshot of the model to <dataset>/snapshots "
            "every N registered frames. Disabled by default; long mapping "
            "runs otherwise hold the model only in memory until completion."
        ),
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        help=(
            "Resume incremental mapping from this saved model directory "
            "(for example a <dataset>/snapshots entry). Requires "
            "--reuse-matches, because mapping continues from the matches "
            "that produced the model."
        ),
    )
    parser.add_argument(
        "--ba-use-gpu",
        action="store_true",
        help=(
            "Solve bundle-adjustment linear systems on a GPU. Requires a "
            "pycolmap whose Ceres was built with CUDA and cuDSS; without "
            "cuDSS, COLMAP silently falls back to CPU sparse solvers."
        ),
    )
    parser.add_argument(
        "--ba-gpu-index",
        default="0",
        help="GPU index for --ba-use-gpu (default: 0).",
    )
    parser.add_argument(
        "--mapping-blas-threads",
        type=int,
        default=48,
        help=(
            "Raise the interposed OpenBLAS to this many threads for the "
            "mapping stage (default: 48; 0 disables). Matching keeps BLAS "
            "single-threaded for faiss safety, but bundle adjustment has a "
            "single BLAS caller whose CHOLMOD factorizations parallelize "
            "well."
        ),
    )
    parser.add_argument(
        "--min-source-registration-ratio",
        type=float,
        default=None,
        help=(
            "For a video_set dataset, additionally require this registration "
            "ratio independently for every source video. Disabled by default."
        ),
    )
    return parser


def _validate_input(
    dataset: Path,
    overlap: int,
    reuse_matches: bool = False,
) -> tuple[Path, list[Path]]:
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
    if reuse_matches:
        if not database_path.is_file() or database_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"--reuse-matches requires an existing COLMAP database: "
                f"{database_path}"
            )
    elif database_path.exists():
        raise FileExistsError(
            f"COLMAP database already exists: {database_path}. "
            "Use a fresh dataset output directory to rerun reconstruction, "
            "or pass --reuse-matches to map from the existing matches."
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


def _video_set_sources(
    dataset: Path,
    min_source_registration_ratio: float | None,
) -> list[dict[str, object]]:
    """Load the source prefixes needed by the optional video-set gate."""
    if min_source_registration_ratio is None:
        return []
    if not 0.0 < min_source_registration_ratio <= 1.0:
        raise ValueError(
            "--min-source-registration-ratio must be greater than 0 and at "
            "most 1"
        )

    metadata_path = dataset / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "--min-source-registration-ratio requires video_set metadata: "
            f"{metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("dataset_type") != "video_set":
        raise ValueError(
            "--min-source-registration-ratio requires dataset_type=video_set"
        )
    raw_sources = metadata.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("video_set metadata must contain a non-empty sources list")

    sources: list[dict[str, object]] = []
    prefixes: list[str] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(f"video_set source {index} must be an object")
        prefix = raw_source.get("image_prefix")
        selected_indices = raw_source.get("selected_indices")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError(
                f"video_set source {index} must have a non-empty image_prefix"
            )
        if not isinstance(selected_indices, list) or not selected_indices:
            raise ValueError(
                f"video_set source {index} must have non-empty selected_indices"
            )
        if any(
            prefix.startswith(other) or other.startswith(prefix)
            for other in prefixes
        ):
            raise ValueError("video_set image_prefix values must not overlap")
        prefixes.append(prefix)
        sources.append(
            {
                "source_index": index,
                "image_prefix": prefix,
                "num_input_images": len(selected_indices),
            }
        )
    return sources


def _source_registration_diagnostics(
    reconstruction: object,
    sources: list[dict[str, object]],
    minimum_ratio: float,
) -> list[dict[str, object]]:
    images = getattr(reconstruction, "images")
    registered_image_ids = getattr(reconstruction, "reg_image_ids")()
    registered_names = [
        str(images[image_id].name) for image_id in registered_image_ids
    ]
    diagnostics: list[dict[str, object]] = []
    for source in sources:
        prefix = str(source["image_prefix"])
        input_count = int(source["num_input_images"])
        registered_count = sum(
            name.startswith(prefix) for name in registered_names
        )
        registration_ratio = registered_count / input_count
        diagnostics.append(
            {
                **source,
                "num_registered_images": registered_count,
                "registration_ratio": registration_ratio,
                "minimum_registration_ratio": minimum_ratio,
                "status": (
                    "passed" if registration_ratio >= minimum_ratio else "failed"
                ),
            }
        )
    return diagnostics


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


def _effective_option(
    options: object,
    name: str,
    fallback: object,
) -> object:
    """Read a PyCOLMAP option while keeping lightweight test doubles simple."""
    return getattr(options, name, fallback)


def _effective_nested_option(
    options: object,
    group: str,
    name: str,
    fallback: object,
) -> object:
    nested = getattr(options, group, None)
    return getattr(nested, name, fallback)


def _raise_blas_threads_for_mapping(threads: int) -> None:
    """Give the LD_PRELOAD-ed pthread OpenBLAS its threads back for mapping.

    Matching runs with OPENBLAS_NUM_THREADS=1 because faiss calls BLAS from
    many OpenMP workers at once. Bundle adjustment instead issues large
    CHOLMOD factorizations from one caller at a time, which the pthread
    OpenBLAS parallelizes internally and safely. Without this, mapping's
    dominant linear solves run on a single core.
    """
    if threads < 1:
        return
    import ctypes

    try:
        # Resolves to the interposed shim by soname when it is preloaded;
        # fails harmlessly when no shim is active (e.g. sequential runs).
        blas = ctypes.CDLL("libopenblas.so.0")
        if blas.openblas_get_parallel() != 1:
            return
        blas.openblas_set_num_threads(threads)
        print(f"Raised interposed OpenBLAS to {threads} threads for mapping")
    except (AttributeError, OSError):
        return


def _vocab_tree_sha256(
    path_spec: str,
    local_file: dict[str, object] | None,
) -> str:
    if local_file is not None:
        return str(local_file["sha256"])

    trailing_digest = path_spec.rsplit(";", maxsplit=1)[-1].lower()
    if len(trailing_digest) == 64 and all(
        character in "0123456789abcdef" for character in trailing_digest
    ):
        return trailing_digest

    path = Path(path_spec).expanduser()
    if path.is_file():
        return str(_file_identity(path.resolve())["sha256"])
    raise RuntimeError(
        "PyCOLMAP's vocabulary-tree specification has no verifiable SHA-256: "
        f"{path_spec}"
    )


def reconstruct(
    dataset: Path,
    overlap: int,
    camera_model: str,
    num_threads: int = -1,
    min_registration_ratio: float = 0.5,
    matching_strategy: str = "sequential",
    min_source_registration_ratio: float | None = None,
    vocab_tree_num_images: int = DEFAULT_VOCAB_TREE_NUM_IMAGES,
    vocab_tree: Path | None = None,
    guided_matching: bool = False,
    sift_affine_shape: bool = False,
    sift_domain_size_pooling: bool = False,
    feature_max_image_size: int | None = None,
    feature_max_num_features: int | None = None,
    init_min_tri_angle: float | None = None,
    reuse_matches: bool = False,
    ba_global_frames_ratio: float | None = None,
    ba_global_points_ratio: float | None = None,
    ba_global_max_refinements: int | None = None,
    ba_global_max_num_iterations: int | None = None,
    snapshot_frames_freq: int | None = None,
    resume_model: Path | None = None,
    mapping_blas_threads: int = 48,
    ba_use_gpu: bool = False,
    ba_gpu_index: str = "0",
) -> Path:
    if num_threads == 0 or num_threads < -1:
        raise ValueError("--num-threads must be -1 or a positive integer")
    for option_name, ratio in (
        ("--ba-global-frames-ratio", ba_global_frames_ratio),
        ("--ba-global-points-ratio", ba_global_points_ratio),
    ):
        if ratio is not None and (
            not math.isfinite(ratio) or ratio <= 1.0
        ):
            raise ValueError(f"{option_name} must be finite and greater than 1")
    for option_name, count in (
        ("--ba-global-max-refinements", ba_global_max_refinements),
        ("--ba-global-max-num-iterations", ba_global_max_num_iterations),
        ("--snapshot-frames-freq", snapshot_frames_freq),
    ):
        if count is not None and count < 1:
            raise ValueError(f"{option_name} must be a positive integer")
    if mapping_blas_threads < 0:
        raise ValueError("--mapping-blas-threads must be 0 or positive")
    resume_model_path: Path | None = None
    if resume_model is not None:
        if not reuse_matches:
            raise ValueError("--resume-model requires --reuse-matches")
        resume_model_path = resume_model.expanduser().resolve()
        missing_model_files = [
            name
            for name in ("cameras.bin", "images.bin", "points3D.bin")
            if not (resume_model_path / name).is_file()
        ]
        if missing_model_files:
            raise FileNotFoundError(
                f"--resume-model is not a complete model: {resume_model_path} "
                f"(missing {', '.join(missing_model_files)})"
            )
    if matching_strategy not in {"sequential", "exhaustive", "vocab_tree"}:
        raise ValueError(
            "--matching must be 'sequential', 'exhaustive', or 'vocab_tree'"
        )
    if vocab_tree_num_images < 1:
        raise ValueError("--vocab-tree-num-images must be positive")
    if feature_max_image_size is not None and feature_max_image_size < 1:
        raise ValueError("--feature-max-image-size must be positive")
    if feature_max_num_features is not None and feature_max_num_features < 1:
        raise ValueError("--feature-max-num-features must be positive")
    if init_min_tri_angle is not None and (
        not math.isfinite(init_min_tri_angle)
        or not 0.0 < init_min_tri_angle <= 180.0
    ):
        raise ValueError(
            "--init-min-tri-angle must be finite, greater than 0, and at most 180"
        )
    _minimum_registered_images(
        total_images=0,
        min_registration_ratio=min_registration_ratio,
    )

    dataset = dataset.expanduser().resolve()
    video_set_sources = _video_set_sources(
        dataset,
        min_source_registration_ratio,
    )
    vocab_tree_path: Path | None = None
    vocab_tree_local_file: dict[str, object] | None = None
    if vocab_tree is not None:
        vocab_tree_path = vocab_tree.expanduser().resolve()
        if not vocab_tree_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary-tree file not found: {vocab_tree_path}"
            )
        vocab_tree_local_file = _file_identity(vocab_tree_path)

    image_dir, images = _validate_input(dataset, overlap, reuse_matches)

    import pycolmap
    minimum_registered = _minimum_registered_images(
        total_images=len(images),
        min_registration_ratio=min_registration_ratio,
    )
    database_path = dataset / "database.db"
    sparse_path = dataset / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    print(f"PyCOLMAP {pycolmap.__version__}")
    reader_options = pycolmap.ImageReaderOptions(camera_model=camera_model)
    extraction_options = pycolmap.FeatureExtractionOptions(num_threads=num_threads)
    if feature_max_image_size is not None:
        extraction_options.max_image_size = feature_max_image_size
    if feature_max_num_features is not None:
        extraction_options.sift.max_num_features = feature_max_num_features
    if sift_affine_shape:
        extraction_options.sift.estimate_affine_shape = True
    if sift_domain_size_pooling:
        extraction_options.sift.domain_size_pooling = True
    if reuse_matches:
        print(
            f"Reusing features and matches for {len(images)} images from "
            f"{database_path}"
        )
    else:
        print(f"Extracting features from {len(images)} images in {image_dir}")
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model=camera_model,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=pycolmap.Device.cpu,
        )

    matching_options = pycolmap.FeatureMatchingOptions(
        num_threads=num_threads,
        use_gpu=False,
    )
    matching_options.guided_matching = guided_matching
    vocab_tree_stats: dict[str, object] | None = None
    if matching_strategy == "sequential":
        if not reuse_matches:
            print(
                f"Sequentially matching {overlap} quadratic-neighbor levels "
                "at frame offsets 1, 2, 4, ..."
            )
            pairing_options = pycolmap.SequentialPairingOptions(
                overlap=overlap,
                quadratic_overlap=True,
            )
            pycolmap.match_sequential(
                database_path=database_path,
                pairing_options=pairing_options,
                matching_options=matching_options,
                device=pycolmap.Device.cpu,
            )
    elif matching_strategy == "exhaustive":
        if not reuse_matches:
            print("Exhaustively matching every image pair")
            pycolmap.match_exhaustive(
                database_path=database_path,
                matching_options=matching_options,
                device=pycolmap.Device.cpu,
            )
    else:
        vocab_tree_kwargs: dict[str, object] = {
            "num_images": vocab_tree_num_images,
            "num_threads": num_threads,
        }
        if vocab_tree_path is not None:
            vocab_tree_kwargs["vocab_tree_path"] = str(vocab_tree_path)
        pairing_options = pycolmap.VocabTreePairingOptions(**vocab_tree_kwargs)
        pairing_options.max_num_features = int(
            _effective_nested_option(
                extraction_options,
                "sift",
                "max_num_features",
                feature_max_num_features or DEFAULT_FEATURE_MAX_NUM_FEATURES,
            )
        )
        effective_vocab_tree_path = str(
            _effective_option(
                pairing_options,
                "vocab_tree_path",
                str(vocab_tree_path) if vocab_tree_path is not None else "",
            )
        )
        vocab_tree_stats = {
            "num_images": int(
                _effective_option(
                    pairing_options,
                    "num_images",
                    vocab_tree_num_images,
                )
            ),
            "num_nearest_neighbors": int(
                _effective_option(pairing_options, "num_nearest_neighbors", 5)
            ),
            "num_checks": int(
                _effective_option(pairing_options, "num_checks", 64)
            ),
            "num_images_after_verification": int(
                _effective_option(
                    pairing_options,
                    "num_images_after_verification",
                    0,
                )
            ),
            "max_num_features": int(
                _effective_option(
                    pairing_options,
                    "max_num_features",
                    feature_max_num_features
                    or DEFAULT_FEATURE_MAX_NUM_FEATURES,
                )
            ),
            "path": effective_vocab_tree_path,
            "sha256": _vocab_tree_sha256(
                effective_vocab_tree_path,
                vocab_tree_local_file,
            ),
        }
        if not reuse_matches:
            print(
                "Vocabulary-tree matching with "
                f"{vocab_tree_stats['num_images']} retrieved images per query"
            )
            pycolmap.match_vocabtree(
                database_path=database_path,
                pairing_options=pairing_options,
                matching_options=matching_options,
                device=pycolmap.Device.cpu,
            )

    print(f"Running incremental structure-from-motion with threads={num_threads}")
    mapping_options = pycolmap.IncrementalPipelineOptions(
        num_threads=num_threads,
    )
    if init_min_tri_angle is not None:
        mapping_options.mapper.init_min_tri_angle = init_min_tri_angle
    if ba_global_frames_ratio is not None:
        mapping_options.ba_global_frames_ratio = ba_global_frames_ratio
    if ba_global_points_ratio is not None:
        mapping_options.ba_global_points_ratio = ba_global_points_ratio
    if ba_global_max_refinements is not None:
        mapping_options.ba_global_max_refinements = ba_global_max_refinements
    if ba_global_max_num_iterations is not None:
        mapping_options.ba_global_max_num_iterations = (
            ba_global_max_num_iterations
        )
    if snapshot_frames_freq is not None:
        snapshot_path = dataset / "snapshots"
        snapshot_path.mkdir(parents=True, exist_ok=True)
        mapping_options.snapshot_path = str(snapshot_path)
        mapping_options.snapshot_frames_freq = snapshot_frames_freq
        print(
            f"Snapshotting the model to {snapshot_path} every "
            f"{snapshot_frames_freq} registered frames"
        )
    if ba_use_gpu:
        mapping_options.ba_use_gpu = True
        mapping_options.ba_gpu_index = ba_gpu_index
        print(f"Bundle adjustment on GPU {ba_gpu_index}")
    mapping_kwargs: dict[str, object] = {}
    if resume_model_path is not None:
        print(f"Resuming incremental mapping from {resume_model_path}")
        mapping_kwargs["input_path"] = str(resume_model_path)
    _raise_blas_threads_for_mapping(mapping_blas_threads)
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_path,
        options=mapping_options,
        **mapping_kwargs,
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
    passed_global_quality_gate = registered >= minimum_registered
    source_registration: list[dict[str, object]] = []
    if min_source_registration_ratio is not None:
        source_registration = _source_registration_diagnostics(
            reconstruction,
            video_set_sources,
            min_source_registration_ratio,
        )
    passed_source_quality_gate = all(
        source["status"] == "passed" for source in source_registration
    )
    passed_quality_gate = (
        passed_global_quality_gate and passed_source_quality_gate
    )
    stats = {
        "camera_mode": "SINGLE",
        "camera_model": camera_model,
        "feature_extraction": {
            "max_image_size": int(
                _effective_option(
                    extraction_options,
                    "max_image_size",
                    feature_max_image_size or DEFAULT_FEATURE_MAX_IMAGE_SIZE,
                )
            ),
            "max_num_features": int(
                _effective_nested_option(
                    extraction_options,
                    "sift",
                    "max_num_features",
                    feature_max_num_features or DEFAULT_FEATURE_MAX_NUM_FEATURES,
                )
            ),
            "estimate_affine_shape": bool(
                _effective_nested_option(
                    extraction_options,
                    "sift",
                    "estimate_affine_shape",
                    sift_affine_shape,
                )
            ),
            "domain_size_pooling": bool(
                _effective_nested_option(
                    extraction_options,
                    "sift",
                    "domain_size_pooling",
                    sift_domain_size_pooling,
                )
            ),
        },
        "feature_matching": {
            "guided_matching": bool(
                _effective_option(
                    matching_options,
                    "guided_matching",
                    guided_matching,
                )
            ),
        },
        "matching_strategy": matching_strategy,
        "matching_overlap": overlap,
        "matching_quadratic_overlap": matching_strategy == "sequential",
        "minimum_registered_images": minimum_registered,
        "minimum_registration_ratio": min_registration_ratio,
        "mapper": {
            "init_min_tri_angle": float(
                _effective_nested_option(
                    mapping_options,
                    "mapper",
                    "init_min_tri_angle",
                    init_min_tri_angle or DEFAULT_INIT_MIN_TRI_ANGLE,
                )
            ),
            "ba_global_frames_ratio": float(
                _effective_option(
                    mapping_options,
                    "ba_global_frames_ratio",
                    ba_global_frames_ratio or DEFAULT_BA_GLOBAL_FRAMES_RATIO,
                )
            ),
            "ba_global_points_ratio": float(
                _effective_option(
                    mapping_options,
                    "ba_global_points_ratio",
                    ba_global_points_ratio or DEFAULT_BA_GLOBAL_POINTS_RATIO,
                )
            ),
            "ba_global_max_refinements": int(
                _effective_option(
                    mapping_options,
                    "ba_global_max_refinements",
                    ba_global_max_refinements
                    or DEFAULT_BA_GLOBAL_MAX_REFINEMENTS,
                )
            ),
            "ba_global_max_num_iterations": int(
                _effective_option(
                    mapping_options,
                    "ba_global_max_num_iterations",
                    ba_global_max_num_iterations
                    or DEFAULT_BA_GLOBAL_MAX_NUM_ITERATIONS,
                )
            ),
        },
        "ba_use_gpu": ba_use_gpu,
        "reused_matches": reuse_matches,
        "resumed_from_model": (
            str(resume_model_path) if resume_model_path is not None else None
        ),
        "snapshot_frames_freq": snapshot_frames_freq or 0,
        "num_input_images": len(images),
        "num_points3D": points,
        "num_registered_images": registered,
        "registration_ratio": registration_ratio,
        "selected_model_id": model_id,
        "selected_model_path": "sparse/0",
        "status": "passed" if passed_quality_gate else "failed",
    }
    if min_source_registration_ratio is not None:
        stats["minimum_source_registration_ratio"] = (
            min_source_registration_ratio
        )
        stats["source_registration"] = source_registration
    if vocab_tree_stats is not None:
        stats["vocab_tree"] = vocab_tree_stats

    print(reconstruction.summary())
    print(f"Selected largest model: {registered}/{len(images)} registered images")
    print(f"Sparse points: {points}")
    if not passed_global_quality_gate:
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
    if not passed_source_quality_gate:
        stats_path = _write_reconstruction_stats(dataset, stats)
        print(f"Reconstruction stats: {stats_path}")
        failed_sources = ", ".join(
            f"{source['image_prefix']}: "
            f"{source['num_registered_images']}/{source['num_input_images']} "
            f"({source['registration_ratio']:.1%})"
            for source in source_registration
            if source["status"] == "failed"
        )
        raise RuntimeError(
            "COLMAP per-source registration quality gate failed: "
            f"{failed_sources}; need at least "
            f"{min_source_registration_ratio:.1%} from every source. "
            "Inspect cross-video overlap and rerun in a fresh output directory."
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
            args.matching,
            args.min_source_registration_ratio,
            args.vocab_tree_num_images,
            args.vocab_tree,
            args.guided_matching,
            args.sift_affine_shape,
            args.sift_domain_size_pooling,
            args.feature_max_image_size,
            args.feature_max_num_features,
            args.init_min_tri_angle,
            args.reuse_matches,
            args.ba_global_frames_ratio,
            args.ba_global_points_ratio,
            args.ba_global_max_refinements,
            args.ba_global_max_num_iterations,
            args.snapshot_frames_freq,
            args.resume_model,
            args.mapping_blas_threads,
            args.ba_use_gpu,
            args.ba_gpu_index,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
