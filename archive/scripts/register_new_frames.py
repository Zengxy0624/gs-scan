#!/usr/bin/env python3
"""Register bootstrap candidate frames into an existing COLMAP reconstruction.

Builds an augmented copy of a fully reconstructed base dataset, extracts
features for only the new candidate images with the base dataset's recorded
extraction options, vocabulary-tree matches only the new images against the
copied database, and resumes incremental mapping from the base sparse model.
The base dataset is never written to.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

# Archived under archive/: put the repo root on sys.path so the shared
# helpers that stayed in scripts/ resolve under direct execution too.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.merge_frame_datasets import _link_or_copy  # noqa: E402
from scripts.prepare_colmap import (  # noqa: E402
    _effective_nested_option,
    _effective_option,
    _file_identity,
    _raise_blas_threads_for_mapping,
    _write_reconstruction_stats,
)


DEFAULT_VOCAB_TREE_NUM_IMAGES = 50
DEFAULT_MAPPING_BLAS_THREADS = 48
DEFAULT_MIN_CANDIDATE_REGISTRATION_RATIO = 0.8
CANDIDATE_NAMES_FILE = "candidate_names.txt"
MATCH_LIST_NAME = "bootstrap_match_list.txt"
MODEL_FILE_NAMES = ("cameras.bin", "images.bin", "points3D.bin")
FEATURE_EXTRACTION_FIELDS = (
    "max_image_size",
    "max_num_features",
    "estimate_affine_shape",
    "domain_size_pooling",
)
MAPPER_BA_FIELDS = (
    "ba_global_frames_ratio",
    "ba_global_points_ratio",
    "ba_global_max_refinements",
    "ba_global_max_num_iterations",
)
MATCHING_OMP_THREADS = 32
OPENBLAS_SHIM_BASENAME = "libopenblas-pthread.so.0"
OPENBLAS_SHIM_RELATIVE = Path("third-party") / "openblas-threaded"


def _enforce_matching_thread_env() -> None:
    """Verify the launch environment vocabulary-tree matching requires.

    faiss (inside pycolmap) calls OpenBLAS sgemm from every OpenMP worker at
    once. Two crash modes occur without these settings: an unpinned OpenBLAS
    races inside the workers (heap corruption), and more than 64 registered
    BLAS threads overflow the preloaded shim's thread-metadata slots (SIGSEGV
    in sgemm_incopy). See third-party/openblas-threaded/README.md.

    All three settings must be present WHEN THE PROCESS LAUNCHES: LD_PRELOAD
    is only honored by the loader, and the preloaded shim sizes its thread
    pool from OPENBLAS_NUM_THREADS in its load-time constructor, before any
    Python code runs. Setting them from inside the process is too late, so a
    wrong launch environment is a hard error rather than a silent fix.
    """
    shim = OPENBLAS_SHIM_RELATIVE / OPENBLAS_SHIM_BASENAME
    relaunch = (
        f"relaunch as: LD_PRELOAD={shim} "
        f"LD_LIBRARY_PATH={OPENBLAS_SHIM_RELATIVE} OPENBLAS_NUM_THREADS=1 "
        f"OMP_NUM_THREADS={MATCHING_OMP_THREADS} python "
        "archive/scripts/register_new_frames.py ... (all on one line; these must be "
        "set at process launch)"
    )
    if OPENBLAS_SHIM_BASENAME not in os.environ.get("LD_PRELOAD", ""):
        raise RuntimeError(
            "vocabulary-tree matching needs the pthread OpenBLAS shim "
            f"preloaded; {relaunch}"
        )
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise RuntimeError(
            "OPENBLAS_NUM_THREADS=1 must be set at launch (the preloaded "
            f"shim reads it before Python starts); {relaunch}"
        )
    omp_threads = os.environ.get("OMP_NUM_THREADS", "")
    if not omp_threads.isdigit() or not 1 <= int(omp_threads) <= (
        MATCHING_OMP_THREADS
    ):
        raise RuntimeError(
            f"OMP_NUM_THREADS must be set at launch to at most "
            f"{MATCHING_OMP_THREADS} (more concurrent OpenMP callers "
            f"overflow the shim's 64 thread-metadata slots); {relaunch}"
        )


def _pin_runtime_threads_for_matching() -> None:
    """Cap already-initialized OpenMP/OpenBLAS pools before matching runs."""
    import ctypes

    for library, symbol, value in (
        ("libgomp.so.1", "omp_set_num_threads", MATCHING_OMP_THREADS),
        ("libopenblas.so.0", "openblas_set_num_threads", 1),
    ):
        try:
            getattr(ctypes.CDLL(library), symbol)(value)
        except (AttributeError, OSError):
            continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register bootstrap candidate frames into an existing COLMAP "
            "model by resuming incremental mapping on an augmented copy of "
            "the base dataset."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help=(
            "Fully reconstructed base dataset (images/, database.db, "
            "sparse/0, reconstruction_stats.json with status passed)."
        ),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help=(
            "bootstrap_candidates dataset (images/, candidate_names.txt, "
            "metadata.json)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh augmented dataset path; must not exist yet.",
    )
    parser.add_argument(
        "--vocab-tree",
        type=Path,
        required=True,
        help="Local COLMAP vocabulary-tree file used to pair the new images.",
    )
    parser.add_argument(
        "--vocab-tree-num-images",
        type=int,
        default=DEFAULT_VOCAB_TREE_NUM_IMAGES,
        help=(
            "Retrieve this many visually similar images per candidate "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--guided-matching",
        action="store_true",
        help="Enable guided feature matching for higher recall at extra cost.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=-1,
        help=(
            "Maximum CPU threads for feature extraction, matching, and "
            "mapping; -1 lets COLMAP use all available threads (default: -1)."
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
        default=DEFAULT_MAPPING_BLAS_THREADS,
        help=(
            "Raise the interposed OpenBLAS to this many threads for the "
            "mapping stage (default: %(default)s; 0 disables)."
        ),
    )
    parser.add_argument(
        "--min-candidate-registration-ratio",
        type=float,
        default=DEFAULT_MIN_CANDIDATE_REGISTRATION_RATIO,
        help=(
            "Fail unless at least this fraction of the candidate images "
            "registers into the resumed model (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--ba-global-frames-ratio",
        type=float,
        help=(
            "Override the base reconstruction's recorded global "
            "bundle-adjustment frame-growth trigger."
        ),
    )
    parser.add_argument(
        "--ba-global-points-ratio",
        type=float,
        help=(
            "Override the base reconstruction's recorded global "
            "bundle-adjustment point-growth trigger."
        ),
    )
    parser.add_argument(
        "--ba-global-max-refinements",
        type=int,
        help=(
            "Override the base reconstruction's recorded maximum refinement "
            "cycles per global bundle adjustment."
        ),
    )
    parser.add_argument(
        "--ba-global-max-num-iterations",
        type=int,
        help=(
            "Override the base reconstruction's recorded maximum solver "
            "iterations per global bundle adjustment."
        ),
    )
    return parser


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return data


def _validate_base(
    base: Path,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    """Check the base dataset is a complete, passed reconstruction."""
    image_dir = base / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Base image directory not found: {image_dir}")
    image_names = sorted(
        path.name for path in image_dir.iterdir() if path.is_file()
    )
    if not image_names:
        raise ValueError(f"Base image directory is empty: {image_dir}")

    database_path = base / "database.db"
    if not database_path.is_file() or database_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Base COLMAP database is missing or empty: {database_path}"
        )

    model_dir = base / "sparse" / "0"
    missing_model_files = [
        name for name in MODEL_FILE_NAMES if not (model_dir / name).is_file()
    ]
    if missing_model_files:
        raise FileNotFoundError(
            f"Base sparse model is incomplete: {model_dir} "
            f"(missing {', '.join(missing_model_files)})"
        )

    stats_path = base / "reconstruction_stats.json"
    stats = _read_json_object(stats_path, "Base reconstruction stats")
    if stats.get("status") != "passed":
        raise ValueError(
            "Base reconstruction did not pass its quality gates "
            f"(status {stats.get('status')!r}): {stats_path}. "
            "Rebuild the base model before registering new frames."
        )
    metadata = _read_json_object(base / "metadata.json", "Base dataset metadata")
    return stats, metadata, image_names


def _adopt_base_options(
    stats: dict[str, object],
    base: Path,
) -> tuple[str, dict[str, object], dict[str, object], int]:
    """Adopt the recorded extraction and mapper options from the base stats."""
    stats_path = base / "reconstruction_stats.json"
    camera_model = stats.get("camera_model")
    if not isinstance(camera_model, str) or not camera_model:
        raise ValueError(f"Base stats record no camera_model: {stats_path}")

    features = stats.get("feature_extraction")
    if not isinstance(features, dict):
        raise ValueError(
            f"Base stats lack a feature_extraction section: {stats_path}"
        )
    missing_fields = [
        field for field in FEATURE_EXTRACTION_FIELDS if field not in features
    ]
    if missing_fields:
        raise ValueError(
            "Base feature_extraction section is missing "
            f"{', '.join(missing_fields)}: {stats_path}"
        )

    mapper = stats.get("mapper")
    if not isinstance(mapper, dict):
        raise ValueError(f"Base stats lack a mapper section: {stats_path}")
    missing_fields = [
        field for field in MAPPER_BA_FIELDS if field not in mapper
    ]
    if missing_fields:
        raise ValueError(
            "Base mapper section is missing "
            f"{', '.join(missing_fields)}: {stats_path}"
        )

    registered = stats.get("num_registered_images")
    if not isinstance(registered, int) or registered < 1:
        raise ValueError(
            f"Base stats record no registered image count: {stats_path}"
        )
    return camera_model, features, mapper, registered


def _validate_candidates(
    candidates: Path,
    base_image_names: Sequence[str],
) -> tuple[dict[str, object], list[str]]:
    """Check the candidate frames and refuse base name collisions."""
    metadata = _read_json_object(
        candidates / "metadata.json", "Candidate metadata"
    )
    if metadata.get("dataset_type") != "bootstrap_candidates":
        raise ValueError(
            f"Expected a bootstrap_candidates dataset: {candidates}"
        )

    names_path = candidates / CANDIDATE_NAMES_FILE
    if not names_path.is_file():
        raise FileNotFoundError(f"Candidate name list not found: {names_path}")
    names = [
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"Candidate name list is empty: {names_path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Candidate name list has duplicates: {names_path}")

    image_dir = candidates / "images"
    base_names = set(base_image_names)
    for name in names:
        if "/" in name or "\\" in name:
            raise ValueError(
                f"Candidate names must be bare file names: {name!r}"
            )
        if not (image_dir / name).is_file():
            raise FileNotFoundError(
                f"Candidate image not found: {image_dir / name}"
            )
        if name in base_names:
            raise ValueError(
                f"Candidate name collides with a base image: {name}. "
                "Bootstrap frames must use the <prefix>m<index>.jpg naming."
            )
    return metadata, names


def register_new_frames(
    base: Path,
    candidates: Path,
    output: Path,
    vocab_tree: Path,
    vocab_tree_num_images: int = DEFAULT_VOCAB_TREE_NUM_IMAGES,
    guided_matching: bool = False,
    num_threads: int = -1,
    ba_use_gpu: bool = False,
    ba_gpu_index: str = "0",
    mapping_blas_threads: int = DEFAULT_MAPPING_BLAS_THREADS,
    min_candidate_registration_ratio: float = (
        DEFAULT_MIN_CANDIDATE_REGISTRATION_RATIO
    ),
    ba_global_frames_ratio: float | None = None,
    ba_global_points_ratio: float | None = None,
    ba_global_max_refinements: int | None = None,
    ba_global_max_num_iterations: int | None = None,
) -> Path:
    if num_threads == 0 or num_threads < -1:
        raise ValueError("--num-threads must be -1 or a positive integer")
    if vocab_tree_num_images < 1:
        raise ValueError("--vocab-tree-num-images must be positive")
    if mapping_blas_threads < 0:
        raise ValueError("--mapping-blas-threads must be 0 or positive")
    if not 0.0 < min_candidate_registration_ratio <= 1.0:
        raise ValueError(
            "--min-candidate-registration-ratio must be greater than 0 and "
            "at most 1"
        )
    for option_name, ratio in (
        ("--ba-global-frames-ratio", ba_global_frames_ratio),
        ("--ba-global-points-ratio", ba_global_points_ratio),
    ):
        if ratio is not None and (not math.isfinite(ratio) or ratio <= 1.0):
            raise ValueError(f"{option_name} must be finite and greater than 1")
    for option_name, count in (
        ("--ba-global-max-refinements", ba_global_max_refinements),
        ("--ba-global-max-num-iterations", ba_global_max_num_iterations),
    ):
        if count is not None and count < 1:
            raise ValueError(f"{option_name} must be a positive integer")

    vocab_tree_path = vocab_tree.expanduser().resolve()
    if not vocab_tree_path.is_file():
        raise FileNotFoundError(
            f"Vocabulary-tree file not found: {vocab_tree_path}"
        )
    vocab_tree_identity = _file_identity(vocab_tree_path)

    base = base.expanduser().resolve()
    candidates = candidates.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output dataset already exists: {output}")
    for source_root in (base, candidates):
        if output == source_root or output.is_relative_to(source_root):
            raise ValueError(
                "Output dataset must not be inside the base or candidates "
                "dataset"
            )

    base_stats, base_metadata, base_image_names = _validate_base(base)
    camera_model, feature_section, mapper_section, base_registered = (
        _adopt_base_options(base_stats, base)
    )
    candidate_metadata, candidate_names = _validate_candidates(
        candidates, base_image_names
    )
    _enforce_matching_thread_env()

    feature_max_image_size = int(feature_section["max_image_size"])  # type: ignore[index]
    feature_max_num_features = int(feature_section["max_num_features"])  # type: ignore[index]
    sift_affine_shape = bool(feature_section["estimate_affine_shape"])  # type: ignore[index]
    sift_domain_size_pooling = bool(feature_section["domain_size_pooling"])  # type: ignore[index]
    effective_ba_frames_ratio = (
        ba_global_frames_ratio
        if ba_global_frames_ratio is not None
        else float(mapper_section["ba_global_frames_ratio"])  # type: ignore[index]
    )
    effective_ba_points_ratio = (
        ba_global_points_ratio
        if ba_global_points_ratio is not None
        else float(mapper_section["ba_global_points_ratio"])  # type: ignore[index]
    )
    effective_ba_max_refinements = (
        ba_global_max_refinements
        if ba_global_max_refinements is not None
        else int(mapper_section["ba_global_max_refinements"])  # type: ignore[index]
    )
    effective_ba_max_num_iterations = (
        ba_global_max_num_iterations
        if ba_global_max_num_iterations is not None
        else int(mapper_section["ba_global_max_num_iterations"])  # type: ignore[index]
    )

    base_image_dir = base / "images"
    base_database = base / "database.db"
    base_model_dir = base / "sparse" / "0"
    candidate_image_dir = candidates / "images"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        staged_images = temporary / "images"
        staged_images.mkdir()
        print(
            f"Linking {len(base_image_names)} base images and "
            f"{len(candidate_names)} candidate images"
        )
        for name in base_image_names:
            _link_or_copy(base_image_dir / name, staged_images / name, "auto")
        for name in candidate_names:
            _link_or_copy(
                candidate_image_dir / name, staged_images / name, "auto"
            )

        database_size = base_database.stat().st_size
        print(
            f"Copying base COLMAP database ({database_size / 1e9:.1f} GB) "
            f"from {base_database}"
        )
        shutil.copyfile(base_database, temporary / "database.db")

        resume_model = temporary / "resume_model"
        resume_model.mkdir()
        for name in MODEL_FILE_NAMES:
            shutil.copyfile(base_model_dir / name, resume_model / name)

        shutil.copyfile(
            candidates / CANDIDATE_NAMES_FILE,
            temporary / CANDIDATE_NAMES_FILE,
        )
        (temporary / MATCH_LIST_NAME).write_text(
            "\n".join(candidate_names) + "\n", encoding="utf-8"
        )

        augmented_metadata = {
            **base_metadata,
            "bootstrap": {
                "candidates_dataset": str(candidates),
                "candidates_instance_id": candidate_metadata.get(
                    "dataset_instance_id"
                ),
                "num_candidates": len(candidate_names),
                "names_file": CANDIDATE_NAMES_FILE,
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(augmented_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Staged augmented dataset: {output}")

    import pycolmap

    _pin_runtime_threads_for_matching()
    print(f"PyCOLMAP {pycolmap.__version__}")
    database_path = output / "database.db"
    image_dir = output / "images"
    resume_model_path = output / "resume_model"
    match_list_path = output / MATCH_LIST_NAME
    num_input_images = len(base_image_names) + len(candidate_names)

    base_model = pycolmap.Reconstruction(str(resume_model_path))
    camera_ids = sorted(base_model.cameras)
    if len(camera_ids) != 1:
        raise RuntimeError(
            f"Base model must have exactly one camera; found "
            f"{len(camera_ids)} in {resume_model_path}"
        )
    existing_camera_id = camera_ids[0]

    reader_options = pycolmap.ImageReaderOptions(camera_model=camera_model)
    reader_options.existing_camera_id = existing_camera_id
    extraction_options = pycolmap.FeatureExtractionOptions(
        num_threads=num_threads
    )
    extraction_options.max_image_size = feature_max_image_size
    extraction_options.sift.max_num_features = feature_max_num_features
    extraction_options.sift.estimate_affine_shape = sift_affine_shape
    extraction_options.sift.domain_size_pooling = sift_domain_size_pooling
    print(
        f"Extracting features for {len(candidate_names)} new images on "
        f"existing camera {existing_camera_id}"
    )
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=camera_model,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cpu,
    )

    database = pycolmap.Database.open(str(database_path))
    try:
        num_cameras = database.num_cameras()
    finally:
        database.close()
    if num_cameras != 1:
        raise RuntimeError(
            f"Feature extraction left {num_cameras} cameras in "
            f"{database_path}; new frames must attach to the base model's "
            "single camera or the shared calibration would be corrupted."
        )

    pairing_options = pycolmap.VocabTreePairingOptions(
        num_images=vocab_tree_num_images,
        num_threads=num_threads,
        vocab_tree_path=str(vocab_tree_path),
    )
    pairing_options.match_list_path = str(match_list_path)
    pairing_options.max_num_features = feature_max_num_features
    matching_options = pycolmap.FeatureMatchingOptions(
        num_threads=num_threads,
        use_gpu=False,
    )
    matching_options.guided_matching = guided_matching
    print(
        f"Vocabulary-tree matching {len(candidate_names)} new images with "
        f"{vocab_tree_num_images} retrieved images per query"
    )
    pycolmap.match_vocabtree(
        database_path=database_path,
        pairing_options=pairing_options,
        matching_options=matching_options,
        device=pycolmap.Device.cpu,
    )

    sparse_path = output / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)
    mapping_options = pycolmap.IncrementalPipelineOptions(
        num_threads=num_threads,
    )
    mapping_options.ba_global_frames_ratio = effective_ba_frames_ratio
    mapping_options.ba_global_points_ratio = effective_ba_points_ratio
    mapping_options.ba_global_max_refinements = effective_ba_max_refinements
    mapping_options.ba_global_max_num_iterations = (
        effective_ba_max_num_iterations
    )
    if ba_use_gpu:
        mapping_options.ba_use_gpu = True
        mapping_options.ba_gpu_index = ba_gpu_index
        print(f"Bundle adjustment on GPU {ba_gpu_index}")
    print(f"Resuming incremental mapping from {resume_model_path}")
    _raise_blas_threads_for_mapping(mapping_blas_threads)
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_path,
        options=mapping_options,
        input_path=str(resume_model_path),
    )
    if not reconstructions:
        raise RuntimeError(
            "COLMAP did not recover a model while resuming from "
            f"{resume_model_path}. Inspect the candidate frames and rerun in "
            "a fresh output directory."
        )

    model_id, reconstruction = max(
        reconstructions.items(),
        key=lambda item: item[1].num_reg_images(),
    )
    registered = reconstruction.num_reg_images()
    points = reconstruction.num_points3D()
    images_by_id = getattr(reconstruction, "images")
    registered_names = {
        str(images_by_id[image_id].name)
        for image_id in reconstruction.reg_image_ids()
    }
    num_candidate_registered = sum(
        name in registered_names for name in candidate_names
    )
    candidate_ratio = num_candidate_registered / len(candidate_names)
    passed_candidate_gate = (
        candidate_ratio >= min_candidate_registration_ratio
    )
    passed_base_gate = registered >= base_registered

    stats: dict[str, object] = {
        "base": {
            "dataset": str(base),
            "num_registered_images": base_registered,
            "num_points3D": base_stats.get("num_points3D"),
            "selected_model_files": base_stats.get("selected_model_files"),
        },
        "camera_mode": "SINGLE",
        "camera_model": camera_model,
        "feature_extraction": {
            "max_image_size": int(
                _effective_option(
                    extraction_options,
                    "max_image_size",
                    feature_max_image_size,
                )
            ),
            "max_num_features": int(
                _effective_nested_option(
                    extraction_options,
                    "sift",
                    "max_num_features",
                    feature_max_num_features,
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
        "matching_strategy": "vocab_tree",
        "mapper": {
            "ba_global_frames_ratio": float(
                _effective_option(
                    mapping_options,
                    "ba_global_frames_ratio",
                    effective_ba_frames_ratio,
                )
            ),
            "ba_global_points_ratio": float(
                _effective_option(
                    mapping_options,
                    "ba_global_points_ratio",
                    effective_ba_points_ratio,
                )
            ),
            "ba_global_max_refinements": int(
                _effective_option(
                    mapping_options,
                    "ba_global_max_refinements",
                    effective_ba_max_refinements,
                )
            ),
            "ba_global_max_num_iterations": int(
                _effective_option(
                    mapping_options,
                    "ba_global_max_num_iterations",
                    effective_ba_max_num_iterations,
                )
            ),
        },
        "ba_use_gpu": ba_use_gpu,
        "bootstrap": {
            "num_candidates": len(candidate_names),
            "num_candidate_registered": num_candidate_registered,
            "candidate_registration_ratio": candidate_ratio,
            "min_candidate_registration_ratio": (
                min_candidate_registration_ratio
            ),
            "match_list": MATCH_LIST_NAME,
            "vocab_tree": {
                "path": str(vocab_tree_path),
                "sha256": vocab_tree_identity["sha256"],
            },
        },
        "num_input_images": num_input_images,
        "num_points3D": points,
        "num_registered_images": registered,
        "registration_ratio": registered / num_input_images,
        "selected_model_id": model_id,
        "selected_model_path": "sparse/0",
        "status": (
            "passed"
            if passed_candidate_gate and passed_base_gate
            else "failed"
        ),
    }

    print(reconstruction.summary())
    print(
        f"Candidate registration: {num_candidate_registered}/"
        f"{len(candidate_names)} ({candidate_ratio:.1%})"
    )
    print(f"Total registered images: {registered}/{num_input_images}")
    if not passed_candidate_gate or not passed_base_gate:
        stats_path = _write_reconstruction_stats(output, stats)
        print(f"Reconstruction stats: {stats_path}")
        failures = []
        if not passed_candidate_gate:
            failures.append(
                f"only {num_candidate_registered}/{len(candidate_names)} "
                f"candidate images registered ({candidate_ratio:.1%}); need "
                f"at least {min_candidate_registration_ratio:.1%}"
            )
        if not passed_base_gate:
            failures.append(
                f"resumed model registers {registered} images but the base "
                f"model registered {base_registered}"
            )
        raise RuntimeError(
            "Bootstrap registration quality gate failed: "
            + "; ".join(failures)
            + ". Sample candidates nearer well-registered views and rerun "
            "in a fresh output directory."
        )

    model_path = sparse_path / "0"
    model_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(model_path)
    required_model_files = tuple(
        model_path / name for name in MODEL_FILE_NAMES
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

    shutil.rmtree(resume_model_path)

    # A passed stats file is the final, atomic completion marker. Never write
    # it before the selected model is durably present and structurally
    # complete.
    stats_path = _write_reconstruction_stats(output, stats)
    print(f"Augmented COLMAP model: {model_path}")
    print(f"Reconstruction stats: {stats_path}")
    return model_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        register_new_frames(
            args.base,
            args.candidates,
            args.output,
            args.vocab_tree,
            args.vocab_tree_num_images,
            args.guided_matching,
            args.num_threads,
            args.ba_use_gpu,
            args.ba_gpu_index,
            args.mapping_blas_threads,
            args.min_candidate_registration_ratio,
            args.ba_global_frames_ratio,
            args.ba_global_points_ratio,
            args.ba_global_max_refinements,
            args.ba_global_max_num_iterations,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
