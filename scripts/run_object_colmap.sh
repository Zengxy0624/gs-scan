#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"
PROJECT_PYTHON="${PROJECT_PYTHON:-${GSPLAT_ROOT}/.venv/bin/python}"
# Interpreter candidates, most capable first. Override with --colmap-python.
GPU_COLMAP_PYTHON="${GPU_COLMAP_PYTHON:-${PROJECT_ROOT}/work/_gpu_colmap/mamba_root/envs/gpubuild/bin/python}"
# Written by scripts/build_gpu_colmap.sh only after pycolmap verified; the
# interpreter alone exists as soon as the conda env is created.
GPU_COLMAP_READY_MARKER="${GPU_COLMAP_READY_MARKER:-${PROJECT_ROOT}/work/_gpu_colmap/BUILD_DONE}"
CUDA_COLMAP_PYTHON="${CUDA_COLMAP_PYTHON:-${PROJECT_ROOT}/work/_colmap_cuda/venv/bin/python}"
DEFAULT_VOCAB_TREE="${PROJECT_ROOT}/work/_colmap_cuda/vocab_tree_faiss_flickr100K_words256K.bin"

DATASET_ROOT=""
GPU_ID="2"
CAMERA_MODEL="OPENCV"
COLMAP_PYTHON=""
FEATURE_GPU="auto"
BA_USE_GPU="auto"
BLAS_SHIM="auto"
VOCAB_TREE_PATH=""
VOCAB_TREE_NUM_IMAGES="200"
COLMAP_THREADS="32"
# Concurrent BLAS callers (COLMAP worker threads + OpenMP workers) must stay
# under the shim's MAX_THREADS=64: 256 retrieval threads + 64 OpenMP workers
# segfaulted inside sgemm on 2026-09-01 (pycolmap 4.2 wheel, 256-core host).
SHIM_MAX_THREADS="32"
MIN_REGISTRATION_RATIO="0.8"
MIN_SOURCE_REGISTRATION_RATIO="0.8"
FEATURE_MAX_IMAGE_SIZE="3840"
FEATURE_MAX_NUM_FEATURES="8192"
INIT_MIN_TRI_ANGLE="8"
# Passed explicitly so pycolmap 4.x runs with the same schedule the pycolmap
# 3.13 production reconstructions used (4.x changed some defaults).
BA_GLOBAL_FRAMES_RATIO="1.1"
BA_GLOBAL_POINTS_RATIO="1.1"
BA_GLOBAL_MAX_REFINEMENTS="5"
BA_GLOBAL_MAX_NUM_ITERATIONS="50"
SNAPSHOT_FRAMES_FREQ="250"
MAPPING_BLAS_THREADS=""
REUSE_MATCHES=0
RESUME_FROM_SNAPSHOT=0
RESUME_MODEL_DIR=""
RETRY_INCOMPLETE=0
DRY_RUN=0

usage() {
    cat <<'USAGE'
Reconstruct one merged frame dataset (from prepare_object_frames.sh or
merge_frame_datasets.py) with COLMAP using the production quality recipe:
vocabulary-tree matching, guided matching, affine/DSP-SIFT at full 3840 px,
one shared camera (EXIF 35 mm-equivalent focal length as prior when present),
80% global and per-video registration gates, model snapshots for resume.
Picks the most capable pycolmap available: the cuDSS build (GPU features +
GPU bundle adjustment), else the CUDA wheel venv (GPU features), else the
gsplat venv (CPU). See docs/OBJECT_PIPELINE.md.

Usage:
  bash scripts/run_object_colmap.sh --dataset DIR [options]

Options:
  --dataset DIR              Merged dataset directory (required).
  --gpu ID                   Physical GPU for feature extraction/matching (and
                             bundle adjustment when available) (default: 2).
  --camera-model MODEL       Shared camera model: OPENCV (default) or
                             OPENCV_FISHEYE for uncorrected fisheye footage.
  --colmap-python PATH       Interpreter running prepare_colmap.py. Default: the
                             cuDSS build under work/_gpu_colmap if present, else
                             the CUDA wheel venv under work/_colmap_cuda, else the
                             gsplat venv (CPU only).
  --feature-gpu auto|on|off  SIFT extraction/matching on the GPU (auto: on when
                             the interpreter has CUDA) (default: auto).
  --ba-use-gpu auto|on|off   Bundle adjustment on the GPU (auto: on only for the
                             cuDSS build; the CUDA wheel lacks cuDSS and would
                             silently fall back to CPU) (default: auto).
  --blas-shim auto|on|off    LD_PRELOAD the pthread OpenBLAS shim (auto: off for
                             the cuDSS build, on for pip wheels, whose serial
                             BLAS crashes under faiss retrieval) (default: auto).
  --vocab-tree PATH          Local vocabulary tree (default: the file under
                             work/_colmap_cuda when present).
  --vocab-tree-num-images N  Retrieved visual neighbors per image (default: 200).
  --colmap-threads N         CPU threads for extraction, matching, and mapping
                             (default: 32; capped at 32 while the BLAS shim is
                             on, because more concurrent BLAS callers overflow
                             the shim's 64-thread build limit and segfault).
  --min-registration-ratio R Overall registration gate (default: 0.8).
  --min-source-registration-ratio R
                             Per-video registration gate (default: 0.8).
  --feature-max-image-size N Feature extraction dimension (default: 3840).
  --feature-max-num-features N
                             SIFT feature cap per image (default: 8192).
  --init-min-tri-angle D     Initial-pair angle in degrees (default: 8).
  --ba-global-frames-ratio R, --ba-global-points-ratio R,
  --ba-global-max-refinements N, --ba-global-max-num-iterations N
                             Global bundle-adjustment schedule (default: the
                             production values 1.1 / 1.1 / 5 / 50, passed
                             explicitly so pycolmap 4.x matches 3.13).
  --snapshot-frames-freq N   Model snapshot every N registered frames; 0 disables
                             (default: 250).
  --mapping-blas-threads N   BLAS threads for mapping under the shim (preparer
                             default: 48).
  --reuse-matches            Map from the existing database.db (same feature and
                             matching settings); stale mapping output is removed.
  --resume-from-snapshot     Continue mapping from the newest snapshot under
                             <dataset>/snapshots. Implies --reuse-matches.
  --retry-incomplete         Archive stale COLMAP state (database, sparse,
                             snapshots, stats) under <parent>/_failed and rebuild;
                             images and metadata are never touched.
  --dry-run                  Print the resolved recipe and commands without writes.
  -h, --help                 Show this help.

Environment overrides:
  PROJECT_PYTHON       Python from the prepared gsplat environment.
  GPU_COLMAP_PYTHON    Path of the cuDSS-capable interpreter (auto-detected;
                       used only once GPU_COLMAP_READY_MARKER exists, default
                       work/_gpu_colmap/BUILD_DONE).
  CUDA_COLMAP_PYTHON   Path of the CUDA-wheel interpreter (auto-detected).
USAGE
}

require_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "${value}" ]]; then
        echo "error: ${option} requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) require_value "$1" "${2:-}"; DATASET_ROOT="$2"; shift 2 ;;
        --gpu) require_value "$1" "${2:-}"; GPU_ID="$2"; shift 2 ;;
        --camera-model) require_value "$1" "${2:-}"; CAMERA_MODEL="$2"; shift 2 ;;
        --colmap-python) require_value "$1" "${2:-}"; COLMAP_PYTHON="$2"; shift 2 ;;
        --feature-gpu) require_value "$1" "${2:-}"; FEATURE_GPU="$2"; shift 2 ;;
        --ba-use-gpu) require_value "$1" "${2:-}"; BA_USE_GPU="$2"; shift 2 ;;
        --blas-shim) require_value "$1" "${2:-}"; BLAS_SHIM="$2"; shift 2 ;;
        --vocab-tree) require_value "$1" "${2:-}"; VOCAB_TREE_PATH="$2"; shift 2 ;;
        --vocab-tree-num-images) require_value "$1" "${2:-}"; VOCAB_TREE_NUM_IMAGES="$2"; shift 2 ;;
        --colmap-threads) require_value "$1" "${2:-}"; COLMAP_THREADS="$2"; shift 2 ;;
        --min-registration-ratio) require_value "$1" "${2:-}"; MIN_REGISTRATION_RATIO="$2"; shift 2 ;;
        --min-source-registration-ratio) require_value "$1" "${2:-}"; MIN_SOURCE_REGISTRATION_RATIO="$2"; shift 2 ;;
        --feature-max-image-size) require_value "$1" "${2:-}"; FEATURE_MAX_IMAGE_SIZE="$2"; shift 2 ;;
        --feature-max-num-features) require_value "$1" "${2:-}"; FEATURE_MAX_NUM_FEATURES="$2"; shift 2 ;;
        --init-min-tri-angle) require_value "$1" "${2:-}"; INIT_MIN_TRI_ANGLE="$2"; shift 2 ;;
        --ba-global-frames-ratio) require_value "$1" "${2:-}"; BA_GLOBAL_FRAMES_RATIO="$2"; shift 2 ;;
        --ba-global-points-ratio) require_value "$1" "${2:-}"; BA_GLOBAL_POINTS_RATIO="$2"; shift 2 ;;
        --ba-global-max-refinements) require_value "$1" "${2:-}"; BA_GLOBAL_MAX_REFINEMENTS="$2"; shift 2 ;;
        --ba-global-max-num-iterations) require_value "$1" "${2:-}"; BA_GLOBAL_MAX_NUM_ITERATIONS="$2"; shift 2 ;;
        --snapshot-frames-freq) require_value "$1" "${2:-}"; SNAPSHOT_FRAMES_FREQ="$2"; shift 2 ;;
        --mapping-blas-threads) require_value "$1" "${2:-}"; MAPPING_BLAS_THREADS="$2"; shift 2 ;;
        --reuse-matches) REUSE_MATCHES=1; shift ;;
        --resume-from-snapshot) RESUME_FROM_SNAPSHOT=1; REUSE_MATCHES=1; shift ;;
        --retry-incomplete) RETRY_INCOMPLETE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

positive_integer() {
    if [[ ! "$1" =~ ^[0-9]+$ ]] || (( 10#$1 < 1 )); then
        echo "error: $2 must be a positive integer" >&2
        exit 2
    fi
}
nonnegative_integer() {
    if [[ ! "$1" =~ ^[0-9]+$ ]]; then
        echo "error: $2 must be a non-negative integer" >&2
        exit 2
    fi
}
positive_decimal_at_most_one() {
    if [[ ! "$1" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || [[ "$1" =~ ^0([.]0+)?$ ]]; then
        echo "error: $2 must be greater than 0 and at most 1" >&2
        exit 2
    fi
}
positive_decimal() {
    if [[ ! "$1" =~ ^([0-9]+)([.][0-9]+)?$ ]] || [[ "$1" =~ ^0+([.]0+)?$ ]]; then
        echo "error: $2 must be positive" >&2
        exit 2
    fi
}
tri_state() {
    case "$1" in
        auto|on|off) ;;
        *) echo "error: $2 must be auto, on, or off" >&2; exit 2 ;;
    esac
}

if [[ -z "${DATASET_ROOT}" ]]; then
    echo "error: --dataset is required" >&2; usage >&2; exit 2
fi
nonnegative_integer "${GPU_ID}" "--gpu"
GPU_ID="$((10#${GPU_ID}))"
case "${CAMERA_MODEL}" in
    OPENCV|OPENCV_FISHEYE|SIMPLE_RADIAL|RADIAL|PINHOLE|SIMPLE_PINHOLE) ;;
    *) echo "error: --camera-model is not a supported COLMAP model" >&2; exit 2 ;;
esac
tri_state "${FEATURE_GPU}" "--feature-gpu"
tri_state "${BA_USE_GPU}" "--ba-use-gpu"
tri_state "${BLAS_SHIM}" "--blas-shim"
positive_integer "${VOCAB_TREE_NUM_IMAGES}" "--vocab-tree-num-images"
positive_integer "${FEATURE_MAX_IMAGE_SIZE}" "--feature-max-image-size"
positive_integer "${FEATURE_MAX_NUM_FEATURES}" "--feature-max-num-features"
if [[ "${COLMAP_THREADS}" != "-1" ]]; then positive_integer "${COLMAP_THREADS}" "--colmap-threads"; fi
positive_decimal_at_most_one "${MIN_REGISTRATION_RATIO}" "--min-registration-ratio"
positive_decimal_at_most_one "${MIN_SOURCE_REGISTRATION_RATIO}" "--min-source-registration-ratio"
positive_decimal "${INIT_MIN_TRI_ANGLE}" "--init-min-tri-angle"
positive_decimal "${BA_GLOBAL_FRAMES_RATIO}" "--ba-global-frames-ratio"
positive_decimal "${BA_GLOBAL_POINTS_RATIO}" "--ba-global-points-ratio"
positive_integer "${BA_GLOBAL_MAX_REFINEMENTS}" "--ba-global-max-refinements"
positive_integer "${BA_GLOBAL_MAX_NUM_ITERATIONS}" "--ba-global-max-num-iterations"
nonnegative_integer "${SNAPSHOT_FRAMES_FREQ}" "--snapshot-frames-freq"
if [[ -n "${MAPPING_BLAS_THREADS}" && "${MAPPING_BLAS_THREADS}" != "0" ]]; then
    positive_integer "${MAPPING_BLAS_THREADS}" "--mapping-blas-threads"
fi

DATASET_ROOT="$(realpath -m -- "${DATASET_ROOT}")"
DATASET_NAME="$(basename -- "${DATASET_ROOT}")"
DATASET_PARENT="$(dirname -- "${DATASET_ROOT}")"
COLMAP_PREPARER="${PROJECT_ROOT}/scripts/prepare_colmap.py"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_dataset_cache.py"
for required_file in "${COLMAP_PREPARER}" "${VALIDATOR}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "error: pipeline file is missing: ${required_file}" >&2
        exit 1
    fi
done

# --- interpreter selection ---------------------------------------------------
INTERPRETER_KIND=""
if [[ -n "${COLMAP_PYTHON}" ]]; then
    INTERPRETER_KIND="custom"
elif [[ -x "${GPU_COLMAP_PYTHON}" && -f "${GPU_COLMAP_READY_MARKER}" ]]; then
    COLMAP_PYTHON="${GPU_COLMAP_PYTHON}"; INTERPRETER_KIND="cudss"
elif [[ -x "${CUDA_COLMAP_PYTHON}" ]]; then
    COLMAP_PYTHON="${CUDA_COLMAP_PYTHON}"; INTERPRETER_KIND="cuda-wheel"
else
    COLMAP_PYTHON="${PROJECT_PYTHON}"; INTERPRETER_KIND="cpu-wheel"
fi
if [[ "${FEATURE_GPU}" == "auto" ]]; then
    case "${INTERPRETER_KIND}" in cudss|cuda-wheel) FEATURE_GPU="on" ;; cpu-wheel) FEATURE_GPU="off" ;; custom) FEATURE_GPU="on" ;; esac
fi
if [[ "${BA_USE_GPU}" == "auto" ]]; then
    case "${INTERPRETER_KIND}" in cudss) BA_USE_GPU="on" ;; *) BA_USE_GPU="off" ;; esac
fi
if [[ "${BLAS_SHIM}" == "auto" ]]; then
    case "${INTERPRETER_KIND}" in cudss) BLAS_SHIM="off" ;; *) BLAS_SHIM="on" ;; esac
fi
if [[ -z "${VOCAB_TREE_PATH}" && -f "${DEFAULT_VOCAB_TREE}" ]]; then
    VOCAB_TREE_PATH="${DEFAULT_VOCAB_TREE}"
fi
if [[ -n "${VOCAB_TREE_PATH}" ]]; then
    VOCAB_TREE_PATH="$(realpath -m -- "${VOCAB_TREE_PATH}")"
fi

# pycolmap pip wheels statically link a serial OpenBLAS that faiss vocab-tree
# retrieval calls from many OpenMP threads at once; interpose the vendored
# pthread OpenBLAS (third-party/openblas-threaded/README.md). The cuDSS build
# links a threaded BLAS and needs neither the shim nor the single-thread cap.
OPENBLAS_SHIM_DIR="${PROJECT_ROOT}/third-party/openblas-threaded"
OPENBLAS_SHIM="${OPENBLAS_SHIM_DIR}/libopenblas-pthread.so.0"
THREAD_NOTE=""
if [[ "${BLAS_SHIM}" == "on" ]]; then
    if [[ "${COLMAP_THREADS}" == "-1" ]] || (( COLMAP_THREADS > SHIM_MAX_THREADS )); then
        THREAD_NOTE=" (requested ${COLMAP_THREADS}; capped at ${SHIM_MAX_THREADS} for the BLAS shim)"
        COLMAP_THREADS="${SHIM_MAX_THREADS}"
    fi
    BLAS_SAFE_THREADS="${COLMAP_THREADS}"
else
    BLAS_SAFE_THREADS="${COLMAP_THREADS}"
    if [[ "${BLAS_SAFE_THREADS}" == "-1" ]] || (( BLAS_SAFE_THREADS > 64 )); then
        BLAS_SAFE_THREADS=64
    fi
fi
COLMAP_COMMAND=(
    env
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    "OMP_NUM_THREADS=${BLAS_SAFE_THREADS}"
)
if [[ "${BLAS_SHIM}" == "on" ]]; then
    COLMAP_COMMAND+=(
        "LD_PRELOAD=${OPENBLAS_SHIM}"
        "LD_LIBRARY_PATH=${OPENBLAS_SHIM_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        OPENBLAS_NUM_THREADS=1
    )
fi
COLMAP_COMMAND+=(
    "${COLMAP_PYTHON}" "${COLMAP_PREPARER}" "${DATASET_ROOT}"
    --matching vocab_tree
    --vocab-tree-num-images "${VOCAB_TREE_NUM_IMAGES}"
    --guided-matching
    --sift-affine-shape
    --sift-domain-size-pooling
    --feature-max-image-size "${FEATURE_MAX_IMAGE_SIZE}"
    --feature-max-num-features "${FEATURE_MAX_NUM_FEATURES}"
    --init-min-tri-angle "${INIT_MIN_TRI_ANGLE}"
    --camera-model "${CAMERA_MODEL}"
    --num-threads "${COLMAP_THREADS}"
    --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
    --min-source-registration-ratio "${MIN_SOURCE_REGISTRATION_RATIO}"
)
[[ -z "${VOCAB_TREE_PATH}" ]] || COLMAP_COMMAND+=(--vocab-tree "${VOCAB_TREE_PATH}")
[[ "${FEATURE_GPU}" != "on" ]] || COLMAP_COMMAND+=(--feature-gpu --feature-gpu-index 0)
# CUDA_VISIBLE_DEVICES makes the selected physical GPU index 0 inside the process.
[[ "${BA_USE_GPU}" != "on" ]] || COLMAP_COMMAND+=(--ba-use-gpu --ba-gpu-index 0)
(( REUSE_MATCHES == 0 )) || COLMAP_COMMAND+=(--reuse-matches)
COLMAP_COMMAND+=(
    --ba-global-frames-ratio "${BA_GLOBAL_FRAMES_RATIO}"
    --ba-global-points-ratio "${BA_GLOBAL_POINTS_RATIO}"
    --ba-global-max-refinements "${BA_GLOBAL_MAX_REFINEMENTS}"
    --ba-global-max-num-iterations "${BA_GLOBAL_MAX_NUM_ITERATIONS}"
)
(( SNAPSHOT_FRAMES_FREQ == 0 )) || COLMAP_COMMAND+=(--snapshot-frames-freq "${SNAPSHOT_FRAMES_FREQ}")
[[ -z "${MAPPING_BLAS_THREADS}" ]] || COLMAP_COMMAND+=(--mapping-blas-threads "${MAPPING_BLAS_THREADS}")

# --- dataset identity (sources + extraction recipe) for validation ------------
read_metadata() {
    "${PROJECT_PYTHON}" - "${DATASET_ROOT}" <<'PY'
import json, sys
from pathlib import Path
m = json.loads((Path(sys.argv[1]) / "metadata.json").read_text())
sources = [s["source_video"] for s in m["sources"]] if m.get("dataset_type") == "video_set" else [m["source_video"]]
print(m["sample_fps"]); print(m["max_frames"])
print(m.get("sharpness_window_seconds", 0.0)); print(m.get("sharpness_keep_ratio", 0.0)); print(m.get("sharpness_drop_ratio", 0.0))
print(len(sources)); print("\n".join(sources))
PY
}
SOURCE_VIDEOS=()
SAMPLE_FPS="?"; MAX_FRAMES="?"; SHARP_WINDOW="?"; SHARP_KEEP="?"; SHARP_DROP="?"
if [[ -f "${DATASET_ROOT}/metadata.json" && -x "${PROJECT_PYTHON}" ]]; then
    mapfile -t META < <(read_metadata)
    SAMPLE_FPS="${META[0]}"; MAX_FRAMES="${META[1]}"; SHARP_WINDOW="${META[2]}"; SHARP_KEEP="${META[3]}"; SHARP_DROP="${META[4]}"
    SOURCE_VIDEOS=("${META[@]:6}")
elif (( DRY_RUN == 0 )); then
    echo "error: dataset metadata not found (or gsplat env missing): ${DATASET_ROOT}/metadata.json" >&2
    exit 1
fi
VALIDATE_COMMAND=(
    "${PROJECT_PYTHON}" "${VALIDATOR}" "${DATASET_ROOT}" "${SOURCE_VIDEOS[@]}"
    --sample-fps "${SAMPLE_FPS}" --max-frames "${MAX_FRAMES}"
    --sharpness-window "${SHARP_WINDOW}" --sharpness-keep-ratio "${SHARP_KEEP}" --sharpness-drop-ratio "${SHARP_DROP}"
)
VALIDATE_COLMAP_COMMAND=(
    "${VALIDATE_COMMAND[@]}"
    --camera-model "${CAMERA_MODEL}"
    --matching-strategy vocab_tree
    --vocab-tree-num-images "${VOCAB_TREE_NUM_IMAGES}"
    --guided-matching --sift-affine-shape --sift-domain-size-pooling
    --feature-max-image-size "${FEATURE_MAX_IMAGE_SIZE}"
    --feature-max-num-features "${FEATURE_MAX_NUM_FEATURES}"
    --init-min-tri-angle "${INIT_MIN_TRI_ANGLE}"
    --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
    --min-source-registration-ratio "${MIN_SOURCE_REGISTRATION_RATIO}"
    --require-colmap
)
[[ -z "${VOCAB_TREE_PATH}" ]] || VALIDATE_COLMAP_COMMAND+=(--vocab-tree "${VOCAB_TREE_PATH}")

if (( RESUME_FROM_SNAPSHOT == 1 )); then
    SNAPSHOT_BASE="${DATASET_ROOT}/snapshots"
    if [[ -d "${SNAPSHOT_BASE}" ]]; then
        while IFS= read -r candidate; do
            if [[ -s "${SNAPSHOT_BASE}/${candidate}/points3D.bin" ]]; then
                RESUME_MODEL_DIR="${SNAPSHOT_BASE}/${candidate}"; break
            fi
        done < <(ls -1 "${SNAPSHOT_BASE}" 2>/dev/null | sort -rn)
    fi
    if [[ -z "${RESUME_MODEL_DIR}" ]]; then
        if (( DRY_RUN == 1 )); then RESUME_MODEL_DIR="${SNAPSHOT_BASE}/<latest>"; else
            echo "error: --resume-from-snapshot found no complete snapshot under ${SNAPSHOT_BASE}" >&2; exit 1
        fi
    fi
    COLMAP_COMMAND+=(--resume-model "${RESUME_MODEL_DIR}")
fi

print_command() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run_command() { print_command "$@"; if (( DRY_RUN == 0 )); then "$@"; fi; }

sparse_model_complete() {
    [[ -s "${DATASET_ROOT}/sparse/0/cameras.bin" && -s "${DATASET_ROOT}/sparse/0/images.bin" && -s "${DATASET_ROOT}/sparse/0/points3D.bin" ]]
}
has_colmap_state() {
    [[ -e "${DATASET_ROOT}/database.db" || -e "${DATASET_ROOT}/sparse" || -e "${DATASET_ROOT}/reconstruction_stats.json" ]]
}
archive_colmap_state() {
    local destination="${DATASET_PARENT}/_failed/${DATASET_NAME}-colmap-${RUN_TIMESTAMP}-$$"
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: COLMAP state is stale or incomplete: ${DATASET_ROOT}" >&2
        echo "Rerun with --retry-incomplete to archive it (images are kept), or --reuse-matches / --resume-from-snapshot to continue." >&2
        return 1
    fi
    mkdir -p "${destination}"
    echo "Archiving stale COLMAP state: ${DATASET_ROOT} -> ${destination}"
    local artifact
    for artifact in database.db database.db-shm database.db-wal sparse snapshots reconstruction_stats.json; do
        [[ -e "${DATASET_ROOT}/${artifact}" ]] && mv "${DATASET_ROOT}/${artifact}" "${destination}/"
    done
    return 0
}

print_recipe() {
    echo "Dataset:        ${DATASET_ROOT} (${#SOURCE_VIDEOS[@]} source videos, ${SAMPLE_FPS} fps, window ${SHARP_WINDOW}s keep ${SHARP_KEEP} drop ${SHARP_DROP})"
    echo "Interpreter:    ${COLMAP_PYTHON} [${INTERPRETER_KIND}]"
    echo "GPU:            physical ${GPU_ID} -> features ${FEATURE_GPU}, bundle adjustment ${BA_USE_GPU}"
    echo "Threads:        ${COLMAP_THREADS}${THREAD_NOTE}; OMP_NUM_THREADS=${BLAS_SAFE_THREADS}; BLAS shim ${BLAS_SHIM}"
    echo "Camera:         single shared ${CAMERA_MODEL} (EXIF 35 mm-equivalent focal prior when present)"
    echo "Matching:       vocab tree ${VOCAB_TREE_NUM_IMAGES} neighbors, guided, affine+DSP SIFT, ${FEATURE_MAX_IMAGE_SIZE} px, ${FEATURE_MAX_NUM_FEATURES} features"
    echo "Vocab tree:     ${VOCAB_TREE_PATH:-<pycolmap default download>}"
    echo "Mapper:         init-min-tri-angle ${INIT_MIN_TRI_ANGLE}; BA schedule frames ${BA_GLOBAL_FRAMES_RATIO} points ${BA_GLOBAL_POINTS_RATIO} refinements ${BA_GLOBAL_MAX_REFINEMENTS} iterations ${BA_GLOBAL_MAX_NUM_ITERATIONS}; snapshots every ${SNAPSHOT_FRAMES_FREQ} frames (CPU bundle adjustment unless the cuDSS build is used)"
    echo "Gates:          registration >= ${MIN_REGISTRATION_RATIO} overall, >= ${MIN_SOURCE_REGISTRATION_RATIO} per video"
}

run_pipeline() {
    print_recipe
    if (( DRY_RUN == 1 )); then
        run_command "${COLMAP_COMMAND[@]}"
        run_command "${VALIDATE_COLMAP_COMMAND[@]}"
        echo "Dry run complete; no files were written."
        return 0
    fi
    if sparse_model_complete && "${VALIDATE_COLMAP_COMMAND[@]}"; then
        echo "Skipping completed COLMAP reconstruction: ${DATASET_ROOT}/sparse/0"
        return 0
    fi
    if ! "${VALIDATE_COMMAND[@]}"; then
        echo "error: dataset frames do not validate against their metadata: ${DATASET_ROOT}" >&2
        return 1
    fi
    if (( REUSE_MATCHES == 1 )); then
        if [[ ! -s "${DATASET_ROOT}/database.db" ]]; then
            echo "error: --reuse-matches needs an existing database: ${DATASET_ROOT}/database.db" >&2
            return 1
        fi
        echo "Reusing features and matches: ${DATASET_ROOT}/database.db"
        rm -rf "${DATASET_ROOT}/sparse"
        rm -f "${DATASET_ROOT}/reconstruction_stats.json"
        (( RESUME_FROM_SNAPSHOT == 1 )) || rm -rf "${DATASET_ROOT}/snapshots"
    elif has_colmap_state; then
        archive_colmap_state || return 1
    fi
    run_command "${COLMAP_COMMAND[@]}"
    if ! sparse_model_complete || ! "${VALIDATE_COLMAP_COMMAND[@]}"; then
        echo "error: COLMAP finished without a valid passing model" >&2
        return 1
    fi
    echo "Completed shared-camera COLMAP model: ${DATASET_ROOT}/sparse/0"
}

if (( DRY_RUN == 1 )); then
    run_pipeline
    exit $?
fi

if [[ ! -x "${PROJECT_PYTHON}" ]]; then
    echo "error: gsplat environment is missing; run scripts/setup_gsplat.sh first" >&2; exit 1
fi
if [[ ! -x "${COLMAP_PYTHON}" ]]; then
    echo "error: COLMAP interpreter is not executable: ${COLMAP_PYTHON}" >&2; exit 1
fi
if [[ "${BLAS_SHIM}" == "on" && ! -f "${OPENBLAS_SHIM}" ]]; then
    echo "error: threaded OpenBLAS shim is missing: ${OPENBLAS_SHIM} (see third-party/openblas-threaded/README.md)" >&2; exit 1
fi
if [[ -n "${VOCAB_TREE_PATH}" && ! -f "${VOCAB_TREE_PATH}" ]]; then
    echo "error: vocabulary tree not found: ${VOCAB_TREE_PATH}" >&2; exit 1
fi
if [[ ! -d "${DATASET_ROOT}/images" ]]; then
    echo "error: dataset images not found: ${DATASET_ROOT}/images" >&2; exit 1
fi
for source_video in "${SOURCE_VIDEOS[@]}"; do
    if [[ ! -f "${source_video}" ]]; then
        echo "error: source video not found: ${source_video}" >&2; exit 1
    fi
done
if [[ "${FEATURE_GPU}" == "on" || "${BA_USE_GPU}" == "on" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -i "${GPU_ID}" >/dev/null; then
        echo "error: GPU ${GPU_ID} is not visible to nvidia-smi" >&2; exit 1
    fi
    echo "GPU ${GPU_ID} current occupancy (shared machine; other users' jobs stay untouched):"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "${GPU_ID}" | sed 's/^/  /'
fi

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCK_DIR="${DATASET_PARENT}/.${DATASET_NAME}-colmap-lock"
LOCK_ACQUIRED=0
cleanup_lock() {
    if (( LOCK_ACQUIRED == 1 )) && [[ -f "${LOCK_DIR}/pid" ]] && [[ "$(<"${LOCK_DIR}/pid")" == "$$" ]]; then
        rm -f "${LOCK_DIR}/pid" "${LOCK_DIR}/hostname"; rmdir "${LOCK_DIR}" 2>/dev/null || true
    fi
}
trap cleanup_lock EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "error: another COLMAP run may be active for ${DATASET_NAME}: ${LOCK_DIR}" >&2
    echo "Verify no run is active before removing a stale lock." >&2
    exit 1
fi
printf '%s\n' "$$" > "${LOCK_DIR}/pid"; hostname > "${LOCK_DIR}/hostname"; LOCK_ACQUIRED=1

LOG_DIR="${DATASET_PARENT}/_logs"
LOG_PATH="${LOG_DIR}/colmap_${DATASET_NAME}.log"
mkdir -p "${LOG_DIR}"
echo "=== ${RUN_TIMESTAMP} run_object_colmap.sh $*" >> "${LOG_PATH}"
set +e
run_pipeline 2>&1 | tee -a "${LOG_PATH}"
pipeline_status="${PIPESTATUS[0]}"
set -e
if (( pipeline_status != 0 )); then
    echo "COLMAP failed; log: ${LOG_PATH}" >&2
    exit "${pipeline_status}"
fi
echo "COLMAP log: ${LOG_PATH}"
