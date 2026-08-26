#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"
TRAINER="${GSPLAT_ROOT}/examples/simple_trainer.py"
PROJECT_PYTHON="${PROJECT_PYTHON:-${GSPLAT_ROOT}/.venv/bin/python}"

CAPTURE_ROOT="${CAPTURE_ROOT:-${PROJECT_ROOT}/captures/ambulance}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/work/ambulance}"
OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/single_video}"

VIDEO_STEM=""
PROFILE="full"
GPU_ID="6"
SAMPLE_FPS="3"
MAX_FRAMES="100000"
CAMERA_MODEL="OPENCV"
MATCHING="sequential"
OVERLAP="15"
VOCAB_TREE_NUM_IMAGES="100"
VOCAB_TREE_PATH=""
COLMAP_THREADS="-1"
MIN_REGISTRATION_RATIO="0.8"
PREPROCESS_ONLY=0
RETRY_INCOMPLETE=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Reconstruct and train gsplat on one video, reusing its existing frame cache.

Unlike the batch scripts, this takes any video stem, so a single capture can be
reconstructed and trained on its own without merging the whole set.

Usage:
  bash scripts/run_single_video_gs.sh --video STEM [options]

Options:
  --video STEM               Video name without extension (required), for
                             example A001_08071524_C009.
  --profile pilot|full       Train for 7,000 or 30,000 steps (default: full).
  --gpu ID                   Physical GPU used for training (default: 6).
  --matching sequential|exhaustive|vocab_tree
                             Feature pairing (default: sequential, which suits
                             one continuous camera path).
  --overlap N                Sequential quadratic-neighbor levels (default: 15).
  --vocab-tree-num-images N  Retrieved neighbors for vocab_tree (default: 100).
  --vocab-tree PATH          Local COLMAP vocabulary tree for vocab_tree
                             matching; the default auto-download does not work
                             on hosts where pycolmap cannot reach GitHub.
  --sample-fps FPS           Frame sampling rate; must match an existing frame
                             cache (default: 3).
  --max-frames N             Sampled-frame cap (default: 100000).
  --camera-model MODEL       Shared COLMAP camera model (default: OPENCV).
  --colmap-threads N         CPU threads; -1 lets COLMAP use all (default: -1).
  --min-registration-ratio R Registration quality gate (default: 0.8).
  --preprocess-only          Stop after COLMAP; do not train.
  --retry-incomplete         Archive stale/partial state and rebuild it.
  --dry-run                  Print every command without writing anything.
  -h, --help                 Show this help.

Environment overrides:
  CAPTURE_ROOT    Source videos (default: captures/ambulance).
  WORK_ROOT       Frame/COLMAP dataset base (default: work/ambulance).
  OUTPUT_ROOT     Durable training output base (default: outputs/single_video).
  PROJECT_PYTHON  Python from the prepared gsplat environment.

The dataset directory is WORK_ROOT/STEM. An existing validated frame cache is
reused as-is, so a capture already sampled by scripts/sample_ambulance_frames.sh
needs no re-extraction; otherwise frames are extracted first. COLMAP writes its
database and sparse model into that same directory, which leaves images/ and
metadata.json untouched and therefore safe to share with a merged dataset.

While a large batch reconstruction is running, pass --colmap-threads to leave
CPU for it; -1 will otherwise contend for every core on the machine.
EOF
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
        --video)
            require_value "$1" "${2:-}"
            VIDEO_STEM="$2"
            shift 2
            ;;
        --profile)
            require_value "$1" "${2:-}"
            PROFILE="$2"
            shift 2
            ;;
        --gpu)
            require_value "$1" "${2:-}"
            GPU_ID="$2"
            shift 2
            ;;
        --matching)
            require_value "$1" "${2:-}"
            MATCHING="$2"
            shift 2
            ;;
        --overlap)
            require_value "$1" "${2:-}"
            OVERLAP="$2"
            shift 2
            ;;
        --vocab-tree-num-images)
            require_value "$1" "${2:-}"
            VOCAB_TREE_NUM_IMAGES="$2"
            shift 2
            ;;
        --vocab-tree)
            require_value "$1" "${2:-}"
            VOCAB_TREE_PATH="$2"
            shift 2
            ;;
        --sample-fps)
            require_value "$1" "${2:-}"
            SAMPLE_FPS="$2"
            shift 2
            ;;
        --max-frames)
            require_value "$1" "${2:-}"
            MAX_FRAMES="$2"
            shift 2
            ;;
        --camera-model)
            require_value "$1" "${2:-}"
            CAMERA_MODEL="$2"
            shift 2
            ;;
        --colmap-threads)
            require_value "$1" "${2:-}"
            COLMAP_THREADS="$2"
            shift 2
            ;;
        --min-registration-ratio)
            require_value "$1" "${2:-}"
            MIN_REGISTRATION_RATIO="$2"
            shift 2
            ;;
        --preprocess-only)
            PREPROCESS_ONLY=1
            shift
            ;;
        --retry-incomplete)
            RETRY_INCOMPLETE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

positive_integer() {
    local value="$1"
    local option="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( 10#${value} < 1 )); then
        echo "error: ${option} must be a positive integer" >&2
        exit 2
    fi
}

positive_decimal_at_most_one() {
    local value="$1"
    local option="$2"
    if [[ ! "${value}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] ||
        [[ "${value}" =~ ^0([.]0+)?$ ]]; then
        echo "error: ${option} must be greater than 0 and at most 1" >&2
        exit 2
    fi
}

positive_decimal() {
    local value="$1"
    local option="$2"
    if [[ ! "${value}" =~ ^([0-9]+)([.][0-9]+)?$ ]] ||
        [[ "${value}" =~ ^0+([.]0+)?$ ]]; then
        echo "error: ${option} must be positive" >&2
        exit 2
    fi
}

if [[ -z "${VIDEO_STEM}" ]]; then
    echo "error: --video is required" >&2
    usage >&2
    exit 2
fi
# The stem becomes a directory name and a manifest scene label, so keep it to
# the same safe character set the extracted frame caches already use.
if [[ ! "${VIDEO_STEM}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "error: --video must be a plain file stem without a path" >&2
    exit 2
fi
case "${PROFILE}" in
    pilot) MAX_STEPS=7000 ;;
    full) MAX_STEPS=30000 ;;
    *)
        echo "error: --profile must be pilot or full" >&2
        exit 2
        ;;
esac
case "${MATCHING}" in
    sequential|exhaustive|vocab_tree) ;;
    *)
        echo "error: --matching must be sequential, exhaustive, or vocab_tree" >&2
        exit 2
        ;;
esac
case "${CAMERA_MODEL}" in
    SIMPLE_PINHOLE|PINHOLE|SIMPLE_RADIAL|RADIAL|OPENCV|OPENCV_FISHEYE) ;;
    *)
        echo "error: --camera-model is not a supported COLMAP model" >&2
        exit 2
        ;;
esac
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "error: --gpu must be one non-negative integer GPU index" >&2
    exit 2
fi
GPU_ID="$((10#${GPU_ID}))"
positive_integer "${OVERLAP}" "--overlap"
positive_integer "${VOCAB_TREE_NUM_IMAGES}" "--vocab-tree-num-images"
positive_integer "${MAX_FRAMES}" "--max-frames"
if (( 10#${MAX_FRAMES} < 2 )); then
    echo "error: --max-frames must be at least 2" >&2
    exit 2
fi
positive_decimal "${SAMPLE_FPS}" "--sample-fps"
positive_decimal_at_most_one \
    "${MIN_REGISTRATION_RATIO}" "--min-registration-ratio"
if [[ "${COLMAP_THREADS}" != "-1" ]]; then
    positive_integer "${COLMAP_THREADS}" "--colmap-threads"
fi
if [[ -n "${VOCAB_TREE_PATH}" && "${MATCHING}" != "vocab_tree" ]]; then
    echo "error: --vocab-tree requires --matching vocab_tree" >&2
    exit 2
fi
if [[ -n "${VOCAB_TREE_PATH}" ]]; then
    VOCAB_TREE_PATH="$(realpath -m -- "${VOCAB_TREE_PATH}")"
fi

EXTRACTOR="${PROJECT_ROOT}/scripts/extract_frames.py"
COLMAP_PREPARER="${PROJECT_ROOT}/scripts/prepare_colmap.py"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_dataset_cache.py"
MANIFEST_BUILDER="${PROJECT_ROOT}/scripts/build_run_manifest.py"
REQUIRED_FILES=(
    "${EXTRACTOR}"
    "${COLMAP_PREPARER}"
    "${VALIDATOR}"
    "${MANIFEST_BUILDER}"
    "${TRAINER}"
)
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "error: pipeline file is missing: ${required_file}" >&2
        exit 1
    fi
done

DATASET_ROOT="${WORK_ROOT}/${VIDEO_STEM}"
CAMERA_MODEL_LABEL="$(printf '%s' "${CAMERA_MODEL}" | tr '[:upper:]' '[:lower:]')"
RUN_KEY="${MATCHING}_single_${CAMERA_MODEL_LABEL}_${MAX_STEPS}steps"
RESULT_DIR="${OUTPUT_BASE}/${VIDEO_STEM}/${RUN_KEY}"
FAILED_OUTPUT_DIR="${OUTPUT_BASE}/${VIDEO_STEM}/_failed"
FAILED_WORK_DIR="${WORK_ROOT}/_failed"
DESIRED_MANIFEST="${DATASET_ROOT}/training_manifest_${RUN_KEY}.json"
FINAL_STEP="$((MAX_STEPS - 1))"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

SOURCE_VIDEO=""
for extension in mov MOV mp4 MP4 mkv MKV; do
    if [[ -f "${CAPTURE_ROOT}/${VIDEO_STEM}.${extension}" ]]; then
        SOURCE_VIDEO="${CAPTURE_ROOT}/${VIDEO_STEM}.${extension}"
        break
    fi
done
SOURCE_VIDEO_FOUND=1
if [[ -z "${SOURCE_VIDEO}" ]]; then
    # Keep a printable path so --dry-run works before the capture is staged.
    SOURCE_VIDEO="${CAPTURE_ROOT}/${VIDEO_STEM}.mov"
    SOURCE_VIDEO_FOUND=0
fi

EXTRACT_COMMAND=(
    uv run --project "${GSPLAT_ROOT}" --no-sync
    python "${EXTRACTOR}" "${SOURCE_VIDEO}" "${DATASET_ROOT}"
    --sample-fps "${SAMPLE_FPS}"
    --max-frames "${MAX_FRAMES}"
)

# pycolmap's wheel statically links a single-threaded OpenBLAS that corrupts
# its buffer bookkeeping and segfaults when faiss vocabulary-tree retrieval
# calls it from many OpenMP threads at once. Interpose the vendored pthread
# OpenBLAS (see third-party/openblas-threaded/README.md), keep each BLAS call
# single-threaded, and keep the OpenMP fan-out inside the shim's
# MAX_THREADS=64 build limit. Only this pthread flavor is safe here; serial
# and OpenMP-flavor builds both crashed full-scale reconstructions.
OPENBLAS_SHIM_DIR="${PROJECT_ROOT}/third-party/openblas-threaded"
OPENBLAS_SHIM="${OPENBLAS_SHIM_DIR}/libopenblas-pthread.so.0"
BLAS_SAFE_THREADS="${COLMAP_THREADS}"
if [[ "${BLAS_SAFE_THREADS}" == "-1" ]] || (( BLAS_SAFE_THREADS > 64 )); then
    BLAS_SAFE_THREADS=64
fi
COLMAP_ENV=(env "OMP_NUM_THREADS=${BLAS_SAFE_THREADS}")
if [[ "${MATCHING}" == "vocab_tree" ]]; then
    COLMAP_ENV+=(
        "LD_PRELOAD=${OPENBLAS_SHIM}"
        "LD_LIBRARY_PATH=${OPENBLAS_SHIM_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        OPENBLAS_NUM_THREADS=1
    )
fi
COLMAP_COMMAND=(
    "${COLMAP_ENV[@]}"
    uv run --project "${GSPLAT_ROOT}" --no-sync
    python "${COLMAP_PREPARER}" "${DATASET_ROOT}"
    --matching "${MATCHING}"
    --overlap "${OVERLAP}"
    --camera-model "${CAMERA_MODEL}"
    --num-threads "${COLMAP_THREADS}"
    --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
)
if [[ "${MATCHING}" == "vocab_tree" ]]; then
    COLMAP_COMMAND+=(--vocab-tree-num-images "${VOCAB_TREE_NUM_IMAGES}")
    if [[ -n "${VOCAB_TREE_PATH}" ]]; then
        COLMAP_COMMAND+=(--vocab-tree "${VOCAB_TREE_PATH}")
    fi
fi

VALIDATE_FRAMES_COMMAND=(
    "${PROJECT_PYTHON}" "${VALIDATOR}"
    "${DATASET_ROOT}" "${SOURCE_VIDEO}"
    --sample-fps "${SAMPLE_FPS}"
    --max-frames "${MAX_FRAMES}"
)
VALIDATE_COLMAP_COMMAND=(
    "${VALIDATE_FRAMES_COMMAND[@]}"
    --camera-model "${CAMERA_MODEL}"
    --matching-strategy "${MATCHING}"
    --overlap "${OVERLAP}"
    --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
    --require-colmap
)
if [[ "${MATCHING}" == "sequential" ]]; then
    VALIDATE_COLMAP_COMMAND+=(--quadratic-overlap)
elif [[ "${MATCHING}" == "vocab_tree" ]]; then
    VALIDATE_COLMAP_COMMAND+=(
        --vocab-tree-num-images "${VOCAB_TREE_NUM_IMAGES}"
    )
    if [[ -n "${VOCAB_TREE_PATH}" ]]; then
        VALIDATE_COLMAP_COMMAND+=(--vocab-tree "${VOCAB_TREE_PATH}")
    fi
fi

MANIFEST_COMMAND=(
    "${PROJECT_PYTHON}" "${MANIFEST_BUILDER}" build
    --scene "${VIDEO_STEM}"
    --dataset "${DATASET_ROOT}"
    --source "${SOURCE_VIDEO}"
    --gsplat-root "${GSPLAT_ROOT}"
    --sample-fps "${SAMPLE_FPS}"
    --max-frames "${MAX_FRAMES}"
    --overlap "${OVERLAP}"
    --matching-strategy "${MATCHING}"
    --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
    --max-steps "${MAX_STEPS}"
    --camera-model "${CAMERA_MODEL}"
    --colmap-threads "${COLMAP_THREADS}"
    --gpu-id "${GPU_ID}"
    --output "${DESIRED_MANIFEST}"
)

TRAINER_COMMAND=(
    env
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="${GPU_ID}"
    uv run --project "${GSPLAT_ROOT}" --no-sync
    python "${TRAINER}" default
    --data-dir "${DATASET_ROOT}"
    --data-factor 1
    --result-dir "${RESULT_DIR}"
    --max-steps "${MAX_STEPS}"
    --eval-steps "${MAX_STEPS}"
    --save-steps "${MAX_STEPS}"
    --ply-steps "${MAX_STEPS}"
    --save-ply
    --disable-viewer
)

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run_command() {
    print_command "$@"
    if (( DRY_RUN == 0 )); then
        "$@"
    fi
}

sparse_model_complete() {
    [[ -s "${DATASET_ROOT}/sparse/0/cameras.bin" ]] &&
        [[ -s "${DATASET_ROOT}/sparse/0/images.bin" ]] &&
        [[ -s "${DATASET_ROOT}/sparse/0/points3D.bin" ]]
}

has_colmap_state() {
    [[ -e "${DATASET_ROOT}/database.db" ]] ||
        [[ -e "${DATASET_ROOT}/sparse" ]] ||
        [[ -e "${DATASET_ROOT}/reconstruction_stats.json" ]]
}

training_artifacts_complete() {
    [[ -s "${RESULT_DIR}/ckpts/ckpt_${FINAL_STEP}_rank0.pt" ]] &&
        [[ -s "${RESULT_DIR}/ply/point_cloud_${FINAL_STEP}.ply" ]] &&
        [[ -s "${RESULT_DIR}/stats/val_step${FINAL_STEP}.json" ]] &&
        [[ -s "${RESULT_DIR}/videos/traj_${FINAL_STEP}.mp4" ]]
}

training_complete() {
    local strict_snapshot="${1:-0}"
    local compare_command=(
        "${PROJECT_PYTHON}" "${MANIFEST_BUILDER}" compare
        "${DESIRED_MANIFEST}" "${RESULT_DIR}/pipeline_manifest.json"
    )
    if (( strict_snapshot == 1 )); then
        compare_command+=(--strict-snapshot)
    fi
    training_artifacts_complete &&
        [[ -s "${RESULT_DIR}/pipeline_manifest.json" ]] &&
        "${compare_command[@]}"
}

archive_frame_cache() {
    local destination="${FAILED_WORK_DIR}/${VIDEO_STEM}-${RUN_TIMESTAMP}-$$"
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: frame cache is stale or incomplete: ${DATASET_ROOT}" >&2
        echo "Rerun with --retry-incomplete to archive and re-extract it." >&2
        return 1
    fi
    mkdir -p "${FAILED_WORK_DIR}"
    echo "Archiving stale frame cache: ${DATASET_ROOT} -> ${destination}"
    mv "${DATASET_ROOT}" "${destination}"
}

# Reconstruction state is archived without the frame cache so a failed COLMAP
# run never forces re-extracting frames that are still provably valid.
archive_colmap_state() {
    local destination="${FAILED_WORK_DIR}/${VIDEO_STEM}-colmap-${RUN_TIMESTAMP}-$$"
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: COLMAP state is stale or incomplete: ${DATASET_ROOT}" >&2
        echo "Rerun with --retry-incomplete to archive and rebuild it." >&2
        return 1
    fi
    mkdir -p "${destination}"
    echo "Archiving stale COLMAP state: ${DATASET_ROOT} -> ${destination}"
    local artifact
    for artifact in database.db database.db-shm database.db-wal sparse \
        reconstruction_stats.json; do
        if [[ -e "${DATASET_ROOT}/${artifact}" ]]; then
            mv "${DATASET_ROOT}/${artifact}" "${destination}/"
        fi
    done
}

prepare_result_directory() {
    if [[ ! -e "${RESULT_DIR}" ]] || training_complete; then
        return 0
    fi
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: incomplete or stale training result: ${RESULT_DIR}" >&2
        echo "Rerun with --retry-incomplete to archive and retrain it." >&2
        return 1
    fi
    local destination="${FAILED_OUTPUT_DIR}/${RUN_KEY}-${RUN_TIMESTAMP}-$$"
    mkdir -p "${FAILED_OUTPUT_DIR}"
    echo "Archiving stale training result: ${RESULT_DIR} -> ${destination}"
    mv "${RESULT_DIR}" "${destination}"
}

run_pipeline() {
    echo "Video: ${VIDEO_STEM}"
    echo "Source: ${SOURCE_VIDEO}"
    echo "Dataset: ${DATASET_ROOT}"
    echo "COLMAP: ${MATCHING} matching, shared ${CAMERA_MODEL} camera"
    if (( PREPROCESS_ONLY == 0 )); then
        echo "Result: ${RESULT_DIR}"
        echo "Profile: ${PROFILE} (${MAX_STEPS} steps) on GPU ${GPU_ID}"
    fi

    if (( DRY_RUN == 1 )); then
        run_command "${EXTRACT_COMMAND[@]}"
        run_command "${COLMAP_COMMAND[@]}"
        if (( PREPROCESS_ONLY == 0 )); then
            run_command "${MANIFEST_COMMAND[@]}"
            run_command mkdir -p "${RESULT_DIR}"
            run_command cp \
                "${DESIRED_MANIFEST}" "${RESULT_DIR}/pipeline_manifest.json"
            run_command "${TRAINER_COMMAND[@]}"
        fi
        echo "Dry run complete; nothing was written."
        return 0
    fi

    if [[ -e "${DATASET_ROOT}" || -L "${DATASET_ROOT}" ]]; then
        if ! "${VALIDATE_FRAMES_COMMAND[@]}"; then
            archive_frame_cache || return 1
        else
            echo "Reusing validated frame cache: ${DATASET_ROOT}/images"
        fi
    fi
    if [[ ! -e "${DATASET_ROOT}" ]]; then
        run_command "${EXTRACT_COMMAND[@]}"
    fi

    if sparse_model_complete && "${VALIDATE_COLMAP_COMMAND[@]}"; then
        echo "Skipping completed COLMAP reconstruction: ${DATASET_ROOT}/sparse/0"
    else
        if has_colmap_state; then
            archive_colmap_state || return 1
        fi
        run_command "${COLMAP_COMMAND[@]}"
        if ! sparse_model_complete || ! "${VALIDATE_COLMAP_COMMAND[@]}"; then
            echo "error: COLMAP finished without a valid passing model" >&2
            return 1
        fi
    fi
    echo "COLMAP model: ${DATASET_ROOT}/sparse/0"

    if (( PREPROCESS_ONLY == 1 )); then
        echo "Preprocessing complete for ${VIDEO_STEM}"
        return 0
    fi

    run_command "${MANIFEST_COMMAND[@]}"
    prepare_result_directory || return 1
    if training_complete; then
        echo "Skipping completed training: ${RESULT_DIR}"
        return 0
    fi

    run_command mkdir -p "${RESULT_DIR}"
    run_command cp "${DESIRED_MANIFEST}" "${RESULT_DIR}/pipeline_manifest.json"
    run_command "${TRAINER_COMMAND[@]}"

    # Rebuild provenance after the long job and require the exact same dataset
    # snapshot that was certified before training.
    run_command "${MANIFEST_COMMAND[@]}"
    if ! training_complete 1; then
        echo "error: training finished without all final artifacts or with a changed dataset" >&2
        return 1
    fi
    echo "Completed ${VIDEO_STEM} gsplat training: ${RESULT_DIR}"
}

if (( DRY_RUN == 1 )); then
    run_pipeline
    exit $?
fi

if [[ ! -x "${PROJECT_PYTHON}" ]]; then
    echo "error: gsplat environment is missing; run scripts/setup_gsplat.sh first" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not available on PATH" >&2
    exit 1
fi
if (( SOURCE_VIDEO_FOUND == 0 )); then
    echo "error: source video not found: ${CAPTURE_ROOT}/${VIDEO_STEM}.{mov,mp4,mkv}" >&2
    exit 1
fi
if [[ -n "${VOCAB_TREE_PATH}" && ! -f "${VOCAB_TREE_PATH}" ]]; then
    echo "error: vocabulary tree not found: ${VOCAB_TREE_PATH}" >&2
    exit 1
fi
if [[ "${MATCHING}" == "vocab_tree" && ! -f "${OPENBLAS_SHIM}" ]]; then
    echo "error: threaded OpenBLAS shim is missing: ${OPENBLAS_SHIM}" >&2
    echo "See third-party/openblas-threaded/README.md to restore it." >&2
    exit 1
fi
if [[ -L "${DATASET_ROOT}" ]]; then
    echo "error: dataset path must not be a symbolic link: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ -e "${DATASET_ROOT}" && ! -d "${DATASET_ROOT}" ]]; then
    echo "error: dataset path exists but is not a directory: ${DATASET_ROOT}" >&2
    exit 1
fi
if (( PREPROCESS_ONLY == 0 )); then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "error: nvidia-smi is required for training" >&2
        exit 1
    fi
    if ! nvidia-smi -i "${GPU_ID}" >/dev/null; then
        echo "error: GPU ${GPU_ID} is not visible to nvidia-smi" >&2
        exit 1
    fi
fi

LOCK_DIR="${WORK_ROOT}/.${VIDEO_STEM}-single-video-lock"
LOCK_ACQUIRED=0

cleanup_lock() {
    if (( LOCK_ACQUIRED == 1 )) &&
        [[ -f "${LOCK_DIR}/pid" ]] &&
        [[ "$(<"${LOCK_DIR}/pid")" == "$$" ]]; then
        rm -f "${LOCK_DIR}/pid" "${LOCK_DIR}/hostname"
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    fi
}
trap cleanup_lock EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "${WORK_ROOT}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "error: another run may be active for ${VIDEO_STEM}: ${LOCK_DIR}" >&2
    echo "Verify no run is active before removing a stale lock." >&2
    exit 1
fi
printf '%s\n' "$$" > "${LOCK_DIR}/pid"
hostname > "${LOCK_DIR}/hostname"
LOCK_ACQUIRED=1

LOG_DIR="${OUTPUT_BASE}/${VIDEO_STEM}/_logs"
LOG_PATH="${LOG_DIR}/${RUN_KEY}.log"
mkdir -p "${LOG_DIR}"
set +e
run_pipeline 2>&1 | tee -a "${LOG_PATH}"
pipeline_status="${PIPESTATUS[0]}"
set -e
if (( pipeline_status != 0 )); then
    echo "Single-video run failed; log: ${LOG_PATH}" >&2
    exit "${pipeline_status}"
fi
echo "Run log: ${LOG_PATH}"
