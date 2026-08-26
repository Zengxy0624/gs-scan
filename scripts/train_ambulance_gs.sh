#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"
TRAINER="${GSPLAT_ROOT}/examples/simple_trainer.py"
PROJECT_PYTHON="${PROJECT_PYTHON:-${GSPLAT_ROOT}/.venv/bin/python}"

CAPTURE_ROOT="${CAPTURE_ROOT:-${PROJECT_ROOT}/captures/ambulance}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/work/ambulance/merged_vocabtree_single_opencv}"
OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/ambulance}"

PROFILE="full"
GPU_ID="0"
SAMPLE_FPS="3"
MAX_FRAMES="100000"
CAMERA_MODEL="OPENCV"
MATCHING_STRATEGY="vocab_tree"
OVERLAP="15"
MIN_REGISTRATION_RATIO="0.8"
MIN_SOURCE_REGISTRATION_RATIO="0.8"
COLMAP_THREADS="-1"
RETRY_INCOMPLETE=0
DRY_RUN=0

VIDEO_STEMS=(
    "A001_08071507_C004"
    "A001_08071511_C005"
    "A001_08071516_C006"
    "A001_08071521_C007"
    "A001_08071522_C008"
    "A001_08071524_C009"
    "A001_08071542_C014"
    "A001_08071544_C016"
    "A001_08071548_C017"
    "A001_08071550_C018"
    "A001_08071550_C019"
    "A001_08071649_C022"
    "A001_08071655_C023"
    "A001_08071655_C024"
    "A001_08071656_C025"
    "A001_08071657_C026"
    "A001_08071657_C027"
    "A001_08071705_C028"
)

usage() {
    cat <<'EOF'
Train gsplat on the merged ambulance vocabulary-tree COLMAP reconstruction.

Usage:
  bash scripts/train_ambulance_gs.sh [options]

Options:
  --profile pilot|full  Train for 7,000 or 30,000 steps (default: full).
  --gpu ID              Physical GPU used for training (default: 0).
  --retry-incomplete    Archive an incomplete/stale result and restart it.
  --dry-run             Print the manifest and trainer commands without writes.
  -h, --help            Show this help.

Environment overrides:
  CAPTURE_ROOT    Source videos (default: captures/ambulance).
  DATASET_ROOT    Merged passing COLMAP dataset.
  OUTPUT_ROOT     Durable training output base (default: outputs/ambulance).
  PROJECT_PYTHON  Python from the prepared gsplat environment.

The trainer uses the same full-quality settings as the existing project jobs:
the upstream default 3DGS strategy, data factor 1, SfM initialization, a single
GPU, final evaluation/checkpoint/PLY at the requested step, and no viewer. The
upstream trainer does not resume partial jobs or extend a pilot into a full run.
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

case "${PROFILE}" in
    pilot) MAX_STEPS=7000 ;;
    full) MAX_STEPS=30000 ;;
    *)
        echo "error: --profile must be pilot or full" >&2
        exit 2
        ;;
esac
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "error: --gpu must be one non-negative integer GPU index" >&2
    exit 2
fi
GPU_ID="$((10#${GPU_ID}))"

FINAL_STEP="$((MAX_STEPS - 1))"
RUN_KEY="vocabtree_single_opencv_${MAX_STEPS}steps"
RESULT_DIR="${OUTPUT_BASE}/${RUN_KEY}"
FAILED_OUTPUT_DIR="${OUTPUT_BASE}/_failed"
DESIRED_MANIFEST="${DATASET_ROOT}/training_manifest_${MAX_STEPS}.json"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

MANIFEST_BUILDER="${PROJECT_ROOT}/scripts/build_run_manifest.py"
REQUIRED_FILES=("${TRAINER}" "${MANIFEST_BUILDER}")
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "error: pipeline file is missing: ${required_file}" >&2
        exit 1
    fi
done

SOURCE_VIDEOS=()
for video_stem in "${VIDEO_STEMS[@]}"; do
    SOURCE_VIDEOS+=("${CAPTURE_ROOT}/${video_stem}.mov")
done

MANIFEST_COMMAND=(
    "${PROJECT_PYTHON}" "${MANIFEST_BUILDER}" build
    --scene ambulance_merged
    --dataset "${DATASET_ROOT}"
)
for source_video in "${SOURCE_VIDEOS[@]}"; do
    MANIFEST_COMMAND+=(--source "${source_video}")
done
MANIFEST_COMMAND+=(
    --gsplat-root "${GSPLAT_ROOT}"
    --sample-fps "${SAMPLE_FPS}"
    --max-frames "${MAX_FRAMES}"
    --overlap "${OVERLAP}"
    --matching-strategy "${MATCHING_STRATEGY}"
    --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
    --min-source-registration-ratio "${MIN_SOURCE_REGISTRATION_RATIO}"
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

run_training() {
    echo "Dataset: ${DATASET_ROOT}"
    echo "Result: ${RESULT_DIR}"
    echo "Profile: ${PROFILE} (${MAX_STEPS} steps)"
    echo "Training GPU: ${GPU_ID}"

    if (( DRY_RUN == 1 )); then
        run_command "${MANIFEST_COMMAND[@]}"
        run_command mkdir -p "${RESULT_DIR}"
        run_command cp "${DESIRED_MANIFEST}" "${RESULT_DIR}/pipeline_manifest.json"
        run_command "${TRAINER_COMMAND[@]}"
        echo "Dry run complete; no training was started."
        return 0
    fi

    run_command "${MANIFEST_COMMAND[@]}"
    prepare_result_directory || return 1
    if training_complete; then
        echo "Skipping completed training: ${RESULT_DIR}"
        return 0
    fi

    run_command mkdir -p "${RESULT_DIR}"
    run_command cp \
        "${DESIRED_MANIFEST}" "${RESULT_DIR}/pipeline_manifest.json"
    run_command "${TRAINER_COMMAND[@]}"

    # Rebuild provenance after the long job and require the exact same dataset
    # snapshot that was certified before training.
    run_command "${MANIFEST_COMMAND[@]}"
    if ! training_complete 1; then
        echo "error: training finished without all final artifacts or with a changed dataset" >&2
        return 1
    fi
    echo "Completed ambulance gsplat training: ${RESULT_DIR}"
}

if (( DRY_RUN == 1 )); then
    run_training
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
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi is required for training" >&2
    exit 1
fi
if ! nvidia-smi -i "${GPU_ID}" >/dev/null; then
    echo "error: GPU ${GPU_ID} is not visible to nvidia-smi" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}/images" ]]; then
    echo "error: merged training images not found: ${DATASET_ROOT}/images" >&2
    exit 1
fi
for model_file in cameras.bin images.bin points3D.bin; do
    if [[ ! -s "${DATASET_ROOT}/sparse/0/${model_file}" ]]; then
        echo "error: passing COLMAP model is incomplete: ${DATASET_ROOT}/sparse/0/${model_file}" >&2
        exit 1
    fi
done
for source_video in "${SOURCE_VIDEOS[@]}"; do
    if [[ ! -f "${source_video}" ]]; then
        echo "error: ambulance source video not found: ${source_video}" >&2
        exit 1
    fi
done

LOCK_DIR="${OUTPUT_BASE}/.ambulance-training-lock"
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

mkdir -p "${OUTPUT_BASE}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "error: another ambulance training run may be active: ${LOCK_DIR}" >&2
    echo "Verify no run is active before removing a stale lock." >&2
    exit 1
fi
printf '%s\n' "$$" > "${LOCK_DIR}/pid"
hostname > "${LOCK_DIR}/hostname"
LOCK_ACQUIRED=1

LOG_DIR="${OUTPUT_BASE}/_logs"
LOG_PATH="${LOG_DIR}/${RUN_KEY}.log"
mkdir -p "${LOG_DIR}"
set +e
run_training 2>&1 | tee -a "${LOG_PATH}"
training_status="${PIPESTATUS[0]}"
set -e
if (( training_status != 0 )); then
    echo "Ambulance training failed; log: ${LOG_PATH}" >&2
    exit "${training_status}"
fi
echo "Training log: ${LOG_PATH}"
