#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"
PROJECT_PYTHON="${PROJECT_PYTHON:-${GSPLAT_ROOT}/.venv/bin/python}"

CAPTURE_ROOT="${CAPTURE_ROOT:-${PROJECT_ROOT}/captures/ambulance}"
FRAME_ROOT="${FRAME_ROOT:-${PROJECT_ROOT}/work/ambulance}"
DATASET_ROOT="${DATASET_ROOT:-${FRAME_ROOT}/merged_vocabtree_single_opencv}"

SAMPLE_FPS="3"
MAX_FRAMES="100000"
CAMERA_MODEL="OPENCV"
VOCAB_TREE_NUM_IMAGES="200"
VOCAB_TREE_PATH=""
COLMAP_THREADS="-1"
MIN_REGISTRATION_RATIO="0.8"
MIN_SOURCE_REGISTRATION_RATIO="0.8"
FEATURE_MAX_IMAGE_SIZE="3840"
FEATURE_MAX_NUM_FEATURES="8192"
INIT_MIN_TRI_ANGLE="8"
LINK_MODE="hardlink"
RETRY_INCOMPLETE=0
REUSE_MATCHES=0
RESUME_FROM_SNAPSHOT=0
RESUME_MODEL_DIR=""
BA_GLOBAL_FRAMES_RATIO=""
BA_GLOBAL_POINTS_RATIO=""
BA_GLOBAL_MAX_REFINEMENTS=""
BA_GLOBAL_MAX_NUM_ITERATIONS=""
SNAPSHOT_FRAMES_FREQ=""
MAPPING_BLAS_THREADS=""
BA_USE_GPU=0
BA_GPU_INDEX="0"
COLMAP_PYTHON="${COLMAP_PYTHON:-}"
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
Merge all sampled ambulance-interior frames and reconstruct one COLMAP model.

The default recipe uses one shared OPENCV camera, vocabulary-tree matching with
200 retrieved images, guided matching, affine/DSP-SIFT at full 3840-pixel
resolution, and 80% global and per-video registration quality gates.

Usage:
  bash scripts/run_ambulance_colmap.sh [options]

Options:
  --vocab-tree-num-images N  Retrieved visual neighbors per image (default: 200).
  --vocab-tree PATH          Local COLMAP vocabulary tree; default auto-downloads.
  --colmap-threads N         CPU threads; -1 lets COLMAP use all (default: -1).
  --min-registration-ratio R
                             Overall registration gate (default: 0.8).
  --min-source-registration-ratio R
                             Per-video registration gate (default: 0.8).
  --feature-max-image-size N Feature extraction dimension (default: 3840).
  --feature-max-num-features N
                             SIFT feature cap per image (default: 8192).
  --init-min-tri-angle D     Initial-pair angle in degrees (default: 8).
  --link-mode hardlink|auto|copy
                             How merged JPEGs reuse source data (default: hardlink).
  --reuse-matches            Map from the dataset's existing database.db instead
                             of re-extracting and re-matching. Requires that the
                             database was built with this script's exact feature
                             and matching settings; stale mapping output is
                             removed first.
  --resume-from-snapshot     Continue mapping from the newest model checkpoint
                             under <dataset>/snapshots instead of restarting
                             mapping from scratch. Implies --reuse-matches.
  --ba-global-frames-ratio R Run a global bundle adjustment when the model grows
                             by this frame factor (COLMAP default: 1.1; larger
                             is faster on big scenes).
  --ba-global-points-ratio R Same trigger for point growth (COLMAP default: 1.1).
  --ba-global-max-refinements N
                             Refinement cycles per global adjustment (COLMAP
                             default: 5).
  --ba-global-max-num-iterations N
                             Solver iterations per global adjustment (COLMAP
                             default: 50).
  --snapshot-frames-freq N   Checkpoint the growing model to <dataset>/snapshots
                             every N registered frames (default: off).
  --mapping-blas-threads N   BLAS threads for the mapping stage's linear solves
                             (preparer default: 48; 0 keeps BLAS single-threaded).
  --ba-use-gpu               Solve bundle adjustment on a GPU. Needs a CUDA+cuDSS
                             pycolmap; set COLMAP_PYTHON to its interpreter (the
                             default is the source-built environment under
                             work/_gpu_colmap when present).
  --ba-gpu-index N           GPU used for --ba-use-gpu (default: 0).
  --retry-incomplete         Archive stale/partial merged work and retry.
  --dry-run                  Print the merge and COLMAP commands without writes.
  -h, --help                 Show this help.

Environment overrides:
  CAPTURE_ROOT    Source videos (default: captures/ambulance).
  FRAME_ROOT      Per-video frame caches (default: work/ambulance).
  DATASET_ROOT    Merged COLMAP dataset output.
  PROJECT_PYTHON  Python from the prepared gsplat environment.

The merged images are hard-linked by default, so the 13 GB JPEG set is not
duplicated. Do not edit hard-linked images in place. The source videos must all
come from the same physical camera with unchanged lens, zoom, resolution, and
stabilization/crop settings for the shared-camera assumption to be valid.
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
        --min-source-registration-ratio)
            require_value "$1" "${2:-}"
            MIN_SOURCE_REGISTRATION_RATIO="$2"
            shift 2
            ;;
        --feature-max-image-size)
            require_value "$1" "${2:-}"
            FEATURE_MAX_IMAGE_SIZE="$2"
            shift 2
            ;;
        --feature-max-num-features)
            require_value "$1" "${2:-}"
            FEATURE_MAX_NUM_FEATURES="$2"
            shift 2
            ;;
        --init-min-tri-angle)
            require_value "$1" "${2:-}"
            INIT_MIN_TRI_ANGLE="$2"
            shift 2
            ;;
        --link-mode)
            require_value "$1" "${2:-}"
            LINK_MODE="$2"
            shift 2
            ;;
        --reuse-matches)
            REUSE_MATCHES=1
            shift
            ;;
        --resume-from-snapshot)
            RESUME_FROM_SNAPSHOT=1
            REUSE_MATCHES=1
            shift
            ;;
        --ba-global-frames-ratio)
            require_value "$1" "${2:-}"
            BA_GLOBAL_FRAMES_RATIO="$2"
            shift 2
            ;;
        --ba-global-points-ratio)
            require_value "$1" "${2:-}"
            BA_GLOBAL_POINTS_RATIO="$2"
            shift 2
            ;;
        --ba-global-max-refinements)
            require_value "$1" "${2:-}"
            BA_GLOBAL_MAX_REFINEMENTS="$2"
            shift 2
            ;;
        --ba-global-max-num-iterations)
            require_value "$1" "${2:-}"
            BA_GLOBAL_MAX_NUM_ITERATIONS="$2"
            shift 2
            ;;
        --snapshot-frames-freq)
            require_value "$1" "${2:-}"
            SNAPSHOT_FRAMES_FREQ="$2"
            shift 2
            ;;
        --mapping-blas-threads)
            require_value "$1" "${2:-}"
            MAPPING_BLAS_THREADS="$2"
            shift 2
            ;;
        --ba-use-gpu)
            BA_USE_GPU=1
            shift
            ;;
        --ba-gpu-index)
            require_value "$1" "${2:-}"
            BA_GPU_INDEX="$2"
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

positive_integer "${VOCAB_TREE_NUM_IMAGES}" "--vocab-tree-num-images"
positive_integer "${FEATURE_MAX_IMAGE_SIZE}" "--feature-max-image-size"
positive_integer "${FEATURE_MAX_NUM_FEATURES}" "--feature-max-num-features"
if [[ "${COLMAP_THREADS}" != "-1" ]]; then
    positive_integer "${COLMAP_THREADS}" "--colmap-threads"
fi
positive_decimal_at_most_one \
    "${MIN_REGISTRATION_RATIO}" "--min-registration-ratio"
positive_decimal_at_most_one \
    "${MIN_SOURCE_REGISTRATION_RATIO}" \
    "--min-source-registration-ratio"
positive_decimal "${INIT_MIN_TRI_ANGLE}" "--init-min-tri-angle"
if [[ -n "${BA_GLOBAL_FRAMES_RATIO}" ]]; then
    positive_decimal "${BA_GLOBAL_FRAMES_RATIO}" "--ba-global-frames-ratio"
fi
if [[ -n "${BA_GLOBAL_POINTS_RATIO}" ]]; then
    positive_decimal "${BA_GLOBAL_POINTS_RATIO}" "--ba-global-points-ratio"
fi
if [[ -n "${BA_GLOBAL_MAX_REFINEMENTS}" ]]; then
    positive_integer \
        "${BA_GLOBAL_MAX_REFINEMENTS}" "--ba-global-max-refinements"
fi
if [[ -n "${BA_GLOBAL_MAX_NUM_ITERATIONS}" ]]; then
    positive_integer \
        "${BA_GLOBAL_MAX_NUM_ITERATIONS}" "--ba-global-max-num-iterations"
fi
if [[ -n "${SNAPSHOT_FRAMES_FREQ}" ]]; then
    positive_integer "${SNAPSHOT_FRAMES_FREQ}" "--snapshot-frames-freq"
fi
if [[ -n "${MAPPING_BLAS_THREADS}" && "${MAPPING_BLAS_THREADS}" != "0" ]]; then
    positive_integer "${MAPPING_BLAS_THREADS}" "--mapping-blas-threads"
fi
if [[ ! "${BA_GPU_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "error: --ba-gpu-index must be a non-negative integer" >&2
    exit 2
fi
if (( BA_USE_GPU == 1 )) && [[ -z "${COLMAP_PYTHON}" ]]; then
    GPU_ENV_PYTHON="${PROJECT_ROOT}/work/_gpu_colmap/mamba_root/envs/gpubuild/bin/python"
    if [[ -x "${GPU_ENV_PYTHON}" ]]; then
        COLMAP_PYTHON="${GPU_ENV_PYTHON}"
    else
        echo "error: --ba-use-gpu needs COLMAP_PYTHON pointing at a CUDA+cuDSS pycolmap" >&2
        exit 2
    fi
fi
case "${LINK_MODE}" in
    hardlink|auto|copy) ;;
    *)
        echo "error: --link-mode must be hardlink, auto, or copy" >&2
        exit 2
        ;;
esac

MERGER="${PROJECT_ROOT}/scripts/merge_frame_datasets.py"
COLMAP_PREPARER="${PROJECT_ROOT}/scripts/prepare_colmap.py"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_dataset_cache.py"
REQUIRED_FILES=("${MERGER}" "${COLMAP_PREPARER}" "${VALIDATOR}")
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "error: pipeline file is missing: ${required_file}" >&2
        exit 1
    fi
done

FRAME_DATASETS=()
SOURCE_VIDEOS=()
for video_stem in "${VIDEO_STEMS[@]}"; do
    FRAME_DATASETS+=("${FRAME_ROOT}/${video_stem}")
    SOURCE_VIDEOS+=("${CAPTURE_ROOT}/${video_stem}.mov")
done

if [[ -n "${VOCAB_TREE_PATH}" ]]; then
    VOCAB_TREE_PATH="$(realpath -m -- "${VOCAB_TREE_PATH}")"
fi

if (( RESUME_FROM_SNAPSHOT == 1 )); then
    SNAPSHOT_BASE="${DATASET_ROOT}/snapshots"
    if [[ -d "${SNAPSHOT_BASE}" ]]; then
        while IFS= read -r candidate; do
            if [[ -s "${SNAPSHOT_BASE}/${candidate}/points3D.bin" ]]; then
                RESUME_MODEL_DIR="${SNAPSHOT_BASE}/${candidate}"
                break
            fi
        done < <(ls -1 "${SNAPSHOT_BASE}" 2>/dev/null | sort -rn)
    fi
    if [[ -z "${RESUME_MODEL_DIR}" ]]; then
        if (( DRY_RUN == 1 )); then
            RESUME_MODEL_DIR="${SNAPSHOT_BASE}/<latest>"
        else
            echo "error: --resume-from-snapshot found no complete snapshot under ${SNAPSHOT_BASE}" >&2
            exit 1
        fi
    fi
fi

MERGE_COMMAND=(
    "${PROJECT_PYTHON}" "${MERGER}" "${DATASET_ROOT}"
    "${FRAME_DATASETS[@]}" --link-mode "${LINK_MODE}"
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
# A CUDA-enabled interpreter (COLMAP_PYTHON) links a proper threaded BLAS, so
# it needs neither the shim nor the single-thread cap the wheel requires.
if [[ -n "${COLMAP_PYTHON}" ]]; then
    COLMAP_COMMAND=(
        env "OMP_NUM_THREADS=${BLAS_SAFE_THREADS}"
        "${COLMAP_PYTHON}"
    )
else
    COLMAP_COMMAND=(
        env
        "LD_PRELOAD=${OPENBLAS_SHIM}"
        "LD_LIBRARY_PATH=${OPENBLAS_SHIM_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        "OMP_NUM_THREADS=${BLAS_SAFE_THREADS}"
        OPENBLAS_NUM_THREADS=1
        uv run --project "${GSPLAT_ROOT}" --no-sync
        python
    )
fi
COLMAP_COMMAND+=(
    "${COLMAP_PREPARER}" "${DATASET_ROOT}"
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
if [[ -n "${VOCAB_TREE_PATH}" ]]; then
    COLMAP_COMMAND+=(--vocab-tree "${VOCAB_TREE_PATH}")
fi
if (( REUSE_MATCHES == 1 )); then
    COLMAP_COMMAND+=(--reuse-matches)
fi
if [[ -n "${RESUME_MODEL_DIR}" ]]; then
    COLMAP_COMMAND+=(--resume-model "${RESUME_MODEL_DIR}")
fi
if [[ -n "${BA_GLOBAL_FRAMES_RATIO}" ]]; then
    COLMAP_COMMAND+=(--ba-global-frames-ratio "${BA_GLOBAL_FRAMES_RATIO}")
fi
if [[ -n "${BA_GLOBAL_POINTS_RATIO}" ]]; then
    COLMAP_COMMAND+=(--ba-global-points-ratio "${BA_GLOBAL_POINTS_RATIO}")
fi
if [[ -n "${BA_GLOBAL_MAX_REFINEMENTS}" ]]; then
    COLMAP_COMMAND+=(
        --ba-global-max-refinements "${BA_GLOBAL_MAX_REFINEMENTS}"
    )
fi
if [[ -n "${BA_GLOBAL_MAX_NUM_ITERATIONS}" ]]; then
    COLMAP_COMMAND+=(
        --ba-global-max-num-iterations "${BA_GLOBAL_MAX_NUM_ITERATIONS}"
    )
fi
if [[ -n "${SNAPSHOT_FRAMES_FREQ}" ]]; then
    COLMAP_COMMAND+=(--snapshot-frames-freq "${SNAPSHOT_FRAMES_FREQ}")
fi
if [[ -n "${MAPPING_BLAS_THREADS}" ]]; then
    COLMAP_COMMAND+=(--mapping-blas-threads "${MAPPING_BLAS_THREADS}")
fi
if (( BA_USE_GPU == 1 )); then
    COLMAP_COMMAND+=(--ba-use-gpu --ba-gpu-index "${BA_GPU_INDEX}")
fi

VALIDATE_FRAMES_COMMAND=(
    "${PROJECT_PYTHON}" "${VALIDATOR}"
    "${DATASET_ROOT}" "${SOURCE_VIDEOS[@]}"
    --sample-fps "${SAMPLE_FPS}"
    --max-frames "${MAX_FRAMES}"
)
VALIDATE_COLMAP_COMMAND=(
    "${VALIDATE_FRAMES_COMMAND[@]}"
    --camera-model "${CAMERA_MODEL}"
    --matching-strategy vocab_tree
    --vocab-tree-num-images "${VOCAB_TREE_NUM_IMAGES}"
    --guided-matching
    --sift-affine-shape
    --sift-domain-size-pooling
    --feature-max-image-size "${FEATURE_MAX_IMAGE_SIZE}"
    --feature-max-num-features "${FEATURE_MAX_NUM_FEATURES}"
    --init-min-tri-angle "${INIT_MIN_TRI_ANGLE}"
    --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
    --min-source-registration-ratio "${MIN_SOURCE_REGISTRATION_RATIO}"
    --require-colmap
)
if [[ -n "${VOCAB_TREE_PATH}" ]]; then
    VALIDATE_COLMAP_COMMAND+=(--vocab-tree "${VOCAB_TREE_PATH}")
fi

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

archive_stale_dataset() {
    local archive_base="$(dirname -- "${DATASET_ROOT}")/_failed"
    local destination="${archive_base}/$(basename -- "${DATASET_ROOT}")-${RUN_TIMESTAMP}-$$"
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: merged dataset is stale or incomplete: ${DATASET_ROOT}" >&2
        echo "Rerun with --retry-incomplete to archive and rebuild it." >&2
        return 1
    fi
    mkdir -p "${archive_base}"
    echo "Archiving stale merged dataset: ${DATASET_ROOT} -> ${destination}"
    mv "${DATASET_ROOT}" "${destination}"
}

run_pipeline() {
    echo "Merged dataset: ${DATASET_ROOT}"
    echo "Sources: ${#VIDEO_STEMS[@]} videos at ${SAMPLE_FPS} frames/second"
    echo "COLMAP: vocabulary tree (${VOCAB_TREE_NUM_IMAGES} neighbors), shared ${CAMERA_MODEL} camera"

    if (( DRY_RUN == 1 )); then
        run_command "${MERGE_COMMAND[@]}"
        run_command "${COLMAP_COMMAND[@]}"
        echo "Dry run complete; no files were written."
        return 0
    fi

    if [[ -e "${DATASET_ROOT}" || -L "${DATASET_ROOT}" ]]; then
        if sparse_model_complete && "${VALIDATE_COLMAP_COMMAND[@]}"; then
            echo "Skipping completed merged COLMAP reconstruction: ${DATASET_ROOT}/sparse/0"
            return 0
        fi
        if (( REUSE_MATCHES == 1 )); then
            if [[ ! -s "${DATASET_ROOT}/database.db" ]]; then
                echo "error: --reuse-matches needs an existing database: ${DATASET_ROOT}/database.db" >&2
                return 1
            fi
            if ! "${VALIDATE_FRAMES_COMMAND[@]}"; then
                echo "error: merged frames no longer validate; rerun without --reuse-matches" >&2
                return 1
            fi
            echo "Reusing validated merged frames and matches: ${DATASET_ROOT}/database.db"
            # Discard mapping output from an interrupted or failed attempt;
            # the feature/match database is kept, and so are the model
            # snapshots when this run resumes from one.
            rm -rf "${DATASET_ROOT}/sparse"
            rm -f "${DATASET_ROOT}/reconstruction_stats.json"
            if (( RESUME_FROM_SNAPSHOT == 0 )); then
                rm -rf "${DATASET_ROOT}/snapshots"
            fi
        elif has_colmap_state || ! "${VALIDATE_FRAMES_COMMAND[@]}"; then
            archive_stale_dataset || return 1
        else
            echo "Reusing validated merged frames: ${DATASET_ROOT}/images"
        fi
    fi

    if [[ ! -e "${DATASET_ROOT}" ]]; then
        if (( REUSE_MATCHES == 1 )); then
            echo "error: --reuse-matches needs an existing merged dataset: ${DATASET_ROOT}" >&2
            return 1
        fi
        run_command "${MERGE_COMMAND[@]}"
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
    echo "error: gsplat environment is missing; run scripts/setup_gsplat.sh first" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not available on PATH" >&2
    exit 1
fi
if [[ -z "${COLMAP_PYTHON}" && ! -f "${OPENBLAS_SHIM}" ]]; then
    echo "error: threaded OpenBLAS shim is missing: ${OPENBLAS_SHIM}" >&2
    echo "See third-party/openblas-threaded/README.md to restore it." >&2
    exit 1
fi
if [[ -n "${COLMAP_PYTHON}" && ! -x "${COLMAP_PYTHON}" ]]; then
    echo "error: COLMAP_PYTHON is not executable: ${COLMAP_PYTHON}" >&2
    exit 1
fi
for index in "${!VIDEO_STEMS[@]}"; do
    if [[ ! -d "${FRAME_DATASETS[index]}" ]]; then
        echo "error: sampled frame dataset not found: ${FRAME_DATASETS[index]}" >&2
        exit 1
    fi
    if [[ ! -f "${SOURCE_VIDEOS[index]}" ]]; then
        echo "error: ambulance video not found: ${SOURCE_VIDEOS[index]}" >&2
        exit 1
    fi
done
if [[ -n "${VOCAB_TREE_PATH}" && ! -f "${VOCAB_TREE_PATH}" ]]; then
    echo "error: vocabulary tree not found: ${VOCAB_TREE_PATH}" >&2
    exit 1
fi

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
# Keyed by dataset so runs on different dataset roots can proceed in parallel.
LOCK_DIR="${FRAME_ROOT}/.$(basename -- "${DATASET_ROOT}")-colmap-lock"
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

mkdir -p "${FRAME_ROOT}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "error: another ambulance COLMAP run may be active: ${LOCK_DIR}" >&2
    echo "Verify no run is active before removing a stale lock." >&2
    exit 1
fi
printf '%s\n' "$$" > "${LOCK_DIR}/pid"
hostname > "${LOCK_DIR}/hostname"
LOCK_ACQUIRED=1

LOG_DIR="${FRAME_ROOT}/_logs"
LOG_PATH="${LOG_DIR}/$(basename -- "${DATASET_ROOT}").log"
mkdir -p "${LOG_DIR}"
set +e
run_pipeline 2>&1 | tee -a "${LOG_PATH}"
pipeline_status="${PIPESTATUS[0]}"
set -e
if (( pipeline_status != 0 )); then
    echo "Ambulance COLMAP failed; log: ${LOG_PATH}" >&2
    exit "${pipeline_status}"
fi
echo "COLMAP log: ${LOG_PATH}"
