#!/usr/bin/env bash
set -Eeuo pipefail

# One object, end to end: frames -> COLMAP -> 3DGS training, by stage.
# Thin driver over scripts/prepare_object_frames.sh, scripts/run_object_colmap.sh
# and scripts/train_experiment.py; every underlying command is printed, so a
# stage can always be re-run by hand. Stages are idempotent where the
# underlying scripts are (frames, COLMAP); training refuses to overwrite an
# existing result directory. See docs/OBJECT_PIPELINE.md.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_PYTHON="${PROJECT_PYTHON:-${PROJECT_ROOT}/third-party/gsplat/.venv/bin/python}"

PROJECT=""
NAME=""
GPU_ID="0"
STAGES="frames,colmap,train"
TAG="exp1"
TRAIN_STEPS="120000"
CAMERA_MODEL="OPENCV"
BA_PRESET="fast"
VOCAB_TREE_NUM_IMAGES="200"
FEATURE_MAX_NUM_FEATURES="8192"
CAP_MAX="6500000"
DRY_RUN=0
CLIPS=()
PREPARE_ARGS=()
COLMAP_ARGS=()

usage() {
    cat <<'USAGE'
Run the object pipeline (frames -> COLMAP -> 3DGS) for one scanned object.

Usage:
  bash scripts/run_object_pipeline.sh --project PROJECT --name NAME [options] CLIP [CLIP ...]

Layout (PROJECT = capture family, NAME = object):
  captures/PROJECT/<CLIP>.mov + annotation.csv   inputs
  work/PROJECT/<CLIP>/                           per-clip frame caches
  work/PROJECT/NAME/                             merged dataset + COLMAP model
  outputs/PROJECT/NAME_TAG/                      training result

Options:
  --project PROJECT      Capture family (required).
  --name NAME            Object / merged dataset name (required).
  --gpu ID               Physical GPU for COLMAP features/BA and training (default: 0).
  --stages LIST          Comma list of frames,colmap,train (default: all).
  --tag TAG              Training result suffix (default: exp1).
  --train-steps N        Training steps: 120000 (production, steps-scaler 4) or
                         any multiple of 30000 <= 120000 down to 30000; 12000 for
                         a quick look (default: 120000).
  --camera-model MODEL   COLMAP shared camera model (default: OPENCV; use
                         OPENCV_FISHEYE for uncorrected fisheye footage).
  --ba-preset fast|production
                         Global BA schedule: fast = 1.3/1.3/2 refinements/30
                         iterations (validated on mannequin_front); production
                         = COLMAP 1.1/1.1/5/50 (default: fast).
  --vocab-tree-num-images N   Retrieved neighbors per image (default: 200; 100
                         roughly halves matching + BA time).
  --feature-max-num-features N  SIFT feature cap per image (default: 8192).
  --cap-max N            MCMC Gaussian budget (default: 6500000).
  --prepare-arg ARG      Extra argument passed to prepare_object_frames.sh (repeatable).
  --colmap-arg ARG       Extra argument passed to run_object_colmap.sh (repeatable).
  --dry-run              Print every command without running anything.
  -h, --help             Show this help.

Run under tmux. Logs: work/PROJECT/_logs/, outputs/PROJECT/_logs/.
USAGE
}

require_value() { if [[ -z "${2:-}" ]]; then echo "error: $1 requires a value" >&2; exit 2; fi; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) require_value "$1" "${2:-}"; PROJECT="$2"; shift 2 ;;
        --name) require_value "$1" "${2:-}"; NAME="$2"; shift 2 ;;
        --gpu) require_value "$1" "${2:-}"; GPU_ID="$2"; shift 2 ;;
        --stages) require_value "$1" "${2:-}"; STAGES="$2"; shift 2 ;;
        --tag) require_value "$1" "${2:-}"; TAG="$2"; shift 2 ;;
        --train-steps) require_value "$1" "${2:-}"; TRAIN_STEPS="$2"; shift 2 ;;
        --camera-model) require_value "$1" "${2:-}"; CAMERA_MODEL="$2"; shift 2 ;;
        --ba-preset) require_value "$1" "${2:-}"; BA_PRESET="$2"; shift 2 ;;
        --vocab-tree-num-images) require_value "$1" "${2:-}"; VOCAB_TREE_NUM_IMAGES="$2"; shift 2 ;;
        --feature-max-num-features) require_value "$1" "${2:-}"; FEATURE_MAX_NUM_FEATURES="$2"; shift 2 ;;
        --cap-max) require_value "$1" "${2:-}"; CAP_MAX="$2"; shift 2 ;;
        --prepare-arg) require_value "$1" "${2:-}"; PREPARE_ARGS+=("$2"); shift 2 ;;
        --colmap-arg) require_value "$1" "${2:-}"; COLMAP_ARGS+=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) CLIPS+=("$1"); shift ;;
    esac
done

plain_name() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "error: $2 must be a plain name" >&2; exit 2; }; }
positive_integer() { [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 >= 1 )) || { echo "error: $2 must be a positive integer" >&2; exit 2; }; }
[[ -n "${PROJECT}" ]] || { echo "error: --project is required" >&2; usage >&2; exit 2; }
[[ -n "${NAME}" ]] || { echo "error: --name is required" >&2; usage >&2; exit 2; }
plain_name "${PROJECT}" "--project"; plain_name "${NAME}" "--name"; plain_name "${TAG}" "--tag"
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || { echo "error: --gpu must be a non-negative integer" >&2; exit 2; }
positive_integer "${TRAIN_STEPS}" "--train-steps"; positive_integer "${VOCAB_TREE_NUM_IMAGES}" "--vocab-tree-num-images"
positive_integer "${FEATURE_MAX_NUM_FEATURES}" "--feature-max-num-features"; positive_integer "${CAP_MAX}" "--cap-max"
case "${BA_PRESET}" in fast|production) ;; *) echo "error: --ba-preset must be fast or production" >&2; exit 2 ;; esac
case "${CAMERA_MODEL}" in OPENCV|OPENCV_FISHEYE|SIMPLE_RADIAL|RADIAL|PINHOLE|SIMPLE_PINHOLE) ;; *) echo "error: --camera-model is not a supported COLMAP model" >&2; exit 2 ;; esac
IFS=',' read -r -a STAGE_LIST <<< "${STAGES}"
for stage in "${STAGE_LIST[@]}"; do
    case "${stage}" in frames|colmap|train) ;; *) echo "error: unknown stage: ${stage}" >&2; exit 2 ;; esac
done
stage_selected() { local s; for s in "${STAGE_LIST[@]}"; do [[ "${s}" == "$1" ]] && return 0; done; return 1; }
if stage_selected frames && (( ${#CLIPS[@]} < 2 )); then
    echo "error: the frames stage needs at least two clip stems" >&2; exit 2
fi

# Training schedule: production = 30k-step recipe scaled x4 (120k). Other
# lengths use steps-scaler 1 with an explicit step count and a proportional
# MCMC relocation stop (116k/120k of the run, as in exp6/exp11).
if (( TRAIN_STEPS == 120000 )); then
    STEPS_SCALER=4; MAX_STEPS=30000; REFINE_STOP=29000
else
    STEPS_SCALER=1; MAX_STEPS="${TRAIN_STEPS}"; REFINE_STOP=$(( TRAIN_STEPS * 29 / 30 ))
fi
FINAL_STEP=$(( TRAIN_STEPS - 1 ))

DATASET="${PROJECT_ROOT}/work/${PROJECT}/${NAME}"
RESULT_DIR="${PROJECT_ROOT}/outputs/${PROJECT}/${NAME}_${TAG}"
TRAIN_LOG_DIR="${PROJECT_ROOT}/outputs/${PROJECT}/_logs"
TRAIN_LOG="${TRAIN_LOG_DIR}/train_${NAME}_${TAG}.log"

PREPARE_COMMAND=(bash "${SCRIPT_DIR}/prepare_object_frames.sh" --project "${PROJECT}" --name "${NAME}")
(( ${#PREPARE_ARGS[@]} == 0 )) || PREPARE_COMMAND+=("${PREPARE_ARGS[@]}")
PREPARE_COMMAND+=("${CLIPS[@]}")

COLMAP_COMMAND=(bash "${SCRIPT_DIR}/run_object_colmap.sh" --dataset "${DATASET}" --gpu "${GPU_ID}"
    --camera-model "${CAMERA_MODEL}" --vocab-tree-num-images "${VOCAB_TREE_NUM_IMAGES}"
    --feature-max-num-features "${FEATURE_MAX_NUM_FEATURES}")
if [[ "${BA_PRESET}" == "fast" ]]; then
    COLMAP_COMMAND+=(--ba-global-frames-ratio 1.3 --ba-global-points-ratio 1.3
        --ba-global-max-refinements 2 --ba-global-max-num-iterations 30 --colmap-threads 64)
fi
(( ${#COLMAP_ARGS[@]} == 0 )) || COLMAP_COMMAND+=("${COLMAP_ARGS[@]}")

TRAIN_COMMAND=(env CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    "${PROJECT_PYTHON}" "${SCRIPT_DIR}/train_experiment.py" mcmc
    --data-dir "${DATASET}" --data-factor 1 --result-dir "${RESULT_DIR}"
    --strategy.cap-max "${CAP_MAX}" --steps-scaler "${STEPS_SCALER}" --app-opt
    --strategy.refine-stop-iter "${REFINE_STOP}" --opacity-reg 0.0005
    --max-steps "${MAX_STEPS}" --eval-steps "${MAX_STEPS}" --save-steps "${MAX_STEPS}" --ply-steps "${MAX_STEPS}"
    --save-ply --disable-viewer)

print_command() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run() { print_command "$@"; if (( DRY_RUN == 0 )); then "$@"; fi; }
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

echo "Object pipeline: project=${PROJECT} name=${NAME} gpu=${GPU_ID} stages=${STAGES}"
echo "  dataset:  ${DATASET}"
echo "  result:   ${RESULT_DIR} (${TRAIN_STEPS} steps, final step ${FINAL_STEP})"
echo "  colmap:   ${CAMERA_MODEL}, vocab ${VOCAB_TREE_NUM_IMAGES} neighbors, ${FEATURE_MAX_NUM_FEATURES} features, BA preset ${BA_PRESET}"
if stage_selected frames; then
    log "stage frames (${#CLIPS[@]} clips)"
    run "${PREPARE_COMMAND[@]}"
fi
if stage_selected colmap; then
    log "stage colmap"
    if (( DRY_RUN == 0 )) && [[ ! -d "${DATASET}/images" ]]; then
        echo "error: dataset not found: ${DATASET} (run the frames stage first)" >&2; exit 1
    fi
    run "${COLMAP_COMMAND[@]}"
fi
if stage_selected train; then
    log "stage train"
    if (( DRY_RUN == 0 )); then
        for f in cameras.bin images.bin points3D.bin; do
            [[ -s "${DATASET}/sparse/0/${f}" ]] || { echo "error: COLMAP model incomplete: ${DATASET}/sparse/0/${f}" >&2; exit 1; }
        done
        if [[ -e "${RESULT_DIR}" ]]; then
            echo "error: result directory exists: ${RESULT_DIR} (choose another --tag)" >&2; exit 1
        fi
        mkdir -p "${TRAIN_LOG_DIR}"
        print_command "${TRAIN_COMMAND[@]}"
        echo "=== $(date -u +%FT%TZ) run_object_pipeline.sh train ${NAME}_${TAG}" >> "${TRAIN_LOG}"
        set +e
        "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${TRAIN_LOG}"
        status="${PIPESTATUS[0]}"
        set -e
        (( status == 0 )) || { echo "training failed (exit ${status}); log: ${TRAIN_LOG}" >&2; exit "${status}"; }
        for f in "ckpts/ckpt_${FINAL_STEP}_rank0.pt" "ply/point_cloud_${FINAL_STEP}.ply" "stats/val_step${FINAL_STEP}.json"; do
            [[ -s "${RESULT_DIR}/${f}" ]] || { echo "error: training finished without ${RESULT_DIR}/${f}" >&2; exit 1; }
        done
        log "training complete: ${RESULT_DIR}"
        echo "View: ${TRAIN_COMMAND[*]:3} --ckpt ${RESULT_DIR}/ckpts/ckpt_${FINAL_STEP}_rank0.pt --view-only --port 8080  (drop --disable-viewer)"
    else
        print_command "${TRAIN_COMMAND[@]}"
    fi
fi
(( DRY_RUN == 0 )) || echo "Dry run complete; nothing was written."
