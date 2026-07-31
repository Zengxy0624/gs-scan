#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"
TRAINER="${GSPLAT_ROOT}/examples/simple_trainer.py"
PROJECT_PYTHON="${PROJECT_PYTHON:-${GSPLAT_ROOT}/.venv/bin/python}"

CAPTURE_ROOT="${CAPTURE_ROOT:-${PROJECT_ROOT}/captures/device_test}"
WORK_BASE="${WORK_ROOT:-${PROJECT_ROOT}/work/device_test}"
OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/device_test}"

PROFILE="full"
GPU_ID="6"
SAMPLE_FPS="6"
MAX_FRAMES="100000"
OVERLAP="15"
COLMAP_THREADS="8"
MIN_REGISTRATION_RATIO="0.5"
ONLY_SCENE=""
RETRY_INCOMPLETE=0
PREPROCESS_ONLY=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Run every device-test capture through frame extraction, COLMAP, and gsplat.

Usage:
  bash scripts/run_device_test.sh [options]

Options:
  --profile pilot|full       Train for 7,000 or 30,000 steps (default: full).
  --gpu ID                   Physical GPU used only for training (default: 6).
  --sample-fps FPS           Uniform video sampling rate (default: 6).
  --max-frames N             Maximum sampled frames per video (default: 100000).
  --overlap N                COLMAP temporal match overlap (default: 15).
  --colmap-threads N         COLMAP CPU threads (default: 8; -1 means all).
  --min-registration-ratio R Fail weak COLMAP models below R (default: 0.5).
  --scene NAME               Run only one named scene.
  --preprocess-only          Stop after COLMAP; do not train gsplat.
  --retry-incomplete         Archive incomplete state and retry it.
  --dry-run                  Print commands without writing or running stages.
  -h, --help                 Show this help.

Environment overrides:
  CAPTURE_ROOT   Input directory (default: captures/device_test).
  WORK_ROOT      Disposable frame/COLMAP base (default: work/device_test).
  OUTPUT_ROOT    Durable gsplat result base (default: outputs/device_test).

The two .insv fisheye streams in the capture directory are intentionally
ignored. This batch processes only the five standard MP4/MOV captures.
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
        --overlap)
            require_value "$1" "${2:-}"
            OVERLAP="$2"
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
        --scene)
            require_value "$1" "${2:-}"
            ONLY_SCENE="$2"
            shift 2
            ;;
        --retry-incomplete)
            RETRY_INCOMPLETE=1
            shift
            ;;
        --preprocess-only)
            PREPROCESS_ONLY=1
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
    pilot)
        MAX_STEPS=7000
        ;;
    full)
        MAX_STEPS=30000
        ;;
    *)
        echo "error: --profile must be 'pilot' or 'full'" >&2
        exit 2
        ;;
esac

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "error: --gpu must be one non-negative integer GPU index" >&2
    exit 2
fi
if [[ ! "${SAMPLE_FPS}" =~ ^([0-9]+)([.][0-9]+)?$ ]] ||
    [[ "${SAMPLE_FPS}" =~ ^0+([.]0+)?$ ]]; then
    echo "error: --sample-fps must be positive" >&2
    exit 2
fi
if [[ ! "${MAX_FRAMES}" =~ ^[0-9]+$ ]] || (( MAX_FRAMES < 2 )); then
    echo "error: --max-frames must be an integer of at least 2" >&2
    exit 2
fi
if [[ ! "${OVERLAP}" =~ ^[0-9]+$ ]] || (( OVERLAP < 1 )); then
    echo "error: --overlap must be a positive integer" >&2
    exit 2
fi
if [[ ! "${COLMAP_THREADS}" =~ ^-?[0-9]+$ ]] ||
    (( COLMAP_THREADS == 0 || COLMAP_THREADS < -1 )); then
    echo "error: --colmap-threads must be -1 or a positive integer" >&2
    exit 2
fi
if [[ ! "${MIN_REGISTRATION_RATIO}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] ||
    [[ "${MIN_REGISTRATION_RATIO}" =~ ^0([.]0+)?$ ]]; then
    echo "error: --min-registration-ratio must be greater than 0 and at most 1" >&2
    exit 2
fi

SCENES=(
    "insta_normal"
    "iphone_18mm"
    "iphone_24mm"
    "osmo_rotate"
    "osmo_stable"
)
SOURCE_FILES=(
    "insta_normal.mp4"
    "iphone_18mm.mov"
    "iphone_24mm.mov"
    "osmo_rotate.MP4"
    "osmo_stable.MP4"
)
CAMERA_MODELS=(
    "SIMPLE_RADIAL"
    "SIMPLE_RADIAL"
    "SIMPLE_RADIAL"
    "SIMPLE_RADIAL"
    "SIMPLE_RADIAL"
)

REQUIRED_FILES=(
    "${PROJECT_ROOT}/scripts/extract_frames.py"
    "${PROJECT_ROOT}/scripts/prepare_colmap.py"
    "${PROJECT_ROOT}/scripts/validate_dataset_cache.py"
    "${PROJECT_ROOT}/scripts/build_run_manifest.py"
    "${TRAINER}"
)
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "error: pipeline file is missing: ${required_file}" >&2
        echo "Run: git submodule update --init --recursive" >&2
        exit 1
    fi
done
if (( DRY_RUN == 0 )) && ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not available on PATH" >&2
    exit 1
fi
if (( DRY_RUN == 0 )) && [[ ! -x "${PROJECT_PYTHON}" ]]; then
    echo "error: gsplat environment is missing; run scripts/setup_gsplat.sh first" >&2
    exit 1
fi
if (( DRY_RUN == 0 && PREPROCESS_ONLY == 0 )); then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "error: nvidia-smi is required for training" >&2
        exit 1
    fi
    if ! nvidia-smi -i "${GPU_ID}" >/dev/null; then
        echo "error: GPU ${GPU_ID} is not visible to nvidia-smi" >&2
        exit 1
    fi
fi

PREPROCESS_KEY="fps${SAMPLE_FPS}_max${MAX_FRAMES}_overlap${OVERLAP}_reg${MIN_REGISTRATION_RATIO}"
WORK_EXPERIMENT="${WORK_BASE}/${PREPROCESS_KEY}"
RUN_KEY="${PREPROCESS_KEY}_${MAX_STEPS}steps"
FAILED_WORK_DIR="${WORK_EXPERIMENT}/_failed"
SUMMARY_PATH="${OUTPUT_BASE}/summary_${RUN_KEY}.tsv"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCK_DIR="${WORK_EXPERIMENT}/.batch-lock"
LOCK_ACQUIRED=0

cleanup_lock() {
    if (( LOCK_ACQUIRED == 1 )) &&
        [[ -f "${LOCK_DIR}/pid" ]] &&
        [[ "$(<"${LOCK_DIR}/pid")" == "$$" ]]; then
        rm -f "${LOCK_DIR}/pid" "${LOCK_DIR}/hostname"
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    fi
}

acquire_lock() {
    local current_host
    local owner_host=""
    local owner_pid=""
    local stale_lock=""
    current_host="$(hostname)"
    mkdir -p "${WORK_EXPERIMENT}" "${OUTPUT_BASE}"
    if [[ -d "${LOCK_DIR}" ]]; then
        [[ -f "${LOCK_DIR}/hostname" ]] &&
            owner_host="$(<"${LOCK_DIR}/hostname")"
        [[ -f "${LOCK_DIR}/pid" ]] && owner_pid="$(<"${LOCK_DIR}/pid")"
        if [[ -n "${owner_host}" ]] && [[ "${owner_host}" != "${current_host}" ]]; then
            echo "error: batch lock belongs to host ${owner_host}: ${LOCK_DIR}" >&2
            exit 1
        fi
        if [[ "${owner_pid}" =~ ^[0-9]+$ ]] && kill -0 "${owner_pid}" 2>/dev/null; then
            echo "error: another batch is active as PID ${owner_pid}: ${LOCK_DIR}" >&2
            exit 1
        fi
        stale_lock="${LOCK_DIR}.stale-${RUN_TIMESTAMP}-$$"
        echo "Archiving stale batch lock: ${LOCK_DIR} -> ${stale_lock}"
        mv "${LOCK_DIR}" "${stale_lock}"
    fi
    mkdir "${LOCK_DIR}"
    printf '%s\n' "$$" > "${LOCK_DIR}/pid"
    printf '%s\n' "${current_host}" > "${LOCK_DIR}/hostname"
    LOCK_ACQUIRED=1
    trap cleanup_lock EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

if (( DRY_RUN == 0 )); then
    acquire_lock
    if [[ ! -f "${SUMMARY_PATH}" ]]; then
        printf 'timestamp\tscene\tstatus\tlog\n' > "${SUMMARY_PATH}"
    fi
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

extraction_structurally_complete() {
    local dataset="$1"
    local first_image=""
    [[ -s "${dataset}/metadata.json" ]] || return 1
    [[ -d "${dataset}/images" ]] || return 1
    first_image="$(find "${dataset}/images" -type f -print -quit)"
    [[ -n "${first_image}" ]]
}

cache_matches_sources() {
    local dataset="$1"
    shift
    extraction_structurally_complete "${dataset}" || return 1
    "${PROJECT_PYTHON}" "${PROJECT_ROOT}/scripts/validate_dataset_cache.py" \
        "${dataset}" "$@" \
        --sample-fps "${SAMPLE_FPS}" \
        --max-frames "${MAX_FRAMES}"
}

stats_passed() {
    local stats_path="$1"
    [[ -s "${stats_path}" ]] &&
        grep -q '"status": "passed"' "${stats_path}"
}

sparse_model_complete() {
    local model_path="$1"
    [[ -s "${model_path}/cameras.bin" ]] &&
        [[ -s "${model_path}/images.bin" ]] &&
        [[ -s "${model_path}/points3D.bin" ]]
}

single_colmap_complete() {
    local dataset="$1"
    [[ -s "${dataset}/database.db" ]] &&
        sparse_model_complete "${dataset}/sparse/0" &&
        stats_passed "${dataset}/reconstruction_stats.json"
}

training_artifacts_complete() {
    local result_dir="$1"
    local final_step=$((MAX_STEPS - 1))
    [[ -s "${result_dir}/ckpts/ckpt_${final_step}_rank0.pt" ]] &&
        [[ -s "${result_dir}/ply/point_cloud_${final_step}.ply" ]] &&
        [[ -s "${result_dir}/stats/val_step${final_step}.json" ]]
}

training_complete() {
    local result_dir="$1"
    local desired_manifest="$2"
    training_artifacts_complete "${result_dir}" &&
        [[ -s "${result_dir}/pipeline_manifest.json" ]] &&
        cmp -s \
            "${desired_manifest}" \
            "${result_dir}/pipeline_manifest.json"
}

has_single_colmap_state() {
    local dataset="$1"
    [[ -e "${dataset}/database.db" ]] ||
        [[ -e "${dataset}/sparse" ]] ||
        [[ -e "${dataset}/reconstruction_stats.json" ]]
}

archive_incomplete() {
    local path="$1"
    local archive_base="$2"
    local label="$3"
    local destination="${archive_base}/${label}-${RUN_TIMESTAMP}-$$"
    mkdir -p "${archive_base}"
    echo "Archiving incomplete path: ${path} -> ${destination}"
    mv "${path}" "${destination}"
}

handle_incomplete_work() {
    local scene="$1"
    local scene_work="$2"
    local reason="$3"
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: ${reason} for ${scene}: ${scene_work}" >&2
        echo "Rerun with --retry-incomplete to archive and recreate it." >&2
        return 1
    fi
    archive_incomplete "${scene_work}" "${FAILED_WORK_DIR}" "${scene}"
}

prepare_work_directory() {
    local scene="$1"
    local scene_work="$2"
    local source_video="$3"
    local dataset="${scene_work}/dataset"

    if (( DRY_RUN == 1 )); then
        return 0
    fi
    if [[ ! -e "${scene_work}" ]]; then
        return 0
    fi
    if ! extraction_structurally_complete "${dataset}"; then
        handle_incomplete_work \
            "${scene}" "${scene_work}" "incomplete extraction"
        return $?
    fi
    if ! cache_matches_sources "${dataset}" "${source_video}"; then
        handle_incomplete_work \
            "${scene}" "${scene_work}" "stale frame cache"
        return $?
    fi
    if has_single_colmap_state "${dataset}" &&
        ! single_colmap_complete "${dataset}"; then
        handle_incomplete_work \
            "${scene}" "${scene_work}" "incomplete COLMAP state"
        return $?
    fi
}

prepare_output_directory() {
    local scene="$1"
    local result_dir="$2"
    local failed_output_dir="$3"
    local desired_manifest="$4"

    if (( DRY_RUN == 1 )); then
        return 0
    fi
    if [[ ! -e "${result_dir}" ]] ||
        training_complete "${result_dir}" "${desired_manifest}"; then
        return 0
    fi
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: incomplete or stale training result for ${scene}: ${result_dir}" >&2
        echo "Its provenance does not match the current source, COLMAP model, or pipeline." >&2
        echo "Rerun with --retry-incomplete to archive and retrain it." >&2
        return 1
    fi
    archive_incomplete "${result_dir}" "${failed_output_dir}" "${RUN_KEY}"
}

process_scene() {
    local scene="$1"
    local source_file="$2"
    local camera_model="$3"
    local source_video="${CAPTURE_ROOT}/${source_file}"
    local scene_work="${WORK_EXPERIMENT}/${scene}"
    local dataset="${scene_work}/dataset"
    local scene_output="${OUTPUT_BASE}/${scene}"
    local result_dir="${scene_output}/${RUN_KEY}"
    local failed_output_dir="${scene_output}/_failed"
    local desired_manifest="${scene_work}/training_manifest_${MAX_STEPS}.json"
    local trainer_command=()
    local manifest_command=()

    echo
    echo "=== ${scene} ==="
    echo "Source: ${source_video}"
    echo "Camera model: ${camera_model}"

    if [[ ! -f "${source_video}" ]]; then
        echo "error: capture not found: ${source_video}" >&2
        return 1
    fi

    prepare_work_directory \
        "${scene}" "${scene_work}" "${source_video}" || return 1

    if ! extraction_structurally_complete "${dataset}"; then
        run_command \
            uv run --project "${GSPLAT_ROOT}" --no-sync \
            python "${PROJECT_ROOT}/scripts/extract_frames.py" \
            "${source_video}" "${dataset}" \
            --sample-fps "${SAMPLE_FPS}" \
            --max-frames "${MAX_FRAMES}" || return 1
    else
        echo "Skipping completed extraction: ${dataset}"
    fi
    if ! single_colmap_complete "${dataset}"; then
        run_command \
            uv run --project "${GSPLAT_ROOT}" --no-sync \
            python "${PROJECT_ROOT}/scripts/prepare_colmap.py" \
            "${dataset}" \
            --overlap "${OVERLAP}" \
            --camera-model "${camera_model}" \
            --num-threads "${COLMAP_THREADS}" \
            --min-registration-ratio \
            "${MIN_REGISTRATION_RATIO}" || return 1
    else
        echo "Skipping completed COLMAP reconstruction: ${dataset}/sparse/0"
    fi

    if (( PREPROCESS_ONLY == 1 )); then
        echo "Preprocessing complete for ${scene}"
        return 0
    fi

    manifest_command=(
        "${PROJECT_PYTHON}"
        "${PROJECT_ROOT}/scripts/build_run_manifest.py"
        --scene "${scene}"
        --dataset "${dataset}"
        --source "${source_video}"
        --pipeline-file "${PROJECT_ROOT}/scripts/extract_frames.py"
        --pipeline-file "${PROJECT_ROOT}/scripts/prepare_colmap.py"
        --pipeline-file "${PROJECT_ROOT}/scripts/validate_dataset_cache.py"
        --pipeline-file "${PROJECT_ROOT}/scripts/build_run_manifest.py"
        --pipeline-file "${PROJECT_ROOT}/scripts/run_device_test.sh"
        --pipeline-file "${TRAINER}"
        --sample-fps "${SAMPLE_FPS}"
        --max-frames "${MAX_FRAMES}"
        --overlap "${OVERLAP}"
        --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
        --max-steps "${MAX_STEPS}"
        --output "${desired_manifest}"
    )
    run_command "${manifest_command[@]}" || return 1

    prepare_output_directory \
        "${scene}" "${result_dir}" "${failed_output_dir}" \
        "${desired_manifest}" || return 1
    if training_complete "${result_dir}" "${desired_manifest}"; then
        echo "Skipping completed training: ${result_dir}"
        return 0
    fi

    run_command mkdir -p "${result_dir}" || return 1
    run_command cp \
        "${desired_manifest}" \
        "${result_dir}/pipeline_manifest.json" || return 1

    trainer_command=(
        env
        CUDA_DEVICE_ORDER=PCI_BUS_ID
        CUDA_VISIBLE_DEVICES="${GPU_ID}"
        uv run
        --project "${GSPLAT_ROOT}"
        --no-sync
        python "${TRAINER}"
        default
        --data-dir "${dataset}"
        --data-factor 1
        --result-dir "${result_dir}"
        --max-steps "${MAX_STEPS}"
        --eval-steps "${MAX_STEPS}"
        --save-steps "${MAX_STEPS}"
        --ply-steps "${MAX_STEPS}"
        --save-ply
        --disable-viewer
    )
    run_command "${trainer_command[@]}" || return 1

    if (( DRY_RUN == 0 )) &&
        ! training_complete "${result_dir}" "${desired_manifest}"; then
        echo "error: training finished without expected final artifacts" >&2
        return 1
    fi
}

selected_count=0
failed_scenes=()
successful_scenes=()

for index in "${!SCENES[@]}"; do
    scene="${SCENES[index]}"
    if [[ -n "${ONLY_SCENE}" ]] && [[ "${scene}" != "${ONLY_SCENE}" ]]; then
        continue
    fi
    selected_count=$((selected_count + 1))
    if (( DRY_RUN == 1 )); then
        set +e
        process_scene \
            "${scene}" \
            "${SOURCE_FILES[index]}" \
            "${CAMERA_MODELS[index]}"
        scene_status=$?
        set -e
    else
        log_path="${OUTPUT_BASE}/${scene}/_logs/${RUN_KEY}.log"
        mkdir -p "$(dirname -- "${log_path}")"
        set +e
        (
            set -Eeuo pipefail
            process_scene \
                "${scene}" \
                "${SOURCE_FILES[index]}" \
                "${CAMERA_MODELS[index]}"
        ) 2>&1 | tee -a "${log_path}"
        scene_status="${PIPESTATUS[0]}"
        set -e
    fi

    if (( scene_status == 0 )); then
        successful_scenes+=("${scene}")
        if (( DRY_RUN == 0 )); then
            printf '%s\t%s\tsuccess\t%s\n' \
                "${RUN_TIMESTAMP}" "${scene}" "${log_path}" >> "${SUMMARY_PATH}"
        fi
    else
        failed_scenes+=("${scene}")
        if (( DRY_RUN == 0 )); then
            printf '%s\t%s\tfailed\t%s\n' \
                "${RUN_TIMESTAMP}" "${scene}" "${log_path}" >> "${SUMMARY_PATH}"
        fi
    fi
done

if (( selected_count == 0 )); then
    echo "error: unknown --scene value: ${ONLY_SCENE}" >&2
    echo "Available scenes: ${SCENES[*]}" >&2
    exit 2
fi

echo
if (( DRY_RUN == 1 )); then
    echo "Dry-run summary (no files written)"
else
    echo "Batch summary"
fi
echo "  Work:    ${WORK_EXPERIMENT}"
echo "  Results: ${OUTPUT_BASE}/<scene>/${RUN_KEY}"
echo "  Passed:  ${successful_scenes[*]:-(none)}"
echo "  Failed:  ${failed_scenes[*]:-(none)}"

if (( ${#failed_scenes[@]} > 0 )); then
    exit 1
fi
