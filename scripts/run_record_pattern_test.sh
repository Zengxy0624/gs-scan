#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"
TRAINER="${GSPLAT_ROOT}/examples/simple_trainer.py"
PROJECT_PYTHON="${PROJECT_PYTHON:-${GSPLAT_ROOT}/.venv/bin/python}"

CAPTURE_ROOT="${CAPTURE_ROOT:-${PROJECT_ROOT}/captures/record_pattern_test}"
WORK_BASE="${WORK_ROOT:-${PROJECT_ROOT}/work/record_pattern_test}"
OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/record_pattern_test}"

PROFILE="full"
GPU_INWARD="6"
GPU_COMBINED="7"
SAMPLE_FPS="6"
MAX_FRAMES="100000"
OVERLAP="15"
COLMAP_THREADS="8"
MAX_EXHAUSTIVE_PAIRS="5000000"
MIN_REGISTRATION_RATIO="0.8"
MIN_SOURCE_REGISTRATION_RATIO="0.8"
CAMERA_MODEL="OPENCV"
RETRY_INCOMPLETE=0
PREPROCESS_ONLY=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Run two record-pattern reconstructions concurrently:
  inward_orbit           -> GPU 6, sequential COLMAP
  inward_orbit_upward    -> GPU 7, exhaustive COLMAP over both videos

Usage:
  bash scripts/run_record_pattern_test.sh [options]

Options:
  --profile pilot|full       Train for 7,000 or 30,000 steps (default: full).
  --gpu-inward ID            GPU for inward_orbit training (default: 6).
  --gpu-combined ID          GPU for combined training (default: 7).
  --sample-fps FPS           Sampling rate for each source (default: 6).
  --max-frames N             Per-video sampled-frame cap (default: 100000).
  --overlap N                Inward quadratic-neighbor levels (default: 15).
  --colmap-threads N         CPU threads per concurrent COLMAP job (default: 8).
  --max-exhaustive-pairs N   Safety cap for the combined matcher (default: 5000000).
  --min-registration-ratio R Overall COLMAP registration gate (default: 0.8).
  --min-source-registration-ratio R
                              Combined per-video gate (default: 0.8).
  --preprocess-only          Stop each job after COLMAP.
  --retry-incomplete         Archive stale/incomplete state and retry it.
  --dry-run                  Print both job plans without writing or running.
  -h, --help                 Show this help.

Environment overrides:
  CAPTURE_ROOT   Input directory (default: captures/record_pattern_test).
  WORK_ROOT      Frame/COLMAP base (default: work/record_pattern_test).
  OUTPUT_ROOT    Durable result base (default: outputs/record_pattern_test).

This script does not load modules, install packages, or alter the shared
environment. Run scripts/setup_gsplat.sh separately before the first job.
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
        --gpu-inward)
            require_value "$1" "${2:-}"
            GPU_INWARD="$2"
            shift 2
            ;;
        --gpu-combined)
            require_value "$1" "${2:-}"
            GPU_COMBINED="$2"
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
        --max-exhaustive-pairs)
            require_value "$1" "${2:-}"
            MAX_EXHAUSTIVE_PAIRS="$2"
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

normalize_decimal() {
    local raw="$1"
    local integer_part="${raw%%.*}"
    local fractional_part=""
    integer_part="$((10#${integer_part}))"
    if [[ "${raw}" == *.* ]]; then
        fractional_part="${raw#*.}"
        while [[ "${fractional_part}" == *0 ]]; do
            fractional_part="${fractional_part%0}"
        done
    fi
    if [[ -n "${fractional_part}" ]]; then
        printf '%s.%s\n' "${integer_part}" "${fractional_part}"
    else
        printf '%s\n' "${integer_part}"
    fi
}

validate_gpu() {
    local value="$1"
    local option="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "error: ${option} must be one non-negative integer GPU index" >&2
        exit 2
    fi
}

validate_gpu "${GPU_INWARD}" "--gpu-inward"
validate_gpu "${GPU_COMBINED}" "--gpu-combined"
GPU_INWARD="$((10#${GPU_INWARD}))"
GPU_COMBINED="$((10#${GPU_COMBINED}))"
if [[ "${GPU_INWARD}" == "${GPU_COMBINED}" ]]; then
    echo "error: the two jobs require distinct GPU indices" >&2
    exit 2
fi
if [[ ! "${SAMPLE_FPS}" =~ ^([0-9]+)([.][0-9]+)?$ ]] ||
    [[ "${SAMPLE_FPS}" =~ ^0+([.]0+)?$ ]]; then
    echo "error: --sample-fps must be positive" >&2
    exit 2
fi
SAMPLE_FPS="$(normalize_decimal "${SAMPLE_FPS}")"
if [[ ! "${MAX_FRAMES}" =~ ^[0-9]+$ ]]; then
    echo "error: --max-frames must be an integer of at least 2" >&2
    exit 2
fi
MAX_FRAMES="$((10#${MAX_FRAMES}))"
if (( MAX_FRAMES < 2 )); then
    echo "error: --max-frames must be an integer of at least 2" >&2
    exit 2
fi
if [[ ! "${OVERLAP}" =~ ^[0-9]+$ ]]; then
    echo "error: --overlap must be a positive integer" >&2
    exit 2
fi
OVERLAP="$((10#${OVERLAP}))"
if (( OVERLAP < 1 )); then
    echo "error: --overlap must be a positive integer" >&2
    exit 2
fi
if [[ ! "${COLMAP_THREADS}" =~ ^[0-9]+$ ]]; then
    echo "error: --colmap-threads must be a positive integer per job" >&2
    exit 2
fi
COLMAP_THREADS="$((10#${COLMAP_THREADS}))"
if (( COLMAP_THREADS < 1 )); then
    echo "error: --colmap-threads must be a positive integer per job" >&2
    exit 2
fi
if [[ ! "${MAX_EXHAUSTIVE_PAIRS}" =~ ^[0-9]+$ ]]; then
    echo "error: --max-exhaustive-pairs must be a positive integer" >&2
    exit 2
fi
MAX_EXHAUSTIVE_PAIRS="$((10#${MAX_EXHAUSTIVE_PAIRS}))"
if (( MAX_EXHAUSTIVE_PAIRS < 1 )); then
    echo "error: --max-exhaustive-pairs must be a positive integer" >&2
    exit 2
fi
validate_ratio() {
    local value="$1"
    local option="$2"
    if [[ ! "${value}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] ||
        [[ "${value}" =~ ^0([.]0+)?$ ]]; then
        echo "error: ${option} must be greater than 0 and at most 1" >&2
        exit 2
    fi
}
validate_ratio "${MIN_REGISTRATION_RATIO}" "--min-registration-ratio"
validate_ratio \
    "${MIN_SOURCE_REGISTRATION_RATIO}" \
    "--min-source-registration-ratio"
MIN_REGISTRATION_RATIO="$(normalize_decimal "${MIN_REGISTRATION_RATIO}")"
MIN_SOURCE_REGISTRATION_RATIO="$(
    normalize_decimal "${MIN_SOURCE_REGISTRATION_RATIO}"
)"

INWARD_VIDEO="${CAPTURE_ROOT}/inward_orbit.mov"
UPWARD_VIDEO="${CAPTURE_ROOT}/upward.mov"
CAMERA_MODEL_KEY="$(printf '%s' "${CAMERA_MODEL}" | tr '[:upper:]' '[:lower:]')"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCK_DIRS=("${WORK_BASE}/.batch-lock" "${OUTPUT_BASE}/.batch-lock")
ACQUIRED_LOCKS=()
CHILD_PIDS=()
PRESERVE_LOCKS=0

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
for capture in "${INWARD_VIDEO}" "${UPWARD_VIDEO}"; do
    if [[ ! -f "${capture}" ]] || [[ ! -r "${capture}" ]]; then
        echo "error: capture is missing or unreadable: ${capture}" >&2
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
    for gpu_id in "${GPU_INWARD}" "${GPU_COMBINED}"; do
        if ! nvidia-smi -i "${gpu_id}" >/dev/null; then
            echo "error: GPU ${gpu_id} is not visible to nvidia-smi" >&2
            exit 1
        fi
    done
fi

cleanup_locks() {
    local lock_dir=""
    local owner_pid=""
    if (( ${#ACQUIRED_LOCKS[@]} == 0 )); then
        return 0
    fi
    if (( PRESERVE_LOCKS == 1 )); then
        for lock_dir in "${ACQUIRED_LOCKS[@]}"; do
            owner_pid=""
            if [[ -f "${lock_dir}/pid" ]]; then
                IFS= read -r owner_pid < "${lock_dir}/pid" || true
            fi
            if [[ "${owner_pid}" == "$$" ]]; then
                printf '%s\n' "cleanup-incomplete" > "${lock_dir}/pid"
            fi
        done
        echo "error: child cleanup could not be verified; locks were retained" >&2
        return 0
    fi
    for lock_dir in "${ACQUIRED_LOCKS[@]}"; do
        owner_pid=""
        if [[ -f "${lock_dir}/pid" ]]; then
            IFS= read -r owner_pid < "${lock_dir}/pid" || true
        fi
        if [[ "${owner_pid}" == "$$" ]]; then
            rm -f \
                "${lock_dir}/pid" \
                "${lock_dir}/hostname" \
                "${lock_dir}/uid" \
                "${lock_dir}/pgids" \
                "${lock_dir}/launch-in-progress" \
                "${lock_dir}/.pgids.tmp-$$"
            rmdir "${lock_dir}" 2>/dev/null || true
        fi
    done
}

process_is_alive() {
    local process_id="$1"
    "${PROJECT_PYTHON}" - "${process_id}" <<'PY'
import errno
import os
import sys

try:
    os.kill(int(sys.argv[1]), 0)
except ProcessLookupError:
    raise SystemExit(1)
except PermissionError:
    raise SystemExit(0)
except OSError as error:
    raise SystemExit(1 if error.errno == errno.ESRCH else 0)
raise SystemExit(0)
PY
}

recorded_process_group_is_alive() {
    local group_id="$1"
    "${PROJECT_PYTHON}" - "${group_id}" <<'PY'
import errno
import os
import sys

try:
    os.killpg(int(sys.argv[1]), 0)
except ProcessLookupError:
    raise SystemExit(1)
except PermissionError:
    raise SystemExit(0)
except OSError as error:
    raise SystemExit(1 if error.errno == errno.ESRCH else 0)
raise SystemExit(0)
PY
}

acquire_one_lock() {
    local lock_dir="$1"
    local current_host="$2"
    local current_uid="$3"
    local stale_lock=""
    local owner_host=""
    local owner_pid=""
    local owner_pgids=()
    local owner_uid=""
    local recorded_group=""

    mkdir -p "$(dirname -- "${lock_dir}")"
    if ! mkdir "${lock_dir}" 2>/dev/null; then
        if [[ ! -d "${lock_dir}" ]]; then
            echo "error: lock path exists and is not a directory: ${lock_dir}" >&2
            exit 1
        fi
        [[ -f "${lock_dir}/hostname" ]] &&
            IFS= read -r owner_host < "${lock_dir}/hostname" || true
        [[ -f "${lock_dir}/pid" ]] &&
            IFS= read -r owner_pid < "${lock_dir}/pid" || true
        [[ -f "${lock_dir}/uid" ]] &&
            IFS= read -r owner_uid < "${lock_dir}/uid" || true
        if [[ -f "${lock_dir}/pgids" ]]; then
            while IFS= read -r recorded_group; do
                [[ -z "${recorded_group}" ]] && continue
                if [[ ! "${recorded_group}" =~ ^[0-9]+$ ]]; then
                    echo "error: batch lock process-group metadata is invalid: ${lock_dir}" >&2
                    exit 1
                fi
                owner_pgids+=("${recorded_group}")
            done < "${lock_dir}/pgids"
        fi
        if [[ -z "${owner_host}" ]] || [[ ! "${owner_pid}" =~ ^[0-9]+$ ]] ||
            [[ ! "${owner_uid}" =~ ^[0-9]+$ ]] ||
            [[ ! -f "${lock_dir}/pgids" ]]; then
            echo "error: batch lock metadata is incomplete: ${lock_dir}" >&2
            echo "Verify that no batch is active before removing it manually." >&2
            exit 1
        fi
        if [[ "${owner_host}" != "${current_host}" ]]; then
            echo "error: batch lock belongs to host ${owner_host}: ${lock_dir}" >&2
            exit 1
        fi
        if process_is_alive "${owner_pid}"; then
            if [[ "${owner_uid}" != "${current_uid}" ]]; then
                echo "error: another user's batch is active as UID ${owner_uid}, PID ${owner_pid}: ${lock_dir}" >&2
            else
                echo "error: another batch is active as PID ${owner_pid}: ${lock_dir}" >&2
            fi
            exit 1
        fi
        if [[ -e "${lock_dir}/launch-in-progress" ]]; then
            echo "error: dead batch owner left an incomplete process-group launch: ${lock_dir}" >&2
            echo "Inspect server processes before removing the lock manually." >&2
            exit 1
        fi
        if (( ${#owner_pgids[@]} > 0 )); then
            for recorded_group in "${owner_pgids[@]}"; do
                if recorded_process_group_is_alive "${recorded_group}"; then
                    echo "error: orphaned batch process group ${recorded_group} is still active: ${lock_dir}" >&2
                    echo "Inspect that process group before removing the lock." >&2
                    exit 1
                fi
            done
        fi
        stale_lock="${lock_dir}.stale-${RUN_TIMESTAMP}-$$"
        echo "Archiving stale batch lock: ${lock_dir} -> ${stale_lock}"
        mv "${lock_dir}" "${stale_lock}"
        mkdir "${lock_dir}"
    fi
    : > "${lock_dir}/pgids"
    printf '%s\n' "$$" > "${lock_dir}/launch-in-progress"
    printf '%s\n' "$$" > "${lock_dir}/pid"
    ACQUIRED_LOCKS+=("${lock_dir}")
    printf '%s\n' "${current_host}" > "${lock_dir}/hostname"
    printf '%s\n' "${current_uid}" > "${lock_dir}/uid"
}

acquire_locks() {
    local current_host=""
    local current_uid=""
    local lock_dir=""
    local previous_lock=""
    current_host="$(hostname)"
    current_uid="$(id -u)"
    mkdir -p "${WORK_BASE}" "${OUTPUT_BASE}"
    for lock_dir in "${LOCK_DIRS[@]}"; do
        if [[ "${lock_dir}" == "${previous_lock}" ]]; then
            continue
        fi
        acquire_one_lock "${lock_dir}" "${current_host}" "${current_uid}"
        previous_lock="${lock_dir}"
    done
}

write_active_process_groups() {
    local child_pid=""
    local lock_dir=""
    local temporary=""
    if (( DRY_RUN == 1 )) || (( ${#ACQUIRED_LOCKS[@]} == 0 )); then
        return 0
    fi
    for lock_dir in "${ACQUIRED_LOCKS[@]}"; do
        temporary="${lock_dir}/.pgids.tmp-$$"
        : > "${temporary}"
        if (( ${#CHILD_PIDS[@]} > 0 )); then
            for child_pid in "${CHILD_PIDS[@]}"; do
                [[ "${child_pid}" =~ ^[0-9]+$ ]] || continue
                printf '%s\n' "${child_pid}" >> "${temporary}"
            done
        fi
        mv "${temporary}" "${lock_dir}/pgids"
    done
}

mark_launch_in_progress() {
    local lock_dir=""
    if (( DRY_RUN == 1 )); then
        return 0
    fi
    for lock_dir in "${ACQUIRED_LOCKS[@]}"; do
        printf '%s\n' "$$" > "${lock_dir}/launch-in-progress"
    done
}

mark_launch_recorded() {
    local lock_dir=""
    if (( DRY_RUN == 1 )); then
        return 0
    fi
    for lock_dir in "${ACQUIRED_LOCKS[@]}"; do
        rm -f "${lock_dir}/launch-in-progress"
    done
}

process_group_is_alive() {
    local group_id="$1"
    kill -0 -- "-${group_id}" 2>/dev/null
}

stop_children() {
    local active_groups=()
    local attempt=0
    local child_pid=""
    local groups_alive=0
    if (( ${#CHILD_PIDS[@]} == 0 )); then
        return 0
    fi
    for child_pid in "${CHILD_PIDS[@]}"; do
        [[ "${child_pid}" =~ ^[0-9]+$ ]] || continue
        if process_group_is_alive "${child_pid}"; then
            active_groups+=("${child_pid}")
            kill -TERM -- "-${child_pid}" 2>/dev/null || true
        fi
    done
    if (( ${#active_groups[@]} == 0 )); then
        CHILD_PIDS=()
        return 0
    fi
    for child_pid in "${active_groups[@]}"; do
        wait "${child_pid}" 2>/dev/null || true
    done
    for ((attempt = 0; attempt < 50; attempt++)); do
        groups_alive=0
        for child_pid in "${active_groups[@]}"; do
            if process_group_is_alive "${child_pid}"; then
                groups_alive=1
                break
            fi
        done
        (( groups_alive == 1 )) || break
        sleep 0.1
    done
    if (( groups_alive == 1 )); then
        for child_pid in "${active_groups[@]}"; do
            kill -KILL -- "-${child_pid}" 2>/dev/null || true
        done
        for ((attempt = 0; attempt < 10; attempt++)); do
            groups_alive=0
            for child_pid in "${active_groups[@]}"; do
                if process_group_is_alive "${child_pid}"; then
                    groups_alive=1
                    break
                fi
            done
            (( groups_alive == 1 )) || break
            sleep 0.1
        done
    fi
    if (( groups_alive == 1 )); then
        PRESERVE_LOCKS=1
    fi
    CHILD_PIDS=()
}

on_exit() {
    local exit_code="$?"
    trap - EXIT HUP INT TERM
    stop_children
    cleanup_locks
    exit "${exit_code}"
}

trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if (( DRY_RUN == 0 )); then
    acquire_locks
fi

print_command() {
    local argument=""
    local command_line="+"
    local quoted=""
    for argument in "$@"; do
        printf -v quoted '%q' "${argument}"
        command_line+=" ${quoted}"
    done
    printf '%s\n' "${command_line}"
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

has_colmap_state() {
    local dataset="$1"
    [[ -e "${dataset}/database.db" ]] ||
        [[ -e "${dataset}/sparse" ]] ||
        [[ -e "${dataset}/reconstruction_stats.json" ]]
}

sparse_model_complete() {
    local model_path="$1"
    [[ -s "${model_path}/cameras.bin" ]] &&
        [[ -s "${model_path}/images.bin" ]] &&
        [[ -s "${model_path}/points3D.bin" ]]
}

report_exhaustive_scale() {
    local dataset="$1"
    local image_count=""
    local pair_count=""
    if (( DRY_RUN == 1 )) || [[ ! -d "${dataset}/images" ]]; then
        return 0
    fi
    image_count="$(find "${dataset}/images" -type f -print | awk 'END {print NR}')"
    if [[ ! "${image_count}" =~ ^[0-9]+$ ]]; then
        echo "warning: could not count images for exhaustive matching" >&2
        return 0
    fi
    pair_count=$((image_count * (image_count - 1) / 2))
    echo "Exhaustive matching scale: ${image_count} images, ${pair_count} candidate pairs"
    if (( pair_count > 1000000 )); then
        echo "warning: exhaustive CPU matching may take a long time at this scale" >&2
    fi
    if (( pair_count > MAX_EXHAUSTIVE_PAIRS )); then
        echo "error: candidate-pair count exceeds the safety cap of ${MAX_EXHAUSTIVE_PAIRS}" >&2
        echo "Lower --sample-fps or explicitly raise --max-exhaustive-pairs." >&2
        return 1
    fi
}

cache_matches() {
    local dataset="$1"
    local matching_strategy="$2"
    local require_colmap="$3"
    local min_source_ratio="$4"
    shift 4
    local sources=("$@")
    local validation_command=(
        "${PROJECT_PYTHON}"
        "${PROJECT_ROOT}/scripts/validate_dataset_cache.py"
        "${dataset}"
        "${sources[@]}"
        --sample-fps "${SAMPLE_FPS}"
        --max-frames "${MAX_FRAMES}"
        --camera-model "${CAMERA_MODEL}"
        --matching-strategy "${matching_strategy}"
        --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
    )
    if [[ "${matching_strategy}" == "sequential" ]]; then
        validation_command+=(
            --overlap "${OVERLAP}"
            --quadratic-overlap
        )
    fi
    if [[ -n "${min_source_ratio}" ]]; then
        validation_command+=(
            --min-source-registration-ratio "${min_source_ratio}"
        )
    fi
    if (( require_colmap == 1 )); then
        validation_command+=(--require-colmap)
    fi
    "${validation_command[@]}"
}

colmap_complete() {
    local dataset="$1"
    local matching_strategy="$2"
    local min_source_ratio="$3"
    shift 3
    local sources=("$@")
    [[ -s "${dataset}/database.db" ]] &&
        sparse_model_complete "${dataset}/sparse/0" &&
        cache_matches \
            "${dataset}" "${matching_strategy}" 1 "${min_source_ratio}" \
            "${sources[@]}"
}

archive_path() {
    local path="$1"
    local archive_base="$2"
    local label="$3"
    local destination="${archive_base}/${label}-${RUN_TIMESTAMP}-$$"
    mkdir -p "${archive_base}"
    echo "Archiving stale path: ${path} -> ${destination}"
    mv "${path}" "${destination}"
}

handle_stale_work() {
    local scene="$1"
    local scene_work="$2"
    local reason="$3"
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: ${reason} for ${scene}: ${scene_work}" >&2
        echo "Rerun with --retry-incomplete to archive and recreate it." >&2
        return 1
    fi
    archive_path "${scene_work}" "${WORK_BASE}/${scene}/_failed" "work"
}

prepare_work_directory() {
    local scene="$1"
    local scene_work="$2"
    local matching_strategy="$3"
    local min_source_ratio="$4"
    shift 4
    local sources=("$@")
    local dataset="${scene_work}/dataset"

    if (( DRY_RUN == 1 )) || [[ ! -e "${scene_work}" ]]; then
        return 0
    fi
    if ! extraction_structurally_complete "${dataset}"; then
        handle_stale_work "${scene}" "${scene_work}" "incomplete extraction"
        return $?
    fi
    if has_colmap_state "${dataset}"; then
        if ! colmap_complete \
            "${dataset}" "${matching_strategy}" "${min_source_ratio}" \
            "${sources[@]}"; then
            handle_stale_work \
                "${scene}" "${scene_work}" "stale or incomplete COLMAP state"
            return $?
        fi
    elif ! cache_matches \
        "${dataset}" "${matching_strategy}" 0 "${min_source_ratio}" \
        "${sources[@]}"; then
        handle_stale_work "${scene}" "${scene_work}" "stale frame cache"
        return $?
    fi
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
    local strict_snapshot="${3:-0}"
    local compare_command=(
        "${PROJECT_PYTHON}"
        "${PROJECT_ROOT}/scripts/build_run_manifest.py"
        compare
        "${desired_manifest}"
        "${result_dir}/pipeline_manifest.json"
    )
    if (( strict_snapshot == 1 )); then
        compare_command+=(--strict-snapshot)
    fi
    training_artifacts_complete "${result_dir}" &&
        [[ -s "${result_dir}/pipeline_manifest.json" ]] &&
        "${compare_command[@]}"
}

prepare_output_directory() {
    local scene="$1"
    local result_dir="$2"
    local desired_manifest="$3"
    if (( DRY_RUN == 1 )) || [[ ! -e "${result_dir}" ]] ||
        training_complete "${result_dir}" "${desired_manifest}"; then
        return 0
    fi
    if (( RETRY_INCOMPLETE == 0 )); then
        echo "error: incomplete or stale result for ${scene}: ${result_dir}" >&2
        echo "Rerun with --retry-incomplete to archive and retrain it." >&2
        return 1
    fi
    archive_path \
        "${result_dir}" "${OUTPUT_BASE}/${scene}/_failed" "result"
}

process_scene() {
    local scene="$1"
    local matching_strategy="$2"
    local gpu_id="$3"
    local min_source_ratio="$4"
    shift 4
    local sources=("$@")
    local preprocess_key=""
    local run_key=""
    local scene_work=""
    local dataset=""
    local result_dir=""
    local desired_manifest=""
    local extraction_command=()
    local reconstruction_command=()
    local manifest_command=()
    local trainer_command=()

    if [[ "${matching_strategy}" == "sequential" ]]; then
        printf -v preprocess_key \
            'fps%s_max%s_cam%s_sequential%s_reg%s' \
            "${SAMPLE_FPS}" "${MAX_FRAMES}" "${CAMERA_MODEL_KEY}" \
            "${OVERLAP}" "${MIN_REGISTRATION_RATIO}"
    else
        printf -v preprocess_key \
            'fps%s_max%s_cam%s_exhaustive_reg%s_source%s' \
            "${SAMPLE_FPS}" "${MAX_FRAMES}" "${CAMERA_MODEL_KEY}" \
            "${MIN_REGISTRATION_RATIO}" "${min_source_ratio}"
    fi
    run_key="${preprocess_key}_${MAX_STEPS}steps"
    scene_work="${WORK_BASE}/${scene}/${preprocess_key}"
    dataset="${scene_work}/dataset"
    result_dir="${OUTPUT_BASE}/${scene}/${run_key}"
    desired_manifest="${scene_work}/training_manifest_${MAX_STEPS}.json"

    echo
    echo "=== ${scene} ==="
    echo "Sources: ${sources[*]}"
    echo "COLMAP: ${matching_strategy}, ${COLMAP_THREADS} CPU threads"
    if (( PREPROCESS_ONLY == 0 )); then
        echo "Training GPU: ${gpu_id}"
    fi

    for source in "${sources[@]}"; do
        if [[ ! -f "${source}" ]]; then
            echo "error: capture not found: ${source}" >&2
            return 1
        fi
    done

    prepare_work_directory \
        "${scene}" "${scene_work}" "${matching_strategy}" \
        "${min_source_ratio}" "${sources[@]}" || return 1

    if ! extraction_structurally_complete "${dataset}"; then
        extraction_command=(
            uv run --project "${GSPLAT_ROOT}" --no-sync
            python "${PROJECT_ROOT}/scripts/extract_frames.py"
            "${sources[0]}" "${dataset}"
        )
        if (( ${#sources[@]} > 1 )); then
            for source in "${sources[@]:1}"; do
                extraction_command+=(--additional-video "${source}")
            done
        fi
        extraction_command+=(
            --sample-fps "${SAMPLE_FPS}"
            --max-frames "${MAX_FRAMES}"
        )
        run_command "${extraction_command[@]}" || return 1
    else
        echo "Skipping completed extraction: ${dataset}"
    fi

    if ! colmap_complete \
        "${dataset}" "${matching_strategy}" "${min_source_ratio}" \
        "${sources[@]}"; then
        if [[ "${matching_strategy}" == "exhaustive" ]]; then
            report_exhaustive_scale "${dataset}" || return 1
        fi
        reconstruction_command=(
            uv run --project "${GSPLAT_ROOT}" --no-sync
            python "${PROJECT_ROOT}/scripts/prepare_colmap.py"
            "${dataset}"
            --matching "${matching_strategy}"
            --overlap "${OVERLAP}"
            --camera-model "${CAMERA_MODEL}"
            --num-threads "${COLMAP_THREADS}"
            --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
        )
        if [[ -n "${min_source_ratio}" ]]; then
            reconstruction_command+=(
                --min-source-registration-ratio "${min_source_ratio}"
            )
        fi
        run_command "${reconstruction_command[@]}" || return 1
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
        build
        --scene "${scene}"
        --dataset "${dataset}"
    )
    for source in "${sources[@]}"; do
        manifest_command+=(--source "${source}")
    done
    manifest_command+=(
        --gsplat-root "${GSPLAT_ROOT}"
        --sample-fps "${SAMPLE_FPS}"
        --max-frames "${MAX_FRAMES}"
        --overlap "${OVERLAP}"
        --matching-strategy "${matching_strategy}"
        --min-registration-ratio "${MIN_REGISTRATION_RATIO}"
        --max-steps "${MAX_STEPS}"
        --camera-model "${CAMERA_MODEL}"
        --colmap-threads "${COLMAP_THREADS}"
        --gpu-id "${gpu_id}"
        --output "${desired_manifest}"
    )
    if [[ -n "${min_source_ratio}" ]]; then
        manifest_command+=(
            --min-source-registration-ratio "${min_source_ratio}"
        )
    fi
    run_command "${manifest_command[@]}" || return 1

    prepare_output_directory \
        "${scene}" "${result_dir}" "${desired_manifest}" || return 1
    if (( DRY_RUN == 0 )) &&
        training_complete "${result_dir}" "${desired_manifest}"; then
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
        CUDA_VISIBLE_DEVICES="${gpu_id}"
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

    if (( DRY_RUN == 0 )); then
        run_command "${manifest_command[@]}" || return 1
        if ! training_complete "${result_dir}" "${desired_manifest}" 1; then
            echo "error: training finished without matching final artifacts" >&2
            return 1
        fi
    fi
}

SCENES=("inward_orbit" "inward_orbit_upward")
LOG_PATHS=("" "")
STATUSES=(1 1)

launch_scene() {
    local index="$1"
    local scene="${SCENES[index]}"
    local log_path="${OUTPUT_BASE}/${scene}/_logs/${RUN_TIMESTAMP}.log"
    LOG_PATHS[index]="${log_path}"
    mark_launch_in_progress
    if (( DRY_RUN == 1 )); then
        if (( index == 0 )); then
            (process_scene "${scene}" sequential "${GPU_INWARD}" "" \
                "${INWARD_VIDEO}") &
        else
            (process_scene "${scene}" exhaustive "${GPU_COMBINED}" \
                "${MIN_SOURCE_REGISTRATION_RATIO}" \
                "${INWARD_VIDEO}" "${UPWARD_VIDEO}") &
        fi
    else
        mkdir -p "$(dirname -- "${log_path}")"
        if (( index == 0 )); then
            (
                set -Eeuo pipefail
                process_scene "${scene}" sequential "${GPU_INWARD}" "" \
                    "${INWARD_VIDEO}" 2>&1 | tee -a "${log_path}"
            ) &
        else
            (
                set -Eeuo pipefail
                process_scene "${scene}" exhaustive "${GPU_COMBINED}" \
                    "${MIN_SOURCE_REGISTRATION_RATIO}" \
                    "${INWARD_VIDEO}" "${UPWARD_VIDEO}" 2>&1 | tee -a "${log_path}"
            ) &
        fi
    fi
    CHILD_PIDS[index]="$!"
    write_active_process_groups
    mark_launch_recorded
    echo "Launched ${scene} as PID ${CHILD_PIDS[index]}"
}

set -m
launch_scene 0
launch_scene 1
set +m

for index in "${!SCENES[@]}"; do
    child_pid="${CHILD_PIDS[index]}"
    set +e
    wait "${child_pid}"
    STATUSES[index]="$?"
    CHILD_PIDS[index]=""
    write_active_process_groups
    set -e
done
CHILD_PIDS=()

successful_scenes=()
failed_scenes=()
for index in "${!SCENES[@]}"; do
    if (( STATUSES[index] == 0 )); then
        successful_scenes+=("${SCENES[index]}")
    else
        failed_scenes+=("${SCENES[index]}")
    fi
done

echo
if (( DRY_RUN == 1 )); then
    echo "Dry-run summary (no files written)"
else
    SUMMARY_PATH="${OUTPUT_BASE}/summary_${RUN_TIMESTAMP}.tsv"
    printf 'scene\tstatus\tlog\n' > "${SUMMARY_PATH}"
    for index in "${!SCENES[@]}"; do
        if (( STATUSES[index] == 0 )); then
            status="success"
        else
            status="failed"
        fi
        printf '%s\t%s\t%s\n' \
            "${SCENES[index]}" "${status}" "${LOG_PATHS[index]}" \
            >> "${SUMMARY_PATH}"
    done
    echo "Run summary: ${SUMMARY_PATH}"
fi
echo "  Passed: ${successful_scenes[*]:-(none)}"
echo "  Failed: ${failed_scenes[*]:-(none)}"

if (( ${#failed_scenes[@]} > 0 )); then
    exit 1
fi
