#!/usr/bin/env bash
set -Eeuo pipefail

# Frames for one scanned object, end to end:
#   per clip: extract (3 fps anchors, sharpest frame within ±0.3 s, keep-ratio
#             0.8) -> drop residual blur (ratio_local < 0.7) -> embed the
#             clip's annotation row as EXIF + image_metadata.csv
#   then:     hardlink-merge the clips into work/<root>/<name> (the COLMAP input)
# Idempotent: finished clip caches that validate are reused; the merged dataset
# is only built when absent. Next steps: scripts/run_object_colmap.sh
# --dataset work/<project>/<name>, then scripts/train_experiment.py
# (or run all stages with scripts/run_object_pipeline.sh). docs/OBJECT_PIPELINE.md.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_PYTHON="${PROJECT_PYTHON:-${PROJECT_ROOT}/third-party/gsplat/.venv/bin/python}"

NAME=""
PROJECT=""
CAPTURE_ROOT=""
WORK_ROOT=""
ANNOTATION=""
MAKE=""
MODEL=""
SOFTWARE=""
SAMPLE_FPS="3"
MAX_FRAMES="100000"
SHARPNESS_WINDOW="0.3"
KEEP_RATIO="0.8"
DROP_RATIO="0.7"
SHARPNESS_WORKERS="16"
DRY_RUN=0
CLIPS=()

usage() {
    cat <<'USAGE'
Build the merged frame dataset for one scanned object (blur-aware extraction,
residual-blur filter, EXIF capture metadata, hardlink merge).

Usage:
  bash scripts/prepare_object_frames.sh --project PROJECT --name NAME [options] CLIP [CLIP ...]

  PROJECT is a capture family: videos in captures/PROJECT/, caches and the merged
  dataset in work/PROJECT/ (override either with --capture-root/--work-root).
  CLIP is a video file stem (e.g. JC_13_1341; .mov/.MOV/.mp4/.MP4/.mkv are
  tried). All clips of one object must come from the same camera and lens.
  See docs/OBJECT_PIPELINE.md.

Options:
  --project PROJECT      Capture family (required unless both roots are given).
  --name NAME            Merged dataset name, e.g. mannequin_back (required).
  --capture-root DIR     Source videos (default: captures/PROJECT).
  --work-root DIR        Frame caches + merged dataset (default: work/PROJECT).
  --annotation CSV       Per-video capture annotation CSV (default:
                         <capture-root>/annotation.csv). Needs a 'file' column
                         matching the video file name; optional columns: lens
                         (e.g. 13mm = 35 mm-equivalent), iso, shutter, aperture,
                         white_balance_k, tint, rec_time, gps, annotation,
                         make, model, software.
  --make/--model/--software STR
                         Camera identity for EXIF; override the CSV's make/
                         model/software columns (omitted when neither is given).
  --sample-fps FPS       Anchor rate (default: 3).
  --sharpness-window S   ± seconds searched for a sharper frame (default: 0.3).
  --keep-ratio R         Keep an anchor at >= R x window maximum (default: 0.8).
  --drop-ratio R         Drop images with ratio_local < R after selection
                         (default: 0.7; 0 disables the filter step).
  --sharpness-workers N  Threads scoring sharpness (default: 16).
  --dry-run              Print the commands without writing anything.
  -h, --help             Show this help.

Log: <work-root>/_logs/prepare_<name>.log. Run under tmux.
USAGE
}

require_value() { if [[ -z "${2:-}" ]]; then echo "error: $1 requires a value" >&2; exit 2; fi; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) require_value "$1" "${2:-}"; NAME="$2"; shift 2 ;;
        --project) require_value "$1" "${2:-}"; PROJECT="$2"; shift 2 ;;
        --capture-root) require_value "$1" "${2:-}"; CAPTURE_ROOT="$2"; shift 2 ;;
        --work-root) require_value "$1" "${2:-}"; WORK_ROOT="$2"; shift 2 ;;
        --annotation) require_value "$1" "${2:-}"; ANNOTATION="$2"; shift 2 ;;
        --make) require_value "$1" "${2:-}"; MAKE="$2"; shift 2 ;;
        --model) require_value "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
        --software) require_value "$1" "${2:-}"; SOFTWARE="$2"; shift 2 ;;
        --sample-fps) require_value "$1" "${2:-}"; SAMPLE_FPS="$2"; shift 2 ;;
        --sharpness-window) require_value "$1" "${2:-}"; SHARPNESS_WINDOW="$2"; shift 2 ;;
        --keep-ratio) require_value "$1" "${2:-}"; KEEP_RATIO="$2"; shift 2 ;;
        --drop-ratio) require_value "$1" "${2:-}"; DROP_RATIO="$2"; shift 2 ;;
        --sharpness-workers) require_value "$1" "${2:-}"; SHARPNESS_WORKERS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) CLIPS+=("$1"); shift ;;
    esac
done

decimal() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "error: $2 must be a non-negative number" >&2; exit 2; }; }
if [[ -z "${NAME}" || ! "${NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "error: --name is required and must be a plain directory name" >&2; exit 2
fi
if (( ${#CLIPS[@]} < 2 )); then
    echo "error: give at least two clip stems (a video_set needs two sources)" >&2; exit 2
fi
for clip in "${CLIPS[@]}"; do
    [[ "${clip}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "error: clip must be a plain file stem: ${clip}" >&2; exit 2; }
done
decimal "${SAMPLE_FPS}" "--sample-fps"; decimal "${SHARPNESS_WINDOW}" "--sharpness-window"
decimal "${KEEP_RATIO}" "--keep-ratio"; decimal "${DROP_RATIO}" "--drop-ratio"
[[ "${SHARPNESS_WORKERS}" =~ ^[0-9]+$ ]] && (( 10#${SHARPNESS_WORKERS} >= 1 )) || { echo "error: --sharpness-workers must be a positive integer" >&2; exit 2; }
if [[ -n "${PROJECT}" && ! "${PROJECT}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "error: --project must be a plain directory name" >&2; exit 2
fi
if [[ -z "${CAPTURE_ROOT}" || -z "${WORK_ROOT}" ]]; then
    if [[ -z "${PROJECT}" ]]; then
        echo "error: give --project PROJECT (or both --capture-root and --work-root)" >&2; exit 2
    fi
    [[ -n "${CAPTURE_ROOT}" ]] || CAPTURE_ROOT="${PROJECT_ROOT}/captures/${PROJECT}"
    [[ -n "${WORK_ROOT}" ]] || WORK_ROOT="${PROJECT_ROOT}/work/${PROJECT}"
fi
[[ -n "${ANNOTATION}" ]] || ANNOTATION="${CAPTURE_ROOT}/annotation.csv"
CAPTURE_ROOT="$(realpath -m -- "${CAPTURE_ROOT}")"; WORK_ROOT="$(realpath -m -- "${WORK_ROOT}")"
ANNOTATION="$(realpath -m -- "${ANNOTATION}")"
MERGED="${WORK_ROOT}/${NAME}"

EXTRACTOR="${PROJECT_ROOT}/scripts/extract_frames.py"
FILTER="${PROJECT_ROOT}/scripts/filter_frame_cache.py"
ANNOTATOR="${PROJECT_ROOT}/scripts/annotate_frame_cache.py"
MERGER="${PROJECT_ROOT}/scripts/merge_frame_datasets.py"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_dataset_cache.py"
for f in "${EXTRACTOR}" "${FILTER}" "${ANNOTATOR}" "${MERGER}" "${VALIDATOR}"; do
    [[ -f "${f}" ]] || { echo "error: pipeline file is missing: ${f}" >&2; exit 1; }
done

find_video() {
    local stem="$1" ext
    for ext in mov MOV mp4 MP4 mkv MKV; do
        if [[ -f "${CAPTURE_ROOT}/${stem}.${ext}" ]]; then printf '%s' "${CAPTURE_ROOT}/${stem}.${ext}"; return 0; fi
    done
    printf '%s' "${CAPTURE_ROOT}/${stem}.mov"; return 1
}

print_command() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run() { print_command "$@"; if (( DRY_RUN == 0 )); then "$@"; fi; }
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

validate_cache() {  # cache video -> 0 if it matches this recipe
    "${PROJECT_PYTHON}" "${VALIDATOR}" "$1" "$2" --sample-fps "${SAMPLE_FPS}" --max-frames "${MAX_FRAMES}" \
        --sharpness-window "${SHARPNESS_WINDOW}" --sharpness-keep-ratio "${KEEP_RATIO}" --sharpness-drop-ratio "${DROP_RATIO}" >/dev/null 2>&1
}

prepare_clip() {
    local stem="$1" video cache raw
    if ! video="$(find_video "${stem}")" && (( DRY_RUN == 0 )); then
        echo "error: video not found for ${stem} under ${CAPTURE_ROOT}" >&2; return 1
    fi
    cache="${WORK_ROOT}/${stem}"
    raw="${WORK_ROOT}/.${stem}.unfiltered"
    if [[ -d "${cache}" ]] && (( DRY_RUN == 0 )); then
        if validate_cache "${cache}" "${video}" && [[ -f "${cache}/image_metadata.csv" ]]; then
            log "${stem}: reusing validated cache ${cache}"; return 0
        fi
        echo "error: ${cache} exists but does not match this recipe; move it aside" >&2; return 1
    fi
    log "${stem}: extract (${SAMPLE_FPS} fps, window ${SHARPNESS_WINDOW}s, keep ${KEEP_RATIO})"
    (( DRY_RUN == 1 )) || rm -rf "${raw}"
    local extract_target="${cache}"
    if [[ "${DROP_RATIO}" != "0" && "${DROP_RATIO}" != "0.0" ]]; then extract_target="${raw}"; fi
    run "${PROJECT_PYTHON}" "${EXTRACTOR}" "${video}" "${extract_target}" \
        --sample-fps "${SAMPLE_FPS}" --max-frames "${MAX_FRAMES}" \
        --sharpness-window "${SHARPNESS_WINDOW}" --sharpness-keep-ratio "${KEEP_RATIO}" \
        --sharpness-workers "${SHARPNESS_WORKERS}"
    if [[ "${extract_target}" == "${raw}" ]]; then
        log "${stem}: drop residual blur (ratio_local < ${DROP_RATIO})"
        run "${PROJECT_PYTHON}" "${FILTER}" "${raw}" "${cache}" --ratio-threshold "${DROP_RATIO}"
        run rm -rf "${raw}"
    fi
    log "${stem}: embed EXIF capture metadata from ${ANNOTATION}"
    local annotate_cmd=("${PROJECT_PYTHON}" "${ANNOTATOR}" "${cache}" --annotation "${ANNOTATION}")
    [[ -z "${MAKE}" ]] || annotate_cmd+=(--make "${MAKE}")
    [[ -z "${MODEL}" ]] || annotate_cmd+=(--model "${MODEL}")
    [[ -z "${SOFTWARE}" ]] || annotate_cmd+=(--software "${SOFTWARE}")
    run "${annotate_cmd[@]}"
    if (( DRY_RUN == 0 )); then
        validate_cache "${cache}" "${video}" || { echo "error: ${cache} failed validation after preparation" >&2; return 1; }
    fi
}

run_pipeline() {
    echo "Object:      ${NAME} (${#CLIPS[@]} clips) -> ${MERGED}"
    echo "Recipe:      ${SAMPLE_FPS} fps, sharpness window ${SHARPNESS_WINDOW}s keep ${KEEP_RATIO}, drop ratio<${DROP_RATIO}, EXIF from ${ANNOTATION}"
    local stem caches=()
    for stem in "${CLIPS[@]}"; do
        prepare_clip "${stem}"
        caches+=("${WORK_ROOT}/${stem}")
    done
    if [[ -d "${MERGED}" ]] && (( DRY_RUN == 0 )); then
        log "merged dataset exists: ${MERGED} (validating)"
        local videos=() s
        for s in "${CLIPS[@]}"; do videos+=("$(find_video "${s}")"); done
        "${PROJECT_PYTHON}" "${VALIDATOR}" "${MERGED}" "${videos[@]}" --sample-fps "${SAMPLE_FPS}" --max-frames "${MAX_FRAMES}" \
            --sharpness-window "${SHARPNESS_WINDOW}" --sharpness-keep-ratio "${KEEP_RATIO}" --sharpness-drop-ratio "${DROP_RATIO}" \
            || { echo "error: existing merged dataset does not validate; remove it to rebuild" >&2; return 1; }
        log "merged dataset valid, nothing to do"
        return 0
    fi
    log "merge ${#caches[@]} caches -> ${MERGED}"
    run "${PROJECT_PYTHON}" "${MERGER}" "${MERGED}" "${caches[@]}" --link-mode hardlink
    echo "Next: bash scripts/run_object_colmap.sh --dataset ${MERGED}"
}

if (( DRY_RUN == 1 )); then run_pipeline; echo "Dry run complete; nothing was written."; exit 0; fi
[[ -x "${PROJECT_PYTHON}" ]] || { echo "error: gsplat environment is missing; run scripts/setup_gsplat.sh first" >&2; exit 1; }
[[ -f "${ANNOTATION}" ]] || { echo "error: annotation CSV not found: ${ANNOTATION}" >&2; exit 1; }
for stem in "${CLIPS[@]}"; do find_video "${stem}" >/dev/null || { echo "error: video not found for ${stem} under ${CAPTURE_ROOT}" >&2; exit 1; }; done
mkdir -p "${WORK_ROOT}/_logs"
LOCK_DIR="${WORK_ROOT}/.${NAME}-prepare-lock"
mkdir "${LOCK_DIR}" 2>/dev/null || { echo "error: another prepare run may be active: ${LOCK_DIR}" >&2; exit 1; }
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
LOG_PATH="${WORK_ROOT}/_logs/prepare_${NAME}.log"
echo "=== $(date -u +%FT%TZ) prepare_object_frames.sh --project ${PROJECT:-?} --name ${NAME} ${CLIPS[*]}" >> "${LOG_PATH}"
set +e
run_pipeline 2>&1 | tee -a "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e
(( status == 0 )) || { echo "prepare failed; log: ${LOG_PATH}" >&2; exit "${status}"; }
echo "Log: ${LOG_PATH}"
