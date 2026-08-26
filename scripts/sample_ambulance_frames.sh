#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GSPLAT_ROOT="${PROJECT_ROOT}/third-party/gsplat"
EXTRACTOR="${PROJECT_ROOT}/scripts/extract_frames.py"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_dataset_cache.py"
PROJECT_PYTHON="${PROJECT_PYTHON:-${GSPLAT_ROOT}/.venv/bin/python}"

CAPTURE_ROOT="${CAPTURE_ROOT:-${PROJECT_ROOT}/captures/ambulance}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/work/ambulance}"

SAMPLE_FPS="3"
MAX_FRAMES="100000"
DRY_RUN=0

# Keep this list in the same order as the "Ambulance Interior" table in
# captures/ambulance/ANNOTATION.md. Calibration/reference captures are omitted.
VIDEO_FILES=(
    "A001_08071507_C004.mov"
    "A001_08071511_C005.mov"
    "A001_08071516_C006.mov"
    "A001_08071521_C007.mov"
    "A001_08071522_C008.mov"
    "A001_08071524_C009.mov"
    "A001_08071542_C014.mov"
    "A001_08071544_C016.mov"
    "A001_08071548_C017.mov"
    "A001_08071550_C018.mov"
    "A001_08071550_C019.mov"
    "A001_08071649_C022.mov"
    "A001_08071655_C023.mov"
    "A001_08071655_C024.mov"
    "A001_08071656_C025.mov"
    "A001_08071657_C026.mov"
    "A001_08071657_C027.mov"
    "A001_08071705_C028.mov"
)

usage() {
    cat <<'EOF'
Sample every ambulance-interior video into its own 3 FPS frame dataset.

Usage:
  bash scripts/sample_ambulance_frames.sh [--dry-run]

Options:
  --dry-run   Print every extraction command without creating or sampling data.
  -h, --help  Show this help.

Environment overrides:
  CAPTURE_ROOT    Input directory (default: captures/ambulance).
  WORK_ROOT       Output directory (default: work/ambulance).
  PROJECT_PYTHON  Python used to validate completed output on rerun.

Each video is written to WORK_ROOT/<video-name-without-extension>/, containing
images/*.jpg and metadata.json. Valid completed datasets are skipped on rerun;
incomplete or mismatched output is rejected without being overwritten.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [[ ! -d "${CAPTURE_ROOT}" ]]; then
    echo "error: ambulance capture directory not found: ${CAPTURE_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${EXTRACTOR}" ]]; then
    echo "error: frame extractor not found: ${EXTRACTOR}" >&2
    exit 1
fi
if [[ ! -f "${VALIDATOR}" ]]; then
    echo "error: dataset validator not found: ${VALIDATOR}" >&2
    exit 1
fi
if [[ -L "${WORK_ROOT}" && ! -e "${WORK_ROOT}" ]]; then
    echo "error: output root is a dangling symbolic link: ${WORK_ROOT}" >&2
    exit 1
fi
if [[ -e "${WORK_ROOT}" && ! -d "${WORK_ROOT}" ]]; then
    echo "error: output root exists but is not a directory: ${WORK_ROOT}" >&2
    exit 1
fi

# Validate the complete batch before starting so a typo, missing capture, or
# stale output cannot leave an unexpectedly partial extraction.
preflight_failed=0
needs_validator=0
declare -A SKIP_DATASET=()
for video_file in "${VIDEO_FILES[@]}"; do
    source_video="${CAPTURE_ROOT}/${video_file}"
    video_name="${video_file%.*}"
    dataset="${WORK_ROOT}/${video_name}"

    if [[ ! -f "${source_video}" || ! -r "${source_video}" ]]; then
        echo "error: ambulance video is missing or unreadable: ${source_video}" >&2
        preflight_failed=1
    fi
    if [[ -L "${dataset}" ]]; then
        echo "error: output path must not be a symbolic link: ${dataset}" >&2
        preflight_failed=1
    elif [[ -e "${dataset}" ]]; then
        if [[ ! -d "${dataset}" ]]; then
            echo "error: output exists but is not a directory: ${dataset}" >&2
            preflight_failed=1
        else
            SKIP_DATASET["${video_file}"]=1
            needs_validator=1
        fi
    fi
done
if (( preflight_failed == 1 )); then
    exit 1
fi

if (( DRY_RUN == 0 || needs_validator == 1 )); then
    if [[ ! -x "${PROJECT_PYTHON}" ]]; then
        echo "error: gsplat environment is missing; run scripts/setup_gsplat.sh first" >&2
        exit 1
    fi
fi

if (( needs_validator == 1 )); then
    for video_file in "${VIDEO_FILES[@]}"; do
        if [[ "${SKIP_DATASET[${video_file}]:-0}" != "1" ]]; then
            continue
        fi
        source_video="${CAPTURE_ROOT}/${video_file}"
        video_name="${video_file%.*}"
        dataset="${WORK_ROOT}/${video_name}"
        if ! "${PROJECT_PYTHON}" "${VALIDATOR}" \
            "${dataset}" "${source_video}" \
            --sample-fps "${SAMPLE_FPS}" \
            --max-frames "${MAX_FRAMES}"; then
            echo "error: existing output is incomplete or mismatched: ${dataset}" >&2
            preflight_failed=1
        fi
    done
fi
if (( preflight_failed == 1 )); then
    echo "Move the reported output aside or choose a fresh WORK_ROOT, then retry." >&2
    exit 1
fi

if (( DRY_RUN == 0 )); then
    if ! command -v uv >/dev/null 2>&1; then
        echo "error: uv is not available on PATH" >&2
        exit 1
    fi
    mkdir -p "${WORK_ROOT}"
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

echo "Sampling ${#VIDEO_FILES[@]} ambulance videos at ${SAMPLE_FPS} frames/second."
for video_file in "${VIDEO_FILES[@]}"; do
    source_video="${CAPTURE_ROOT}/${video_file}"
    video_name="${video_file%.*}"
    dataset="${WORK_ROOT}/${video_name}"

    echo
    echo "=== ${video_name} ==="
    if [[ "${SKIP_DATASET[${video_file}]:-0}" == "1" ]]; then
        echo "Skipping validated frame dataset: ${dataset}"
        continue
    fi
    run_command \
        uv run --project "${GSPLAT_ROOT}" --no-sync \
        python "${EXTRACTOR}" \
        "${source_video}" "${dataset}" \
        --sample-fps "${SAMPLE_FPS}" \
        --max-frames "${MAX_FRAMES}"
done

if (( DRY_RUN == 1 )); then
    echo
    echo "Dry run complete; no frames were extracted."
else
    echo
    echo "Completed all ambulance frame datasets under: ${WORK_ROOT}"
fi
