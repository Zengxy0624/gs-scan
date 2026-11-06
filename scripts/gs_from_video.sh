#!/bin/bash
# One phone video -> frames -> COLMAP -> gsplat PLY, the single-video counterpart of run_object_pipeline.sh
# (docs/OBJECT_PIPELINE.md §8). Calls the step scripts unchanged; run_object_pipeline.sh does not fit one video: its
# frames stage needs two clips and its COLMAP step adds the per-video gate that prepare_colmap.py rejects for one video.
# Frames and COLMAP features/matching use the production settings. Four differences from the production recipe:
#   - mapping: global_mapper (GLOMAP) of the cuDSS build, run as CLI commands, instead of incremental mapping;
#   - --strategy.cap-max 1000000 instead of 6.5M: the splat is rendered at every simulation step;
#   - no --app-opt: real2sim-eval's GSProcessor (SH degree 3 only) cannot load the PLY it exports; exposure is locked;
#   - --pose-opt: refines the COLMAP poses of the training frames.
# Frames and COLMAP that already validate are reused; training refuses an existing result directory.
#
#   bash scripts/gs_from_video.sh PROJECT CLIP GPU        # in tmux; needs work/_gpu_colmap (build_gpu_colmap.sh)
#
# Reads captures/PROJECT/CLIP.MOV and its row in captures/PROJECT/annotation.csv; writes work/PROJECT/CLIP/ and
# outputs/PROJECT/CLIP/.
set -Eeuo pipefail

[[ $# -eq 3 ]] || { echo "usage: $0 PROJECT CLIP GPU" >&2; exit 2; }
PROJECT="$1"; CLIP="$2"; GPU="$3"
H="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${H}"
PY="${H}/third-party/gsplat/.venv/bin/python"
COLMAP_ENV="${H}/work/_gpu_colmap/mamba_root/envs/gpubuild"  # cuDSS build: bundle adjustment on the GPU
VIDEO="${H}/captures/${PROJECT}/${CLIP}.MOV"
DATASET="${H}/work/${PROJECT}/${CLIP}"
RESULT="${H}/outputs/${PROJECT}/${CLIP}"
[[ -f "${VIDEO}" ]] || { echo "error: video not found: ${VIDEO}" >&2; exit 1; }

FRAMES=(--sample-fps 3 --max-frames 100000 --sharpness-window 0.3 --sharpness-keep-ratio 0.8)
EXTRACT=(--ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --FeatureExtraction.max_image_size 3840
    --SiftExtraction.max_num_features 8192 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
    --FeatureExtraction.use_gpu 1 --FeatureExtraction.gpu_index 0)
MATCH=(--VocabTreeMatching.vocab_tree_path "${H}/work/_colmap_cuda/vocab_tree_faiss_flickr100K_words256K.bin"
    --VocabTreeMatching.num_images 200 --VocabTreeMatching.max_num_features 8192 --FeatureMatching.guided_matching 1
    --FeatureMatching.use_gpu 1 --FeatureMatching.gpu_index 0)
VALIDATE=("${PY}" "${H}/scripts/validate_dataset_cache.py" "${DATASET}" "${VIDEO}" "${FRAMES[@]}" --sharpness-drop-ratio 0.7)
MARKER="${DATASET}/colmap_global.txt"  # recipe + model checksums, written last

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
run() { printf '+'; printf ' %q' "$@"; printf '\n'; "$@"; }
marker() { printf '%s\n' "${EXTRACT[*]}" "${MATCH[*]}" global_mapper; (cd "${DATASET}/sparse/0" && sha256sum cameras.bin images.bin points3D.bin); }

log "frames -> ${DATASET}"
if [[ -d "${DATASET}" ]]; then
    "${VALIDATE[@]}" >/dev/null && [[ -f "${DATASET}/image_metadata.csv" ]] \
        || { echo "error: ${DATASET} does not match this recipe; move it aside" >&2; exit 1; }
    log "frames: reusing"
else
    RAW="${H}/work/${PROJECT}/.${CLIP}.unfiltered"
    rm -rf "${RAW}"
    run "${PY}" "${H}/scripts/extract_frames.py" "${VIDEO}" "${RAW}" "${FRAMES[@]}" --sharpness-workers 16
    run "${PY}" "${H}/scripts/filter_frame_cache.py" "${RAW}" "${DATASET}" --ratio-threshold 0.7
    rm -rf "${RAW}"
    run "${PY}" "${H}/scripts/annotate_frame_cache.py" "${DATASET}" --annotation "${H}/captures/${PROJECT}/annotation.csv" \
        --make Apple --model "iPhone 16 Pro Max" --software "Blackmagic Cam 3.5.200045"
    run "${VALIDATE[@]}"
fi

log "colmap -> ${DATASET}/sparse/0"
if [[ -f "${MARKER}" ]] && diff -q <(marker 2>/dev/null) "${MARKER}" >/dev/null; then
    log "colmap: reusing"
else
    for f in database.db sparse; do
        [[ ! -e "${DATASET}/${f}" ]] || { echo "error: stale COLMAP output ${DATASET}/${f}; move it aside" >&2; exit 1; }
    done
    # CPU SIFT (affine + DSP at 4K) peaks at ~1.9 GB per thread: leave 2.5 GB of available RAM per thread, at most 32.
    THREADS=$(awk '/MemAvailable/ {t = int($2 / 1048576 / 2.5); print (t < 32 ? t : 32)}' /proc/meminfo)
    C=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=32 "${COLMAP_ENV}/bin/colmap")
    DB=(--database_path "${DATASET}/database.db")
    run "${C[@]}" feature_extractor "${DB[@]}" --image_path "${DATASET}/images" "${EXTRACT[@]}" \
        --FeatureExtraction.num_threads "${THREADS}"
    run "${C[@]}" vocab_tree_matcher "${DB[@]}" "${MATCH[@]}" --FeatureMatching.num_threads "${THREADS}" \
        --VocabTreeMatching.num_threads "${THREADS}"
    mkdir "${DATASET}/sparse"
    run "${C[@]}" global_mapper "${DB[@]}" --image_path "${DATASET}/images" --output_path "${DATASET}/sparse" \
        --GlobalMapper.num_threads "${THREADS}"
    # prepare_colmap.py's gate: one model holding at least 80% of the frames
    run "${COLMAP_ENV}/bin/python" -c "
import pathlib, sys, pycolmap
d = pathlib.Path(sys.argv[1]); models = sorted(p.name for p in (d / 'sparse').iterdir() if p.is_dir())
reg, n = pycolmap.Reconstruction(d / 'sparse' / '0').num_reg_images(), len(list((d / 'images').iterdir()))
print(f'models {models}, registered {reg}/{n}'); sys.exit(models != ['0'] or reg < 0.8 * n)
" "${DATASET}"
    marker > "${MARKER}"
fi

log "train -> ${RESULT}"
[[ ! -e "${RESULT}" ]] || { echo "error: ${RESULT} exists; move it aside" >&2; exit 1; }
run env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${H}/scripts/train_experiment.py" mcmc \
    --data-dir "${DATASET}" --data-factor 1 --result-dir "${RESULT}" \
    --strategy.cap-max 1000000 --steps-scaler 4 --strategy.refine-stop-iter 29000 --opacity-reg 0.0005 \
    --pose-opt --eval-steps 30000 --save-steps 30000 --ply-steps 30000 --save-ply --disable-viewer
cat "${RESULT}"/stats/val_step*.json; echo

log "done: $(ls "${RESULT}"/ply/point_cloud_*.ply)"
