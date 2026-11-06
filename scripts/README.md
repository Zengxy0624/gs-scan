# scripts/

Flat on purpose: running tmux jobs, shell history, and the test suite all
reference these paths. Every script has `--dry-run` where it launches work and
a `tests/test_<stem>.py`. The end-to-end runbook is `docs/OBJECT_PIPELINE.md`;
retired scripts live in `archive/` and are not listed below.

## The object pipeline (clips → frames → COLMAP → 3DGS)

- `run_object_pipeline.sh` — driver: `--project P --name N [--gpu G] CLIP…`
  runs the three stages below (`--stages frames,colmap,train`), prints every
  underlying command, `--ba-preset fast|production`, `--train-steps`,
  `--tag`, pass-through `--prepare-arg/--colmap-arg`.
- `prepare_object_frames.sh` — stage 1: per clip, blur-aware extraction →
  residual-blur filter → EXIF capture metadata; then hardlink-merge into
  `work/P/N`. Idempotent; `--project`/`--capture-root`/`--work-root`,
  `--annotation`, `--sample-fps`, `--sharpness-window`, `--keep-ratio`,
  `--drop-ratio`.
- `run_object_colmap.sh` — stage 2: production COLMAP recipe on one merged
  dataset (`--dataset`); auto-selects the cuDSS GPU build → CUDA wheel → CPU;
  BLAS shim + 32-thread cap for pip wheels; explicit BA schedule; snapshots /
  `--resume-from-snapshot` / `--reuse-matches`; per-clip registration gates;
  lock + log under `<parent>/_logs/`.
- `train_experiment.py` — stage 3: instrumented `simple_trainer` wrapper
  (per-image + per-clip val CSVs, opacity/scale distributions, optional W&B
  via `WANDB_PROJECT`). `--ckpt CKPT` re-renders the trajectory video with the
  `--app-opt` module restored; `--ckpt CKPT --view-only [--port]` serves the
  viser viewer from a checkpoint. Tests: `test_train_experiment_ckpt.py`,
  `test_train_experiment_wandb.py`.
- `gs_from_video.sh` — one video end to end (`PROJECT CLIP GPU`): stage 1
  steps without the merge, COLMAP as cuDSS CLI commands with global mapping,
  training with a 1 M budget, `--pose-opt` and no `--app-opt`
  (`docs/OBJECT_PIPELINE.md` §8). No `--dry-run` or test yet.

## Stage building blocks

- `extract_frames.py` — one video → `images/*.jpg` + content-hashed
  `metadata.json`; `--sharpness-window S --sharpness-keep-ratio R` swaps
  blurry anchors for the sharpest frame within ±S s; writes
  `sharpness_scores.csv` / `sharpness_all_frames.csv`. `--additional-video`
  builds a multi-video set directly.
- `filter_frame_cache.py` — hardlinked copy of a cache without images whose
  `ratio_local` < threshold (optional `--raw-floor`); renumbers, rewrites
  metadata + CSVs, records the threshold in the recipe.
- `annotate_frame_cache.py` — embeds the video's annotation-CSV row as EXIF in
  every image (ISO, shutter, f-number, 35 mm-equivalent focal length → COLMAP
  focal prior, WB, per-frame capture time, GPS, description, make/model/
  software) and writes `image_metadata.csv`; lossless.
- `merge_frame_datasets.py` — hardlink-merge validated single-video caches
  into one `video_set` (`NN_<clip>_` prefixes); refuses mixed recipes; carries
  `capture_metadata`, concatenates `image_metadata.csv`.
- `prepare_colmap.py` — images → `database.db` + `sparse/0` via pycolmap 3.13
  or 4.x: vocab-tree/sequential/exhaustive matching, single shared camera,
  `--feature-gpu`, `--ba-use-gpu`, BA-schedule tuning, snapshots/resume,
  match reuse, registration gates, `reconstruction_stats.json`.
- `validate_dataset_cache.py` — verifies a cache or merged dataset against its
  sources and recipe (fps, window, keep/drop ratios) and, with
  `--require-colmap`, the recorded COLMAP configuration and model files.

## Environment

- `setup_gsplat.sh` — pinned gsplat environment (`third-party/gsplat/.venv`).
- `build_gpu_colmap.sh` — source-build the CUDA + cuDSS pycolmap for GPU
  bundle adjustment into `work/_gpu_colmap/` (micromamba + conda-forge; Ceres
  master, COLMAP 4.2.0; phased/resumable; `--dry-run`).

## Inspection

- `export_rerun_viz.py` — sparse model + camera poses → `.rrd` for rerun.
- `export_cameras.py` — COLMAP camera poses (c2w, OpenCV axes) in the frame
  of the gsplat PLY (gsplat's `Parser`, `normalize=True`) → `cameras.npz`.
  No test yet.

## Environment contract (matching runs)

Any pycolmap *pip wheel* (CPU 3.13 or `pycolmap-cuda12`) statically links a
serial OpenBLAS that segfaults when faiss vocabulary-tree retrieval calls it
from many threads. Launch such runs with the vendored pthread shim:

```bash
LD_PRELOAD=$PWD/third-party/openblas-threaded/libopenblas-pthread.so.0 \
LD_LIBRARY_PATH=$PWD/third-party/openblas-threaded \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=32 ... --num-threads 32
```

Both COLMAP worker threads and OpenMP workers count against the shim's 64
concurrent-caller limit; 256 + 64 crashed, 32 + 32 is safe.
`run_object_colmap.sh` applies all of this itself;
`archive/scripts/register_new_frames.py` refuses to start without it. The cuDSS
build (`work/_gpu_colmap`) links a threaded BLAS and needs neither. Details and
forensics: `third-party/openblas-threaded/README.md`.

## Retired

The ambulance-era curation and hard-example-mining tools
(`filter_sharp_dataset.py`, `sample_bootstrap_frames.py`,
`register_new_frames.py`, `score_candidate_frames.py`) and the standalone
`build_run_manifest.py` moved to `archive/scripts/`, with their tests in
`archive/tests/`. They are unmaintained and excluded from
`unittest discover -s tests`; `archive/README.md` says what each did and what
replaced it.
