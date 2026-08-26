# scripts/

Flat on purpose: running tmux jobs, shell history, and the test suite all
reference these paths, so files are grouped here by role instead of moved into
subdirectories. Every script is covered by `tests/` (same stem, `test_` prefix,
except where noted).

## Pipeline, in execution order

**1. Frames from video**
- `extract_frames.py` — one video → `dataset/images/*.jpg` + content-hashed
  `metadata.json` (uniform sampling, default 3 fps).
- `sample_ambulance_frames.sh` — batch wrapper: every ambulance video → its own
  frame cache under `work/ambulance/` (tested via `test_ambulance_scripts.py`).
- `merge_frame_datasets.py` — hardlink-merge validated single-video caches into
  one `video_set` dataset with per-source `NN_<video>_` name prefixes.

**2. COLMAP reconstruction**
- `prepare_colmap.py` — images → `database.db` + `sparse/0` via pycolmap.
  Vocab-tree/sequential/exhaustive matching, single shared camera, BA-schedule
  tuning, snapshots/resume, GPU BA, match reuse.
- `run_ambulance_colmap.sh` — production wrapper: merge + reconstruct with the
  quality recipe, OpenBLAS shim environment, locking, registration gates,
  `--resume-from-snapshot`, `COLMAP_PYTHON` override for the GPU-BA
  interpreter (tested via `test_ambulance_scripts.py`).

**3. Training**
- `train_experiment.py` — instrumented `simple_trainer` wrapper (per-image +
  per-source val CSVs, opacity/scale distribution logging). The trainer used
  for every experiment since exp6 — see `docs/EXPERIMENTS.md` for recipes
  (no dedicated unit test; smoke-tested by running).
- `train_ambulance_gs.sh` — manifest-gated batch trainer with fixed profiles
  (tested via `test_ambulance_scripts.py`).
- `run_single_video_gs.sh` — one-video COLMAP+GS convenience pipeline.

**4. Dataset curation**
- `filter_sharp_dataset.py` — build a blur-filtered copy of a reconstructed
  dataset: removed frames are deregistered from a copy of the COLMAP model
  (no new BA); supports a protected-names list so bootstrap candidates
  survive base-frame cuts.

**5. Bootstrap loop (hard-example mining; see docs/EXPERIMENTS.md)**
- `sample_bootstrap_frames.py` — extract 60fps midpoint candidates around
  weak videos / worst-scoring frames from the source videos.
- `register_new_frames.py` — register candidates into an existing model:
  augmented dataset copy, extraction+matching restricted to the new images,
  resume mapping, single-camera and registration-ratio gates. Enforces the
  OpenBLAS launch-environment contract (fails fast with the fix).
- `score_candidate_frames.py` — render registered candidates from a trained
  PLY at their COLMAP poses, rank by exposure-neutralized PSNR (worst = best
  round-2 targets).

**Provenance / environment**
- `build_run_manifest.py` — bind a training run to source/model/code identity.
- `validate_dataset_cache.py` — verify frame caches against their metadata.
- `setup_gsplat.sh` — pinned environment bootstrap.
- `run_device_test.sh`, `run_record_pattern_test.sh` — earlier device/record
  comparison batches (see `docs/device_test.md`, `docs/record_pattern_test.md`).

## Environment contract (matching runs)

Any run that performs pycolmap feature matching must be launched with the
pthread OpenBLAS shim preloaded and thread pins set **at launch**:

```bash
LD_PRELOAD=$PWD/third-party/openblas-threaded/libopenblas-pthread.so.0 \
LD_LIBRARY_PATH=$PWD/third-party/openblas-threaded \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=32 ...
```

`run_ambulance_colmap.sh` sets this itself; `register_new_frames.py` verifies
it and refuses to start otherwise. Rationale and crash forensics:
`third-party/openblas-threaded/README.md`.
