# 3DGS-Health conventions

3D Gaussian Splatting pipeline (video → blur-aware frames → COLMAP → gsplat)
for medical-simulation scenes and objects, on the shared shenlong GPU servers
(currently shenlong-gpu-02; the ambulance work was done on gpu-01).

## Ground rules

- **The user launches training and long GPU jobs themselves** (in tmux).
  Prepare data/scripts, validate on real data, hand over the exact command.
  Anything Claude itself runs that takes more than a few minutes goes in a
  named tmux session with a durable log under `work/…/_logs/` (laptop-sleep
  proof; progress must be recoverable after a disconnect).
- **Shared GPUs**: check `nvidia-smi` for other users' jobs before ANY GPU use.
- **Never modify `third-party/gsplat`** (pinned submodule; manifest provenance
  requires it clean). Project extensions live in `scripts/`.
- Any pycolmap **feature-matching** run needs the OpenBLAS shim environment set
  AT LAUNCH — see `scripts/README.md` and
  `third-party/openblas-threaded/README.md`. Two crash modes without it.
- Datasets under `work/` are immutable once reconstructed: curation always
  builds a NEW dataset directory via hardlinks (`scripts/filter_frame_cache.py`
  pre-COLMAP; the retired post-COLMAP tools in `archive/scripts/`); no in-place
  edits, no re-running BA.

## Where things are

- `docs/OBJECT_PIPELINE.md` — **the pipeline runbook**: one object from clips
  to a trained splat (`scripts/run_object_pipeline.sh`), every knob, and the
  failure modes seen so far. Start here for a new recording.
- `docs/EXPERIMENTS.md` — the experiment ledger (datasets, runs, numbers,
  verdicts, queue). **Update it when an experiment finishes.**
- `scripts/README.md` — per-script index in pipeline order.
- `archive/` — retired scripts, their tests, and the old runbooks. Not
  maintained, excluded from `unittest discover -s tests`; see
  `archive/README.md` before reviving anything.
- Experiment outputs: `outputs/ambulance/<run>/` with `stats/val_step*.json`,
  per-image CSVs, `tb/`, `ply/`, `videos/`.
- Cross-run comparisons must use shared-identical val frames from the
  per-image CSVs, never headline numbers across different datasets.

## Code style

- argparse + pathlib, module docstring, type hints, atomic tempdir staging
  (`tempfile.mkdtemp(prefix=".<name>.", dir=parent)` then `replace`),
  `main()` catching (OSError, RuntimeError, ValueError) → `error: ...` on
  stderr, exit 1. Reference: `scripts/merge_frame_datasets.py`.
- Every script gets a `tests/test_<stem>.py` (unittest style, stubbed
  pycolmap via `patch.dict(sys.modules, ...)` — see
  `tests/test_prepare_colmap.py`).
- Run the suite:
  `third-party/gsplat/.venv/bin/python -m unittest tests.test_<name>` (no
  pytest in the pinned venv).
