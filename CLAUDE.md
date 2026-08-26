# 3DGS-Health conventions

Ambulance 3D Gaussian Splatting pipeline on shenlong-gpu-01 (shared machine).

## Ground rules

- **The user launches training and long GPU jobs themselves** (in tmux).
  Prepare data/scripts, validate on real data, hand over the exact command.
- **Shared GPUs**: check `nvidia-smi` for other users' jobs before ANY GPU use.
- **Never modify `third-party/gsplat`** (pinned submodule; manifest provenance
  requires it clean). Project extensions live in `scripts/`.
- Any pycolmap **feature-matching** run needs the OpenBLAS shim environment set
  AT LAUNCH — see `scripts/README.md` and
  `third-party/openblas-threaded/README.md`. Two crash modes without it.
- Datasets under `work/` are immutable once reconstructed: curation
  (blur filters, bootstrap augmentation) always builds a NEW dataset directory
  via hardlinks + model-copy deregistration (`scripts/filter_sharp_dataset.py`,
  `scripts/register_new_frames.py`); no in-place edits, no re-running BA.

## Where things are

- `docs/EXPERIMENTS.md` — the experiment ledger (datasets, runs, numbers,
  verdicts, queue). **Update it when an experiment finishes.**
- `scripts/README.md` — per-script index in pipeline order.
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
  `tests/test_register_new_frames.py`).
- Run the suite:
  `third-party/gsplat/.venv/bin/python -m unittest tests.test_<name>` (no
  pytest in the pinned venv).
