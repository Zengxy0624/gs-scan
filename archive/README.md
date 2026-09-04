# archive/

Retired code and runbooks, kept out of `scripts/`, `tests/` and `docs/` so the
live pipeline stays readable. Nothing here is part of the current object
pipeline (`docs/OBJECT_PIPELINE.md`); nothing here is maintained. It is kept
because the ambulance experiments in `docs/EXPERIMENTS.md` were produced with
it and the numbers there are only reproducible with these scripts.

Everything still imports the shared helpers that stayed in `scripts/`
(`merge_frame_datasets.py`, `prepare_colmap.py`), so it keeps working as long
as those keep their current internals — but that is not a contract. If one of
these tools becomes useful again, move it back into `scripts/` and re-adopt it
rather than growing a second pipeline in here.

## Running the archived code

Repo root must be the working directory (it is what makes `scripts.…` and
`archive.scripts.…` importable):

```bash
# a script
third-party/gsplat/.venv/bin/python archive/scripts/filter_sharp_dataset.py --help

# the archived tests (excluded from `unittest discover -s tests`)
third-party/gsplat/.venv/bin/python -m unittest discover -s archive/tests -t .
```

## `scripts/` — what each one did and what replaced it

| Script | Purpose | Status |
| --- | --- | --- |
| `filter_sharp_dataset.py` | Blur-filter an already reconstructed dataset; removed frames are deregistered from a *copy* of the COLMAP model so no bundle adjustment is re-run. | Superseded by `scripts/filter_frame_cache.py`, which drops blurry frames before COLMAP ever sees them. |
| `sample_bootstrap_frames.py` | Extract candidate midpoint frames around weak videos / worst-scoring frames of a trained run. | Ambulance bootstrap loop; never used on the object pipeline. |
| `register_new_frames.py` | Register those candidates into an existing model, restricting matching to the new images. | Same loop. Refuses to start without the OpenBLAS shim (see `third-party/openblas-threaded/README.md`). |
| `score_candidate_frames.py` | Rank candidate frames by render error against a trained checkpoint. | Same loop. |
| `build_run_manifest.py` | Provenance manifest (dataset identity + gsplat commit + settings) for a training run. | Standalone; never wired into `train_experiment.py`. |

## `docs/` — retired runbooks

`device_test.md` and `record_pattern_test.md` are the early capture-comparison
batches. Their launchers (`run_device_test.sh`, `run_record_pattern_test.sh`)
were deleted, so these runbooks are historical only; the results they describe
are summarised in `docs/EXPERIMENTS.md`.
