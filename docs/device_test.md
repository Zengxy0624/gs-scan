# Device comparison batch

This runbook processes the five standard device-test videos. The two `.insv`
fisheye streams are intentionally ignored.

## Inputs

Place the source videos under `captures/device_test/` with these exact names:

```text
captures/device_test/
├── insta_normal.mp4
├── iphone_18mm.mov
├── iphone_24mm.mov
├── osmo_rotate.MP4
└── osmo_stable.MP4
```

Each video produces its own scene folder.

## Run on the server

Complete the server setup described in the main README first. For later runs:

```bash
cd ~/projects/3DGS-Health
module load cuda-toolkit/13.0  # if required on each new cluster login
git submodule update --init --recursive
bash scripts/run_device_test.sh --profile full --gpu 6
```

If the CUDA module is already loaded and the submodule is initialized, only the
last line is needed. COLMAP frame matching and reconstruction run on the CPU
with eight threads by default. Only gsplat training uses the requested GPU;
`--gpu 6` sets `CUDA_VISIBLE_DEVICES=6` for the trainer.

The script does not load modules, run setup, install packages, or modify
anything outside the selected work and output roots.

The full-profile defaults are:

- 6 sampled frames per second;
- at most 100,000 frames per video;
- a dense 15-frame COLMAP match window plus sparse longer-baseline pairs;
- a minimum 50% COLMAP registration ratio for each scene;
- 8 COLMAP CPU threads; and
- 30,000 gsplat training steps.

Jobs run sequentially. A failed scene is recorded, and the remaining scenes are
still attempted.

## Work state and durable outputs

Generated frames, the COLMAP database, and the sparse model are stored as
repo-local work state:

```text
work/device_test/fps6_max100000_overlap15_reg0.5/
├── insta_normal/dataset/
├── iphone_18mm/dataset/
├── iphone_24mm/dataset/
├── osmo_rotate/dataset/
└── osmo_stable/dataset/
```

Durable training artifacts and logs are separate:

```text
outputs/device_test/
├── insta_normal/
│   ├── fps6_max100000_overlap15_reg0.5_30000steps/
│   │   └── pipeline_manifest.json
│   └── _logs/
├── iphone_18mm/
├── iphone_24mm/
├── osmo_rotate/
├── osmo_stable/
└── summary_fps6_max100000_overlap15_reg0.5_30000steps.tsv
```

`work/` is the safe repo-local default: unlike system `/tmp`, it is less likely
to be purged during a long run and makes completed stages reusable. On a server
with a documented project-scratch location, redirect only that work root:

```bash
WORK_ROOT=/path/to/project-scratch/device_test \
  bash scripts/run_device_test.sh --profile full --gpu 6
```

`OUTPUT_ROOT` can similarly redirect durable results. `CAPTURE_ROOT` can point
at an equivalent input directory containing the five exact source names.

Keep `outputs/` durable. A completed result is reused by semantic identity
(source-video content, extraction/COLMAP recipe, gsplat commit, and trainer
configuration), so rebuilding byte-different COLMAP files from the same recipe
does not automatically discard a finished 30,000-step result. Its original
exact frame/model hashes remain recorded in `pipeline_manifest.json`. Within
the same work-dataset instance, any frame or sparse-model mutation invalidates
reuse. Across a clean re-extraction, the extracted-image digest must still
match; only the regenerated COLMAP binaries may differ semantically.

## Preprocess, resume, and retry

Useful narrower runs are:

```bash
# Extract frames and run CPU COLMAP for all five scenes; do not train.
bash scripts/run_device_test.sh --preprocess-only

# Run a short end-to-end pilot for one scene.
bash scripts/run_device_test.sh \
  --profile pilot \
  --gpu 6 \
  --scene iphone_18mm

# Show all commands and paths without creating files or running stages.
bash scripts/run_device_test.sh --dry-run
```

`--profile pilot` trains for 7,000 steps; `--profile full` trains for 30,000.
Use `--scene iphone_18mm` (or another listed scene name) to run only one video.

Completed extraction, COLMAP, and training stages are skipped on rerun. Before
reusing extracted frames, the batch checks source identity, sample rate, frame
cap, and expected numbered images. New caches record a content digest, so an
identical copied or restored source is accepted even if its path or modification
time changed. Completed COLMAP state must also match the requested camera model,
overlap recipe, image count, and selected-model completion marker.

Incomplete or stale work is not silently overwritten. Without
`--retry-incomplete`, the scene fails with an explanation; with it, the batch
moves that state into a timestamped `_failed/` directory and starts fresh. For
a work-state problem this archives the whole scene work directory, including
extracted frames; it does not retry only COLMAP. A lock prevents two copies of
the same batch from writing the same work tree. Live processes owned by another
server user are treated as active rather than as stale locks.

Each training result contains `pipeline_manifest.json`. Its semantic identity
binds reuse to result-affecting inputs, including the exact clean gsplat
submodule commit and every trainer option used by this batch. Separate
provenance records the runtime packages, resource choices, and exact dataset
snapshot without making harmless path, timestamp, or nondeterministic COLMAP
binary changes force retraining.

Caches and result manifests created before this integrity schema do not contain
enough information for automatic adoption. The first run after upgrading fails
closed; use `--retry-incomplete` only after reviewing the paths it reports. The
old state is archived, but preprocessing and any legacy training result must be
rebuilt.

`--dry-run` performs no writes. Use it to verify capture names, selected scenes,
paths, GPU selection, and generated commands before committing server time.

## Compare results

For a full run, the main per-scene numbers are:

```text
outputs/device_test/<scene>/fps6_max100000_overlap15_reg0.5_30000steps/
├── stats/val_step29999.json       # PSNR, SSIM, LPIPS and validation timing
└── pipeline_manifest.json         # reconstruction stats used by this result

work/device_test/fps6_max100000_overlap15_reg0.5/<scene>/dataset/
└── reconstruction_stats.json     # most recent work reconstruction
```

Compare both reconstruction coverage and validation metrics; a high image metric
does not rescue a fragmented or weakly registered model. For metrics tied to a
specific checkpoint, use
`pipeline_manifest.json.provenance.dataset_snapshot.reconstruction_stats`; the
scratch work tree may have been rebuilt after that result finished. For a fair
device comparison, keep capture duration and route similar. With one-image
training batches and a fixed 30,000-step budget, clips with many more extracted
frames receive fewer updates per view. Scene normalization also means different
camera trajectories can change effective optimization scale, so treat this
five-video batch as an exploratory comparison rather than a controlled
benchmark.

The default COLMAP quality gate requires at least 50% registration and at least
10 images when the input contains 10 or more. Override it deliberately, for
example `--min-registration-ratio 0.7`; weak reconstructions fail rather than
silently proceeding to GPU training.
