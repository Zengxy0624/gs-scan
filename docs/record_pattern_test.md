# Record-pattern comparison batch

This batch builds two independent Gaussian-splat reconstructions from the
record-pattern captures:

| Scene | Source video(s) | COLMAP matching | Training GPU |
| --- | --- | --- | --- |
| `inward_orbit` | `inward_orbit.mov` | Sequential | 6 |
| `inward_orbit_upward` | `inward_orbit.mov`, then `upward.mov` | Exhaustive | 7 |

The two complete jobs run concurrently, like starting them in two terminals.
Their frame extraction and CPU COLMAP stages are independent. As soon as one
job finishes COLMAP, it can begin training on its assigned GPU while the other
job continues.

## Server command

Place the captures at these exact paths:

```text
captures/record_pattern_test/
├── inward_orbit.mov
└── upward.mov
```

After the repository's gsplat environment has been set up, run from the
repository root:

```bash
bash scripts/run_record_pattern_test.sh
```

The default is a full 30,000-step run. The launcher always runs both scenes;
it does not have a single-scene option. A failure in one job does not stop the
other job, and the command exits nonzero after both finish if either failed.

For a shorter 7,000-step experiment:

```bash
bash scripts/run_record_pattern_test.sh --profile pilot
```

## Default processing recipe

Each source is sampled uniformly at 6 frames per second. The default
`--max-frames 100000` is a cap **per source video**, not per dataset. The
inward-only dataset therefore contains at most 100,000 frames, while the
combined dataset can contain at most 200,000 frames. The actual counts are
normally much lower and are determined by video duration and sampling rate.

Both reconstructions use one shared `OPENCV` COLMAP camera calibration within
their dataset:

- `inward_orbit` uses quadratic sequential matching at frame offsets `1, 2,
  4, ...`, with 15 levels by default. `--overlap` controls this job only.
- `inward_orbit_upward` uses exhaustive matching across every extracted image
  pair. This is intentionally more expensive, but it gives the two separate
  clips a much better chance of forming one connected reconstruction.

Before the combined COLMAP stage starts, the launcher prints the extracted
image count and the resulting exhaustive candidate-pair count. For the current
419.36-second and 81.12-second captures, 6 FPS is about 3,004 images and 4.51
million candidate pairs, so expect this CPU stage to be substantial. The
launcher warns above one million pairs and fails closed above five million by
default, preventing an unexpectedly large extraction from launching billions
of comparisons. To deliberately raise the guard after checking the printed
count:

```bash
bash scripts/run_record_pattern_test.sh --max-exhaustive-pairs 8000000
```

Lowering `--sample-fps` is usually the better response if the guard trips.

COLMAP must register at least 80% of all input images. The combined scene also
requires at least 80% registration independently from both `inward_orbit.mov`
and `upward.mov`. If either source falls below the per-source gate, training is
not started; this prevents an apparently successful combined result that
actually discarded one of the videos.

COLMAP runs on the CPU. Both jobs use 8 CPU threads by default, so two active
COLMAP stages can request roughly 16 worker threads in total. Adjust the
per-job limit on a shared server, for example:

```bash
bash scripts/run_record_pattern_test.sh --colmap-threads 4
```

Only gsplat training uses the GPUs. Physical GPU 6 is assigned to
`inward_orbit`, and physical GPU 7 to `inward_orbit_upward`, through a
process-local `CUDA_VISIBLE_DEVICES` setting. Override them with distinct GPU
indices if necessary:

```bash
bash scripts/run_record_pattern_test.sh \
  --gpu-inward 6 \
  --gpu-combined 7
```

## Work, results, and logs

Extracted frames, the COLMAP database, and sparse reconstruction are
disposable work products under `work/record_pattern_test/`. With the default
recipe, the layout is:

```text
work/record_pattern_test/
├── inward_orbit/
│   └── fps6_max100000_camopencv_sequential15_reg0.8/
│       ├── dataset/
│       │   ├── images/
│       │   ├── database.db
│       │   ├── metadata.json
│       │   ├── reconstruction_stats.json
│       │   └── sparse/0/
│       └── training_manifest_30000.json
└── inward_orbit_upward/
    └── fps6_max100000_camopencv_exhaustive_reg0.8_source0.8/
        ├── dataset/
        │   ├── images/
        │   ├── database.db
        │   ├── metadata.json
        │   ├── reconstruction_stats.json
        │   └── sparse/0/
        └── training_manifest_30000.json
```

Durable training outputs and per-job logs go under
`outputs/record_pattern_test/`:

```text
outputs/record_pattern_test/
├── inward_orbit/
│   ├── _logs/<UTC-timestamp>.log
│   └── fps6_max100000_camopencv_sequential15_reg0.8_30000steps/
│       ├── ckpts/
│       ├── ply/
│       ├── stats/
│       └── pipeline_manifest.json
├── inward_orbit_upward/
│   ├── _logs/<UTC-timestamp>.log
│   └── fps6_max100000_camopencv_exhaustive_reg0.8_source0.8_30000steps/
│       ├── ckpts/
│       ├── ply/
│       ├── stats/
│       └── pipeline_manifest.json
└── summary_<UTC-timestamp>.tsv
```

Changing a result-affecting option changes the directory key. The profile name
itself is not used as a folder name; the key records `7000steps` or
`30000steps` instead.

You can relocate the three roots for server storage without editing the
script:

```bash
CAPTURE_ROOT=/data/captures/record_pattern_test \
WORK_ROOT=/scratch/$USER/record_pattern_test \
OUTPUT_ROOT=/data/results/record_pattern_test \
bash scripts/run_record_pattern_test.sh
```

## Dry-run, preprocessing, and retry

Inspect both plans without creating files, acquiring the batch lock, running
COLMAP, or using a GPU:

```bash
bash scripts/run_record_pattern_test.sh --dry-run
```

Extract frames and run both COLMAP jobs, but do not train:

```bash
bash scripts/run_record_pattern_test.sh --preprocess-only
```

Rerunning the same command validates and reuses completed extraction, COLMAP,
and training stages. Source order and content, frame integrity, reconstruction
recipe, quality gates, gsplat revision, and trainer settings are checked before
reuse.

Stale or incomplete state fails closed by default. After reading the reported
paths and logs, archive the stale state and recreate it with:

```bash
bash scripts/run_record_pattern_test.sh --retry-incomplete
```

Retry moves old work or results into the corresponding scene's `_failed/`
directory; it does not silently overwrite them. Batch locks under both
`work/record_pattern_test/.batch-lock` and
`outputs/record_pattern_test/.batch-lock` prevent two copies of this launcher
from operating on the same work or result tree simultaneously. Foreign-host,
live-process, and incomplete locks fail closed instead of being overwritten.

Ctrl-C, HUP, or TERM is forwarded to both complete job process groups, including
their `uv`, Python, COLMAP, and training descendants. The launcher waits about
five seconds before escalating surviving groups to `KILL`. If it still cannot
verify cleanup, it deliberately retains `cleanup-incomplete` locks; inspect the
server processes before removing those locks manually.

The locks also record both active job process-group IDs. If the launcher itself
is forcibly killed and cannot run cleanup, the next invocation refuses to
archive its dead-owner locks while either recorded group is still alive. A
`launch-in-progress` marker also closes the brief process-spawn/metadata window;
if that marker survives a dead owner, inspect processes and remove the lock
manually only after confirming no job remains.

## Capture compatibility

The combined reconstruction assumes both videos were recorded with the same
camera, lens, resolution, orientation, zoom, and stabilization mode. Frame
extraction rejects different decoded image dimensions, but it cannot detect a
changed zoom, crop, stabilization setting, or other intrinsics change. If
those settings differ, do not interpret the shared calibration as reliable;
record the two patterns again with fixed, matching camera settings.

The two trajectories must also see enough common visual structure. Exhaustive
matching can find non-adjacent overlap, but it cannot connect clips that share
no recognizable scene content.

## Shared-server scope

The launcher does not run `module load`, install packages, call the setup
script, or synchronize dependencies. Its `uv run` calls use `--no-sync`. It
only writes the configured work/output trees and sets CUDA visibility for its
own training subprocesses, so it does not modify another user's environment.

Run `scripts/setup_gsplat.sh` separately once if this checkout has not been set
up. If your cluster requires a CUDA module, load it in your own shell before
starting the batch; that shell setup is outside this launcher.
