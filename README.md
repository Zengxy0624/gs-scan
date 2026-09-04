# 3DGS-Health

Server-side pipeline that turns handheld video of a medical-simulation scene
or object into a 3D Gaussian Splat:

```text
video clips -> sharp frames (blur-aware) -> COLMAP (GPU) -> gsplat training -> viewer / PLY
```

Two capture families so far: the **ambulance interior** (18 clips, iPhone 16
Pro Max, 2026-08) and the **Jump Center** mannequin and equipment (21 clips,
iPhone 17 Pro, 2026-08-27). Everything runs on the shared shenlong GPU servers
without root.

## Start here

- **`docs/OBJECT_PIPELINE.md`** — the runbook: inputs, the three stages, every
  knob, timings, log messages and failure modes, viewing.
- **`docs/EXPERIMENTS.md`** — the ledger: datasets, runs, numbers, verdicts,
  queue. Update it when something finishes.
- **`scripts/README.md`** — per-script index.
- `CLAUDE.md` — repository conventions (shared GPUs, immutable datasets, tmux).

## Repository layout

```text
3DGS-Health/
├── scripts/                 # pipeline (driver, stages, building blocks)
├── tests/                   # one test_<stem>.py per script; run with the gsplat venv
├── docs/                    # OBJECT_PIPELINE.md runbook, EXPERIMENTS.md ledger
├── archive/                 # retired scripts/tests/runbooks — see archive/README.md
├── captures/<project>/      # ignored: raw videos + annotation.csv
├── work/<project>/          # ignored: frame caches, merged datasets, COLMAP models
├── work/_gpu_colmap/        # ignored: cuDSS pycolmap build (scripts/build_gpu_colmap.sh)
├── outputs/<project>/       # ignored: training results
└── third-party/
    ├── gsplat/              # pinned upstream submodule — never modify
    └── openblas-threaded/   # pthread OpenBLAS shim for pip pycolmap wheels
```

## Setup (once per server)

```bash
git submodule update --init --recursive
module load cuda-toolkit/13.0          # nvcc for the gsplat CUDA extension
bash scripts/setup_gsplat.sh           # third-party/gsplat/.venv (torch cu130, pycolmap 3.13)
# restore the OpenBLAS shim binaries: third-party/openblas-threaded/README.md
bash scripts/build_gpu_colmap.sh       # GPU bundle adjustment (tmux, ~15 min)
loginctl enable-linger                 # tmux survives logout
```

## Run one object

```bash
tmux new -s back
bash scripts/run_object_pipeline.sh --project jump_center --name mannequin_back --gpu 2 \
  JC_13_1341 JC_14_1343 JC_15_1344 JC_16_1347 JC_17_1349 JC_18_1351 JC_19_1354
```

Frames in minutes, COLMAP in hours, training ~4 h; results under
`outputs/jump_center/mannequin_back_exp1/`. Details and the by-hand commands:
`docs/OBJECT_PIPELINE.md`.

## Tests

```bash
third-party/gsplat/.venv/bin/python -m unittest discover -s tests
```

Fast and local: pycolmap is stubbed, extraction uses a synthetic video, the
shell wrappers are exercised in `--dry-run`. Real reconstructions are
validated on the server and recorded in the ledger. The retired suite lives
apart and is not part of that run:

```bash
third-party/gsplat/.venv/bin/python -m unittest discover -s archive/tests -t .
```

## Not in this repository

Raw captures, datasets, models and results (all ignored), the cuDSS build
itself (rebuilt from `scripts/build_gpu_colmap.sh`), and any changes inside
`third-party/gsplat`.
