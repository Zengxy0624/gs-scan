# gs-scan

Turns a handheld phone video of a scene or object into a 3D Gaussian Splat
(PLY), on Linux GPU machines without root:

```text
video -> sharp frames (blur-aware) -> COLMAP: features, matching, global mapping (GLOMAP) -> gsplat training -> PLY
```

The three COLMAP steps are subcommands of one `colmap` binary, the cuDSS build
from `scripts/build_gpu_colmap.sh`: `feature_extractor`, `vocab_tree_matcher`
and `global_mapper`. `global_mapper` is GLOMAP, built into COLMAP since 4.0; no
standalone glomap is used. `scripts/gs_from_video.sh` maps this way and needs
the cuDSS build. The production recipe for several clips,
`scripts/run_object_pipeline.sh`, maps incrementally through pycolmap and
falls back to a CUDA wheel or CPU without the cuDSS build.

Started from Wenzhou Ding's 3DGS-Health (git remote `upstream`), built for
medical-simulation scenes: the ambulance interior and the Jump Center
mannequin (`docs/EXPERIMENTS.md`). Here it reconstructs the sim_rl robot scene
(`captures/lab_scan/`, iPhone 16 Pro Max + Blackmagic Cam with locked
exposure) for rendering in simulation.

## Start here

- **`docs/OBJECT_PIPELINE.md`** — the runbook: inputs, the three stages, every
  knob, timings, log messages and failure modes, viewing; §8 is
  `gs_from_video.sh` and the per-machine environment notes.
- **`scripts/README.md`** — per-script index.
- `docs/EXPERIMENTS.md` — upstream's ledger for the ambulance and Jump Center
  runs.
- `CLAUDE.md` — upstream's repository conventions, not yet updated for
  gs-scan; `export_cameras.py` does not follow its argparse / `main()` rule, and
  neither new script has the test it asks for.

## Repository layout

```text
gs-scan/
├── scripts/                 # pipeline (drivers, stages, building blocks)
├── tests/                   # unittest suite; run with the gsplat venv
├── docs/                    # OBJECT_PIPELINE.md runbook, EXPERIMENTS.md ledger
├── archive/                 # retired scripts/tests/runbooks — see archive/README.md
├── captures/<project>/      # ignored: raw videos + annotation.csv
├── work/<project>/          # ignored: frame caches, merged datasets, COLMAP models
├── work/_gpu_colmap/        # ignored: cuDSS COLMAP build (scripts/build_gpu_colmap.sh)
├── work/_colmap_cuda/       # ignored: vocabulary tree (and the optional CUDA-wheel venv)
├── work/_logs/              # ignored: build and run logs
├── outputs/<project>/       # ignored: training results
└── third-party/
    ├── gsplat/              # pinned upstream submodule — never modify
    └── openblas-threaded/   # pthread OpenBLAS shim for pip pycolmap wheels
```

## Setup (once per machine)

```bash
git submodule update --init --recursive
module load cuda-toolkit/13.0          # nvcc for the gsplat CUDA extension
bash scripts/setup_gsplat.sh           # third-party/gsplat/.venv (torch cu130, pycolmap 3.13)
bash scripts/build_gpu_colmap.sh       # cuDSS COLMAP, in tmux; --cuda-arch 120 (default, RTX PRO 6000) or 89 (RTX 4090)
# download vocab_tree_faiss_flickr100K_words256K.bin (COLMAP 3.11.1 release asset) to work/_colmap_cuda/ by hand;
#   the built-in downloader fails
# restore the OpenBLAS shim binaries: third-party/openblas-threaded/README.md
#   (used by pip-wheel COLMAP and tests/test_openblas_shim.py; gs_from_video.sh does not need it)
loginctl enable-linger                 # tmux survives logout
```

`build_gpu_colmap.sh` took 7–9 min on an RTX 4090 workstation with
`--jobs 16`. Without the `module` command (e.g. an Ubuntu 22.04
workstation), run `setup_gsplat.sh` with `CUDA_HOME` and `PATH` pointing at a
CUDA 13.0 toolkit (nvcc 13.3 / 13.4 cannot build the torch 2.9.1 CUDA
extensions; fused-ssim fails first). If
`build_gpu_colmap.sh` cannot solve its environment because micromamba reads
the driver's CUDA as 11.8, prefix it with `CONDA_OVERRIDE_CUDA=<driver CUDA,
e.g. 13.2>`. More per-machine notes: `docs/OBJECT_PIPELINE.md` §8.

## Run one video

From the repo root:

```bash
tmux new -s <clip>
mkdir -p work/_logs && PYTHONUNBUFFERED=1 bash scripts/gs_from_video.sh <project> <clip> <gpu> 2>&1 \
  | tee -a work/_logs/gs_<clip>.log
```

Inputs: the video at exactly `captures/<project>/<clip>.MOV`, and one row in
`captures/<project>/annotation.csv` whose `file` column is `<clip>.MOV`; set
`lens` too, it gives COLMAP its focal prior (other columns:
`docs/OBJECT_PIPELINE.md` §1). `<gpu>` is one `nvidia-smi` index, used for
both COLMAP and training. The camera make / model / software written to EXIF
are fixed in the script (iPhone 16 Pro Max, Blackmagic Cam 3.5.200045) and
override those columns of `annotation.csv`.

Frames, the three COLMAP steps and gsplat training run in one go; the PLY
lands in `outputs/<project>/<clip>/ply/`. The script writes no log and takes
no lock itself, hence the `tee`. When training starts, check gsplat's
`Scene scale:` line: our scans gave 1.8–2.2; a camera placed far away by
COLMAP gave 698 and the run diverged. If it is far above ~3, stop with Ctrl-C
and see `docs/OBJECT_PIPELINE.md` §8, "Misplaced cameras" (the script's gate,
one model with ≥ 80 % registered, does not catch it).

On a rerun, frames and COLMAP results that still validate are reused. An
existing `outputs/<project>/<clip>/`, or a `database.db` / `sparse/` left by
an interrupted COLMAP step, stops the script with "move it aside".

Measured with this script: 307 frames on an RTX 4090 — frames 9 min, COLMAP
10 min, training ~2 h 50 min; on an RTX PRO 6000, training 1,393 frames took
51 min. Camera poses in the frame of the PLY, for aligning the splat to
another frame: `third-party/gsplat/.venv/bin/python scripts/export_cameras.py
work/<project>/<clip> outputs/<project>/<clip>/cameras.npz`.

The recipe and its four differences from the production one are in §8;
several clips of one object with the production recipe:
`scripts/run_object_pipeline.sh` (§6).

## Tests

```bash
third-party/gsplat/.venv/bin/python -m unittest discover -s tests
```

Fast and local: pycolmap is stubbed, extraction uses a synthetic video, the
pipeline shell wrappers run in `--dry-run`. `gs_from_video.sh`,
`export_cameras.py` and `setup_gsplat.sh` have no tests yet. Upstream's real
reconstructions are recorded in `docs/EXPERIMENTS.md`; the sim_rl lab scans in
sim_rl's `docs/09_scene_scan.md` and `daily/`. The retired suite lives apart
and is not part of that run:

```bash
third-party/gsplat/.venv/bin/python -m unittest discover -s archive/tests -t .
```

## Not in this repository

Raw captures, datasets, models and results (all ignored), the cuDSS build
itself (rebuilt from `scripts/build_gpu_colmap.sh`), the vocabulary tree, the
OpenBLAS shim binaries, and any changes inside `third-party/gsplat`.
