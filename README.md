# Ambulance 3DGS

A small, server-oriented pipeline for reconstructing an ambulance as a
3D Gaussian Splat:

```text
video -> sampled images -> PyCOLMAP cameras/sparse model -> gsplat training
```

This repository intentionally contains no semantic-mapping or VGGT-omega code.
Frame extraction is a lightweight OpenCV step.

This standalone pipeline was extracted from the 3DGS path in
[gxywang/3dmap](https://github.com/gxywang/3dmap) at commit `134ed07`.
Its setup versions and server-operation safeguards are intentionally kept
aligned with that reference pipeline.

## Repository layout

```text
3DGS-Health/
├── scripts/
│   ├── extract_frames.py        # one video -> dataset/images/*.jpg
│   ├── prepare_colmap.py        # images -> database.db + sparse/0
│   ├── run_device_test.sh       # one-command, five-video comparison batch
│   ├── build_run_manifest.py    # bind results to source/model/code provenance
│   ├── validate_dataset_cache.py
│   └── setup_gsplat.sh          # pinned server environment
├── docs/
│   └── device_test.md            # five-video server batch runbook
├── tests/                       # local-safe regression and smoke tests
├── captures/                    # ignored raw inputs
├── work/                        # ignored frames and COLMAP state
├── outputs/                     # ignored durable gsplat results
└── third-party/
    └── gsplat/                   # official pinned Git submodule
```

The scripts and tests are project-owned. `third-party/gsplat` is an unmodified
checkout of the official
[nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
repository, pinned by this repository's Git commit. Training and viewing call
its upstream example programs directly.

There is deliberately no root Python environment. The setup script creates one
environment at `third-party/gsplat/.venv`, and every pipeline stage uses it.

## 1. Clone and set up the server

Clone recursively, or initialize the submodule after cloning:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The server needs `git`, `uv`, Python 3.11, a C/C++ compiler, Ninja, an NVIDIA
driver, and a CUDA toolkit compatible with the selected PyTorch wheel. If the
cluster uses environment modules:

```bash
module load cuda-toolkit/13.0
nvidia-smi
nvcc --version
uv --version
```

Create the environment:

```bash
bash scripts/setup_gsplat.sh
```

The unchanged environment policy carried over from `3dmap` defaults to:

- Python 3.11
- PyTorch 2.9.1 and torchvision 0.24.1
- CUDA 13.0 wheels (`cu130`)
- NumPy 2.x
- CPU PyCOLMAP `>=3.13,<4`

To match a CUDA 12.8 server instead:

```bash
uv venv --clear --python 3.11 third-party/gsplat/.venv
GSPLAT_TORCH_BACKEND=cu128 bash scripts/setup_gsplat.sh
```

Always clear the existing environment before changing CUDA backends. Otherwise,
the unchanged `torch==2.9.1` requirement or an existing compiled gsplat
extension may leave the previous backend installed.

Keep `--no-sync` on the commands below. It prevents `uv` from trying to
reconcile the deliberately assembled upstream environment.

## Additional runbooks

- [Device comparison batch](docs/device_test.md)

## 2. Capture the ambulance

Use one physical camera, fixed lens/zoom/resolution/orientation, and keep the
ambulance and its equipment static. Move slowly, avoid motion blur, and preserve
large overlap between views while adding lateral movement rather than only
rotating in place. Reflective metal, glass, and plain white panels benefit from
several oblique views with textured surroundings visible.

Record the interior and exterior separately unless there is a long, textured,
well-observed transition through the doorway. Keep raw video outside the repo
or under ignored `captures/`.

## 3. Extract a fresh dataset

Start with 3 sampled frames per second and a high, effectively non-binding
cap of 100,000 frames:

```bash
uv run --project third-party/gsplat --no-sync \
  python scripts/extract_frames.py \
  /absolute/path/to/ambulance.MOV \
  work/ambulance_input_v1 \
  --sample-fps 3 \
  --max-frames 100000

DATASET="$(realpath work/ambulance_input_v1)"
```

The cap does not truncate the video: frames are distributed uniformly from the
beginning through the end. The command refuses an existing dataset directory,
so use versioned names for retries.

The output is:

```text
ambulance_input_v1/
├── images/
│   ├── 000000.jpg
│   └── ...
└── metadata.json
```

Before reconstruction, inspect the images for correct rotation, sharpness,
stable exposure, continuous motion, sufficient overlap, and accidental lens
switches.

## 4. Recover cameras with COLMAP

```bash
uv run --project third-party/gsplat --no-sync \
  python scripts/prepare_colmap.py "$DATASET" \
  --overlap 15 \
  --num-threads 8
```

The defaults use one shared `SIMPLE_RADIAL` camera, a dense sequential
video-frame match window plus sparse longer-baseline pairs, incremental
structure from motion, and the largest recovered model. `--num-threads` bounds
feature extraction, matching, and mapping, not just the first two stages.
The complete dataset then contains:

```text
ambulance_input_v1/
├── images/
├── metadata.json
├── database.db
├── reconstruction_stats.json
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

The command has a default-on hard quality gate: the largest model must register
at least 50% of the input and at least 10 images when 10 or more are available.
Set a deliberate threshold with `--min-registration-ratio 0.7`. Use the short
pilot's validation renders as the visual geometry gate. If registration is
weak, create a new dataset at 5 FPS, remove blurred frames, or recapture with
more overlap and texture. Increasing `--overlap` helps only when the useful
dense matching window is wider than 15 frames.

## 5. Run a short pilot

Use one GPU and disable the interactive viewer for unattended runs:

```bash
CUDA_VISIBLE_DEVICES=0 uv run \
  --project third-party/gsplat --no-sync \
  python third-party/gsplat/examples/simple_trainer.py default \
  --data_dir "$DATASET" \
  --data_factor 1 \
  --result_dir outputs/ambulance_pilot_v1 \
  --max_steps 7000 \
  --save_ply \
  --disable_viewer
```

Review validation renders and the PLY before paying for the full run. The
upstream simple trainer does not resume this pilot into a longer run.

## 6. Run full training

```bash
CUDA_VISIBLE_DEVICES=0 uv run \
  --project third-party/gsplat --no-sync \
  python third-party/gsplat/examples/simple_trainer.py default \
  --data_dir "$DATASET" \
  --data_factor 1 \
  --result_dir outputs/ambulance_v1 \
  --save_ply \
  --disable_viewer
```

Important upstream behavior:

- `--data_factor 1` reads `images/`. Factors such as 2 or 4 expect prepared
  `images_2/` or `images_4/` directories; they do not resize automatically.
- The default coordinate system is normalized for reconstruction and is not
  metric without an external scale reference.
- Prefer a single GPU. The example's multi-rank PLY/output behavior is not a
  production distributed job manager.
- Keep `--disable_viewer` on batch jobs so they terminate after training.

Results include checkpoints, validation renders and metrics, a rendered
trajectory video, and a PLY when `--save_ply` is enabled.

## 7. View a result

On a machine where the viewer port is accessible:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project third-party/gsplat --no-sync \
  python third-party/gsplat/examples/simple_viewer.py \
  --ckpt outputs/ambulance_v1/ckpts/ckpt_29999_rank0.pt \
  --output_dir outputs/ambulance_v1/viewer \
  --port 8080
```

Confirm the actual checkpoint filename with
`ls outputs/ambulance_v1/ckpts/`. On a remote server, forward the port over SSH
instead of exposing it publicly.

## Local smoke tests

After setup, the lightweight project-owned tests can run without starting
training:

```bash
uv run --project third-party/gsplat --no-sync \
  python -m unittest discover -s tests -v
```

The extraction integration test uses OpenCV from the gsplat environment and
writes only to a temporary directory. COLMAP orchestration is tested with an
in-memory stand-in; real reconstruction still belongs in the server validation
run.

## What was intentionally excluded

- semantic segmentation, semantic maps, and object pipelines;
- VGGT, VGGT-omega, and VGGT-SLAM;
- project-wide setup/environment files from `3dmap`;
- a custom trainer or changes inside upstream gsplat;
- local GPU/COLMAP integration tests that do not represent the target server.
