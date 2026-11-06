# Object pipeline: clips → sharp frames → COLMAP (GPU) → 3D Gaussian Splat

One scanned object (a mannequin side, a device, a room) filmed as several
non-continuous video clips with one camera and one lens becomes a trained
gsplat model in three stages. Everything below is generic: a *project* is a
capture family (`captures/<project>/`), an *object* is one merged dataset
(`work/<project>/<name>/`), and a *result* is `outputs/<project>/<name>_<tag>/`.

```text
captures/<project>/<clip>.mov + annotation.csv
        │  1. frames   prepare_object_frames.sh      (CPU, ~1 min per video-minute)
        ▼
work/<project>/<clip>/          per-clip frame cache (images/, metadata.json,
                                 sharpness_*.csv, image_metadata.csv)
work/<project>/<name>/          hardlink merge = the COLMAP dataset
        │  2. colmap   run_object_colmap.sh           (GPU features + GPU BA, hours)
        ▼
work/<project>/<name>/sparse/0  + database.db + reconstruction_stats.json
        │  3. train    train_experiment.py            (GPU, ~4 h for 3k frames @4K)
        ▼
outputs/<project>/<name>_<tag>/ ckpts/ ply/ stats/ videos/ tb/
```

`scripts/run_object_pipeline.sh` runs the three stages in order (or any
subset via `--stages`) and prints every underlying command, so each stage can
also be run by hand exactly as shown in its output.

## 0. One-time server setup

| What | How | Why |
|---|---|---|
| gsplat env (`third-party/gsplat/.venv`) | `module load cuda-toolkit/13.0 && bash scripts/setup_gsplat.sh` | pinned torch cu130 + CPU pycolmap 3.13 + gsplat CUDA extension |
| OpenBLAS pthread shim | `third-party/openblas-threaded/README.md` | pip pycolmap wheels segfault in vocab-tree matching without it |
| GPU COLMAP (cuDSS) | `bash scripts/build_gpu_colmap.sh` (tmux, ~15 min) | GPU SIFT/matching **and** GPU bundle adjustment; keeps COLMAP's direct sparse solver up to 4,000 images (CPU switches to slow iterative PCG at 1,000) |
| `loginctl enable-linger` | once per user | tmux sessions survive logout |

`run_object_colmap.sh` picks the most capable interpreter automatically:
cuDSS build (`work/_gpu_colmap/BUILD_DONE` present) → CUDA wheel venv
(`work/_colmap_cuda/venv`, optional) → gsplat venv (CPU only). The vocabulary
tree lives at `work/_colmap_cuda/vocab_tree_faiss_flickr100K_words256K.bin`
(COLMAP 3.11.1 release asset); pass `--vocab-tree` to use another.

## 1. Inputs

**Videos.** Same physical camera, same lens, same resolution and frame rate
for all clips of one object (COLMAP fits *one* shared camera). Different
lenses (e.g. 13 mm and 24 mm iPhone cameras) are different objects/datasets.
File stems are the clip names (`JC_13_1341.mov` → `JC_13_1341`).

**`annotation.csv`** in the capture directory, one row per video. Required
column: `file` (video file name). Optional, all used when present:

| column | example | used for |
|---|---|---|
| `lens` | `13mm` | EXIF `FocalLengthIn35mmFilm` → COLMAP focal prior (1,324 px at 4K for 13 mm instead of the 4,608 px default guess; refined to 1,586 px on the mannequin) |
| `iso`, `shutter`, `aperture` | `100`, `1/120`, `f2.2` | EXIF exposure tags; `image_metadata.csv` |
| `white_balance_k`, `tint`, `color` | `3490`, `2`, `Rec.709` | EXIF UserComment + CSV (appearance differences between clips) |
| `rec_time` | `2026-08-27T12:45:44-0500` | per-frame `DateTimeOriginal` = start + timestamp |
| `gps` | `+40.11075-088.22833/` | EXIF GPS |
| `annotation` | `Back: left arm` | EXIF ImageDescription |
| `make`, `model`, `software` | `Apple`, `iPhone 17 Pro`, `Blackmagic Cam 3.4` | EXIF camera identity (or pass `--make/--model/--software`) |

gsplat's loader reads the exposure tags, so the EXIF is not cosmetic.

## 2. Stage 1 — frames (`prepare_object_frames.sh`)

```bash
bash scripts/prepare_object_frames.sh --project jump_center --name mannequin_back \
  JC_13_1341 JC_14_1343 JC_15_1344 JC_16_1347 JC_17_1349 JC_18_1351 JC_19_1354
```

Per clip, then merge:

1. **Blur-aware extraction** (`extract_frames.py --sharpness-window 0.3
   --sharpness-keep-ratio 0.8`). Anchors every 1/3 s. Every source frame is
   scored (variance of Laplacian) in one decode pass; an anchor whose score is
   below 0.8× the sharpest frame within ±0.3 s is replaced by that frame.
   Picks stay strictly increasing and unique, one image per anchor. Writes
   `sharpness_scores.csv` (selected frames) and `sharpness_all_frames.csv`.
   Measured on the front mannequin: 53 % of anchors replaced, replacements
   1.27–1.36× sharper (median), visibly motion-blurred anchors gone.
2. **Residual-blur filter** (`filter_frame_cache.py --ratio-threshold 0.7`):
   drops images still below 0.7× the rolling median of their ±10 neighbours
   (whole-window blur, i.e. camera sweeps longer than 0.6 s). Front mannequin:
   92 of 3,227 (2.9 %). New hard-linked cache, renumbered; the unfiltered one
   is deleted. `--drop-ratio 0` disables.
3. **EXIF + per-image CSV** (`annotate_frame_cache.py`), lossless.
4. **Merge** (`merge_frame_datasets.py`, hardlinks, `NN_<clip>_NNNNNN.jpg`
   names; `image_metadata.csv` concatenated). The merger refuses to mix caches
   with different recipes (fps, window, keep ratio, drop ratio).

Idempotent: valid caches and a valid merged dataset are reused. Log:
`work/<project>/_logs/prepare_<name>.log`. QA helpers used so far live in the
ledger (montages under `work/<project>/_qa/`).

**Do not** reuse absolute sharpness thresholds across lenses/scenes: the
13 mm mannequin close-ups score 8–20 while sharp; the ambulance floor of 20
would delete them. Relative rules only.

## 3. Stage 2 — COLMAP (`run_object_colmap.sh`)

```bash
bash scripts/run_object_colmap.sh --dataset work/jump_center/mannequin_back --gpu 2 \
  --ba-global-frames-ratio 1.3 --ba-global-points-ratio 1.3 \
  --ba-global-max-refinements 2 --ba-global-max-num-iterations 30 --colmap-threads 64
```

Recipe (the ambulance production recipe): vocabulary-tree matching with 200
retrieved neighbours (this is what links non-continuous clips: retrieval is by
appearance, not frame order), guided matching, affine-shape + DSP-SIFT at
3,840 px, 8,192 features (note: COLMAP's cap applies to keypoint *locations*;
with orientations/affine shapes 9–12k descriptors per image are normal),
`init-min-tri-angle 8`, one shared `OPENCV` camera (use `OPENCV_FISHEYE` only
for uncorrected fisheye footage), registration gates 0.8 overall and 0.8 per
clip, snapshots every 250 registered frames.

Options that matter:

| flag | default | notes |
|---|---|---|
| `--ba-global-*` | 1.1 / 1.1 / 5 / 50 | production. **1.3 / 1.3 / 2 / 30 finished 3,135 frames in 14 h** where the CPU/production run needed >9 h for 1,247; the ledger recorded the lazier schedule as costing "nothing measurable". `run_object_pipeline.sh --ba-preset fast` sets it |
| `--vocab-tree-num-images` | 200 | 100 ≈ half the observations ≈ half the BA time |
| `--colmap-threads` | 32 | capped at 32 whenever the BLAS shim is on (pip wheels); the cuDSS build takes 64 |
| `--resume-from-snapshot` | | continue mapping from the newest snapshot (keeps database) |
| `--reuse-matches` | | remap from `database.db` with new mapper settings |
| `--retry-incomplete` | | archive a stale COLMAP state to `_failed/` and rebuild |

Outputs: `sparse/0/`, `database.db`, `reconstruction_stats.json` (per-clip
registration under `source_registration`), log
`work/<project>/_logs/colmap_<name>.log`.

What the log means:

- `Linear solver failure. Failed to compute a step: Success.` — Ceres/cuDSS hit
  a non-positive-definite pivot; the step is retried with more damping.
  Harmless (54 in the mannequin run, every BA completed).
- `Clamping features from N to 32768` — an image has >32,768 descriptors; the
  GPU matcher drops the excess for that image only.
- GPU utilisation near 0 % during mapping is expected: only the reduced camera
  system is solved on the GPU; Jacobians, Schur elimination and
  retriangulation over ~20 M observations are CPU and partly single-threaded.
  The GPU build's real win is keeping the direct sparse solver past 1,000
  images.
- Per-clip gate failure (`source_registration … failed`): rerun mapping only
  with `--reuse-matches --min-source-registration-ratio 0.5`, or switch to
  exhaustive matching on the GPU for small sets.

Time budget seen: extraction 11 min (CPU; affine/DSP SIFT has no GPU path),
matching 23 min (GPU), mapping 14 h for 3,135 frames / 20.7 M observations.

## 4. Stage 3 — training (`train_experiment.py`)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
third-party/gsplat/.venv/bin/python scripts/train_experiment.py mcmc \
  --data-dir work/jump_center/mannequin_back --data-factor 1 \
  --result-dir outputs/jump_center/mannequin_back_exp1 \
  --strategy.cap-max 6500000 --steps-scaler 4 --app-opt \
  --strategy.refine-stop-iter 29000 --opacity-reg 0.0005 \
  --eval-steps 30000 --save-steps 30000 --ply-steps 30000 --save-ply --disable-viewer \
  2>&1 | tee -a outputs/jump_center/_logs/train_mannequin_back_exp1.log
```

The exp6/exp11 production recipe: MCMC with a 6.5 M Gaussian budget, 120k
steps (`--steps-scaler 4` × 30k; eval/save/PLY at the final step 119999),
appearance optimisation (`--app-opt`, absorbs exposure differences between
clips), relocation stopped at 116k, opacity regulariser 5e-4, full 4K
(`--data-factor 1`). ~4 h and 16 GB on an RTX PRO 6000 for 3,135 frames.
Shorter runs: `--steps-scaler 1 --max-steps N --eval-steps N --save-steps N
--ply-steps N --strategy.refine-stop-iter 0.97N` (the driver's
`--train-steps`). Outputs: `stats/val_step*.json`, per-image and per-clip
CSV/JSON, `ply/point_cloud_*.ply`, `videos/traj_*.mp4`, `tb/`.

Reference numbers (mannequin_front_exp1, 392 val images): PSNR 29.4, SSIM
0.951, LPIPS 0.155; per clip 27–31.5 dB. The ambulance interior reached
24.5 / 0.81 / 0.36 — expect object scans to land far above that.

**Background.** Half of the pixels/observations are room, so roughly half the
budget lands on the background; the object still gets ~3 M Gaussians. If the
object needs more, mask the background (segmentation masks in the loss) or
crop the exported PLY. Nothing in the recipe needs to change for a baseline.

## 5. Viewing and exporting

- Interactive, true training colours: the training command with
  `--ckpt outputs/…/ckpts/ckpt_119999_rank0.pt --view-only --port 8080`
  (drop `--disable-viewer`); `ssh -L 8080:localhost:8080 <server>`, open
  http://localhost:8080. ~1 min to start, ~3 GB GPU.
- Re-render the trajectory video: same flags with `--ckpt …` (no
  `--view-only`), ~35 min.
- PLY in SuperSplat / the Scenes repo: `ply/point_cloud_119999.ply` (neutral
  baked appearance, 364 MB for 6.5 M Gaussians).

## 6. Driver

```bash
tmux new -s back
bash scripts/run_object_pipeline.sh --project jump_center --name mannequin_back --gpu 2 \
  JC_13_1341 JC_14_1343 JC_15_1344 JC_16_1347 JC_17_1349 JC_18_1351 JC_19_1354
```

`--stages frames,colmap,train` (default all), `--ba-preset fast|production`,
`--train-steps 120000|…|12000`, `--tag`, `--camera-model`,
`--vocab-tree-num-images`, `--feature-max-num-features`, `--cap-max`,
`--prepare-arg …`, `--colmap-arg …`, `--dry-run`. New recordings: drop the
videos in `captures/<project>/`, add their rows to `annotation.csv`, pass the
stems. A single-clip object (`prepare_object_frames.sh` needs two) is
prepared by hand: `extract_frames.py` → `filter_frame_cache.py` →
`annotate_frame_cache.py`, then `run_object_colmap.sh --dataset <cache>`; or
run it end to end with `gs_from_video.sh` (§8).

## 7. Conventions

- Long jobs run in tmux with logs under `work/<project>/_logs/` and
  `outputs/<project>/_logs/`; the scripts append to those logs and hold a lock
  per dataset.
- Datasets are never edited in place; curation produces new hardlinked
  directories. The merger and validator enforce identical extraction recipes.
- Every script has `--dry-run` and a `tests/test_<stem>.py`; run the suite
  with `third-party/gsplat/.venv/bin/python -m unittest discover -s tests`.
- Record every finished dataset/run in `docs/EXPERIMENTS.md`.

## 8. Single video (`gs_from_video.sh`)

```bash
tmux new -s <clip>
bash scripts/gs_from_video.sh <project> <clip> <gpu>
```

One clip (`captures/<project>/<clip>.MOV` plus its `annotation.csv` row) runs
end to end into `work/<project>/<clip>/` and `outputs/<project>/<clip>/`.
Frames use §2's settings (steps 1–3, no merge); COLMAP runs as CLI commands of
the cuDSS build with §3's extraction and matching settings. It differs from
the production recipe in four places:

| | production (§3–4) | `gs_from_video.sh` | why |
|---|---|---|---|
| mapping | incremental | `global_mapper` (GLOMAP), COLMAP defaults | 1,393 frames, same matches: 37 min vs 7 h 11 min, all registered in one model; per-frame pose difference median 0.62 mm / 0.027°, P99 3.8 mm / 0.26° (metric scale from a robot arm in the scene) |
| Gaussian budget | 6.5 M | 1 M | the splat is rendered at every simulation step |
| `--app-opt` | on | off | real2sim-eval's `GSProcessor` reads SH degree 3 only and cannot load the PLY it exports; exposure is locked during capture |
| `--pose-opt` | off | on | refines the COLMAP poses of the training frames |

After mapping the script requires exactly one model with ≥ 80 % of the frames
registered, then writes `colmap_global.txt` (recipe + model checksums); a rerun
reuses COLMAP only when it matches. COLMAP threads = min(32, available RAM /
2.5 GB): CPU affine + DSP-SIFT at 4K peaks at ~1.9 GB per thread.

Failure modes seen:

- **Misplaced cameras.** In symmetric scenes, or in frames that see almost no
  texture, a few frames land in the wrong place and neither the registration
  count nor the mean reprojection error shows it (1,393 frames: incremental
  mapping flipped 8 frames by ~180°, global mapping misplaced 1 frame holding
  6 points). One camera placed at 683× the median distance (a frame showing
  only a black cloth and a white object) made gsplat's scene scale 698 instead
  of 2.6–2.8 and training diverged (held-out PSNR 16.4 dB); a dataset without
  the 59 frames holding < 50 points trained to 26.7 dB. The gate above does
  not catch this: before training, check the max / median camera distance
  from the centre (normal < 2) and the points per frame.
- **Held-out scores with `--pose-opt`.** Only training frames get refined
  poses; every 8th (eval) frame keeps its COLMAP pose, so PSNR / LPIPS read
  low and are not comparable to runs without pose optimisation.
- **Stopping training early.** `--steps-scaler 4` also multiplies
  `--eval-steps` / `--save-steps` / `--ply-steps`; with 30000 they fire only at
  the last step, so a run stopped midway leaves no checkpoint or PLY (1 M
  Gaussians: ~1 h on an idle RTX PRO 6000).

Environment notes (any entry point):

- **No visible GPU at install time** (e.g. a cluster login node): fused-ssim's
  `setup.py` ignores `TORCH_CUDA_ARCH_LIST` and reads only
  `CUDA_ARCHITECTURES` or the detected GPU; with neither it builds sm_75/80/89
  and training elsewhere fails with "no kernel image". Set
  `CUDA_ARCHITECTURES` (120 for RTX PRO 6000, 80 for A100) and check with
  `cuobjdump --list-elf`.
- **Ubuntu 22.04.** The vendored OpenBLAS shim is the 24.04 package (needs
  GLIBC 2.38): regenerate it from the local apt packages as its README says
  (22.04 gives OpenBLAS 0.3.20 pthread, `MAX_THREADS=64`; check with
  `tests/test_openblas_shim.py`). nvcc 13.3 / 13.4 cannot compile the torch
  2.9.1 extensions (`List_inl.h:201`: need 'typename'); run `setup_gsplat.sh`
  with `CUDA_HOME` and `PATH` pointing at a 13.0 toolkit. micromamba may
  detect the driver's CUDA as 11.8, and `build_gpu_colmap.sh` then cannot
  solve its environment; prefix `CONDA_OVERRIDE_CUDA=<driver CUDA, e.g. 13.2>`.
- **Vocabulary tree.** Download
  `vocab_tree_faiss_flickr100K_words256K.bin` to `work/_colmap_cuda/` by hand;
  the built-in downloader fails.
