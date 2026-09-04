# Ambulance 3DGS — experiment ledger

Single source of truth for every reconstruction, training run, and verdict.
Update this file whenever an experiment finishes. Metric conventions: PSNR/SSIM
higher is better, LPIPS lower is better; all evals at native 4K
(`--data-factor 1`) on the run's own validation split (every 8th image by
sorted name). **Headline numbers are NOT comparable across different datasets**
(each dataset reshuffles the val split); cross-run verdicts below always use
the shared-identical-frames method: intersect the two runs' per-image CSVs
(`stats/val_per_image_step*.csv`) and compare means on the common frames only.

## Datasets

| Dataset (`work/ambulance/`) | Images | Registered | Points | Notes |
|---|---|---|---|---|
| `merged_gpu` | 8,012 | 7,995 (99.8%) | 6,499,411 | THE base. 18 videos @3fps, vocab-tree + single OPENCV camera, GPU BA. Mean reproj 1.40 px |
| `merged_vocabtree_single_opencv` | 8,012 | 7,995 | 6,447,599 | Pure-CPU twin (~1.5 weeks). Reproj 1.389 px. Agrees with GPU model to 0.009% of scene radius / 0.11° after sim3 — indistinguishable |
| `merged_gpu_sharp040` | 7,347 | 7,346 | 6,232,370 | Mild blur cut: ratio<0.40 (665 removed) |
| `merged_gpu_sharp050a20` | 6,515 | 6,515 | 5,825,141 | Aggressive cut: ratio<0.50 OR raw<20 (1,497 removed) |
| `merged_gpu_r1` | 9,572 | 9,556 | 7,030,921 | Bootstrap round 1: base + 1,560 blur-gated midpoint candidates, all registered (100%), reproj improved to 1.372 px |
| `merged_gpu_r1_sharp050a20` | 8,075 | 8,075 | 6,399,706 | r1 minus the 1,497 aggressive-cut base frames; candidates kept. exp11's data |
| `boot_r1_candidates` | 1,560 | — | — | Round-1 candidates (sampled 1,705 midpoints from 5 weak videos + worst-200 neighborhoods; 145 blur-rejected) |
| `merged_gpu_r2` | 10,185 | 10,168 | 7,195,356 | Bootstrap round 2: r1 + 613 candidates (dense 60fps door windows, c017 midpoints, exp11 worst-150 neighborhoods), 612/613 registered |
| `merged_gpu_r2_sharp050a20` | 8,688 | 8,687 | 6,576,977 | r2 minus aggressive-cut base frames; both candidate rounds protected. exp12's data |

Blur filter: variance-of-Laplacian per frame; `ratio` = raw / rolling median of
±10 same-video neighbors. Lesson: the ratio misses blurry *segments* (whole
fast sweeps where every neighbor is smeared → ratio≈1); the absolute floor
`raw<20` catches those (blind-rating validated). Filtered datasets are built by
`archive/scripts/filter_sharp_dataset.py` — frames are deregistered from a copy
of the COLMAP model, no new bundle adjustment, base never modified.

## Training runs

All runs use `scripts/train_experiment.py` (instrumented: per-image/per-source
val CSVs, opacity/scale distributions in TensorBoard) except run1/exp1–exp5
which predate it. Common recipe since exp4b: `mcmc --strategy.cap-max 6500000
--steps-scaler 4 --app-opt` (= 120k steps); exp6 added
`--strategy.refine-stop-iter 29000` (relocation to 116k) and
`--opacity-reg 0.0005`, which every later run keeps.

| Run (`outputs/ambulance/`) | Dataset | What changed | PSNR | SSIM | LPIPS | Gaussians | Verdict |
|---|---|---|---|---|---|---|---|
| `vocabtree_single_opencv_30000steps` (run1) | merged_gpu | stock default, 30k steps | 21.49 | 0.781 | 0.465 | 3,367,508 | baseline; white fog (57% splats <0.1 opacity) |
| `exp_ss4_app_opreg` (exp1) | merged_gpu | ss4 + app + AA + opacity-reg 0.01 | 22.13 | — | — | 332,445 | FAIL: opacity reg + resets = near-extinction (bottomed at 173,816 GS / 16.93 @28k, partially recovered by 120k; still far below peers) |
| `exp2_ss4_app_absgrad` | merged_gpu | ss4, app, AA, absgrad, grow 0.0008, prune 0.02 | 24.05 | 0.812 | 0.399 | 2,454,359 | schedule+appearance worth ~+2.6 dB |
| `exp3_grow` | merged_gpu | grow 0.0004, prune 0.01 | 19.40 | — | — | 4,273,349 | FAIL: population churn, never converges |
| `exp4_mcmc` | merged_gpu | MCMC preset (opacity-reg 0.01, AA) | — | — | — | — | FAIL: opacity collapse → relocation storm → CUDA crash @13.6k |
| `exp4b_mcmc` | merged_gpu | MCMC cap 6.5M, opacity-reg 0.001, no AA | 24.56 | 0.822 | 0.366 | 6,500,000 | first MCMC success; 53% dead splats (relocation stops @100k) |
| `exp5_prune005` | merged_gpu | exp2 + prune 0.005 | 24.37 | 0.817 | 0.385 | 3,693,531 | good, beaten by exp4b |
| `exp6_mcmc_fullrefine` | merged_gpu | + refine-stop 116k, opacity-reg 0.0005 | 24.60 | 0.823 | 0.356 | 6,500,000 | dead frac 53%→35% (+1.1M live); **unfiltered-base reference** |
| `exp7_boot_r1` | merged_gpu_r1 | exp6 recipe + bootstrap data | 24.46* | 0.818 | 0.367 | 7,030,921 | **bootstrap works**: +0.28 dB vs exp6 on 168 shared frames; targeted videos +0.31…+0.77, others +0.17. (*headline depressed: val includes ~195 hard candidate frames) |
| `exp8_sharp040` | merged_gpu_sharp040 | exp6 recipe + mild deblur | 24.64* | 0.820 | 0.352 | 6,500,000 | LPIPS −0.0075 vs exp6 on 122 shared frames; PSNR flat |
| `exp10_sharp050a20` | merged_gpu_sharp050a20 | exp6 recipe + aggressive deblur | 24.68* | 0.813 | 0.349 | 6,500,000 | LPIPS −0.0056 vs exp8 on 95 shared frames (72/95 improved); PSNR flat |
| `exp11_boot_sharp` | merged_gpu_r1_sharp050a20 | bootstrap + aggressive deblur combined | 24.54* | 0.812 | 0.359 | 6,500,000 | **PRODUCTION MODEL**: beats exp7 (+0.15 dB, LPIPS −0.011 on 96 shared) and edges exp10 (+0.03 dB, LPIPS −0.002 on 173 shared); only model with both data fixes. Use `ply/point_cloud_119999_pruned005.ply` |

(*) headline on that run's own split — see comparability note at top.

## Key verdicts

- **Levers ranked by yield**: schedule+appearance +2.6 dB ≫ capacity (MCMC)
  +0.5 ≫ targeted data (bootstrap) +0.28 ≫ recycler fix +0.05 ≈ deblur
  (LPIPS-only). Recipe tuning is exhausted; data quality/quantity is what
  moves.
- **Bootstrap round 1 (advisor's hard-example mining): validated.** 100%
  candidate registration, +0.28 dB overall on shared frames, gains
  concentrated in the targeted videos. Scorer localized the worst remaining
  region to two windows in the door video: c004 source frames ~6270–6430 and
  ~10330–10430 (reflective glass) → round-2 targets.
- **Dirty-wall diagnosis**: speckle texture collapsing into grey smears on
  *solid* splats (not floaters — pruning all <0.05-opacity splats costs
  ~0.01 dB and looks identical). Scene/coverage-driven (exp4b-vs-exp5
  per-image r=0.88).
- **Ceiling**: capture gap (~3% of frames look up), unfixable in software.
  Needs one 60–90 s up-facing pass (film at ~45°, not straight up, so
  registration has wall context).
- **CPU vs GPU COLMAP**: identical models (see Datasets); the lazier GPU BA
  schedule costs nothing measurable at ~10× less wall time.
- **PSNR ceiling**: ~24.5–25 dB is near the GT noise floor (motion blur,
  operator reflections). Judge future runs by LPIPS + eyes.

## Queue / progress

- [x] Bootstrap pipeline built, unit-tested, validated end-to-end on real data
- [x] Round 1: sample → blur-gate → register → score → exp7
- [x] Deblur arm: exp8 (mild), exp10 (aggressive)
- [x] **exp11 = production model** (wins both pairwise comparisons); pruned
      PLY exported; traj video not rendered (tmux died mid-render, cosmetic)
- [x] Advisor asks: wandb mirroring in `train_experiment.py`
      (`WANDB_PROJECT=...`), `scripts/export_rerun_viz.py` (.rrd), plus a
      shareable WebGL SfM explorer artifact; splat webviews live in the
      user's SuperSplat scenes repo (WenzhouDing/3DGS-Health-Scenes, PLY→SOG
      — feed it the `_pruned005.ply` exports)
- [x] Round 2 registered (612/613; door windows dense + c017 + exp11 worst
      neighborhoods, aggressive blur gate) → exp12 next
- [ ] Ceiling capture pass (when filming is possible)
- [ ] Reserve lever: `--steps-scaler 8` (240k steps) on the final dataset
- [ ] VR export (`exp4b .../point_cloud_119999_vr500k.ply`, 28 MB) render-check
      still pending a free GPU

## Jump Center capture (2026-08-27) — mannequin + objects

Second scene set, captured on an **iPhone 17 Pro** (Blackmagic Cam, 3840x2160
HEVC 60 fps, no rotation metadata) under `captures/jump_center/` with
`annotation.csv`. Five objects, treated as **separate datasets** (shared
single-camera assumption holds only within one lens):

| Object | Clips | Lens | Notes |
|---|---|---|---|
| Front mannequin | JC_01–JC_11 (11 clips, ~18 min) | 13 mm f/2.2 | in progress (frames extracted) |
| Back mannequin | JC_13–JC_19 (7 clips) | 13 mm f/2.2 | queued |
| Transparent tube | JC_12 | 24 mm f/1.8 | queued |
| Zoll ECG | JC_20 | 24 mm f/1.8 | queued |
| Airway opener | JC_21 | 24 mm f/1.8 | queued; shutter 1/279 + WB 3320 K differ from the rest (appearance only) |

### Front-mannequin frame caches (`work/jump_center/<clip>/`)

Extraction recipe: `extract_frames.py --sample-fps 3 --max-frames 100000
--sharpness-window 0.3 --sharpness-keep-ratio 0.8` — 3 fps anchors; an
anchor whose variance-of-Laplacian is below 0.8x the sharpest frame within
±0.3 s (±18 source frames) is replaced by that frame (blur handled at
extraction; see `scripts/README.md`). Per-frame scores are in each cache's
`sharpness_scores.csv` / `sharpness_all_frames.csv`. All 11 caches validate
(`validate_dataset_cache.py ... --sharpness-window 0.3 --sharpness-keep-ratio 0.8`).

| Clip | Content | Source frames | Selected | Anchors moved | Median gain |
|---|---|---:|---:|---:|---:|
| JC_01_1245 | front, medium height | 2,054 | 103 | 57 | 1.27x |
| JC_02_1248 | front, medium height | 5,918 | 296 | 78 | — |
| JC_03_1250 | front, low height | 4,751 | 238 | 80 | — |
| JC_04_1253 | front, high height | 4,108 | 206 | 73 | — |
| JC_05_1254 | front, head | 4,535 | 227 | 148 | 1.31x |
| JC_06_1257 | front, head (long) | 12,751 | 638 | 436 | 1.36x |
| JC_07_1308 | front, left arm | 6,973 | 349 | 206 | 1.29x |
| JC_08_1313 | front, right arm | 8,566 | 429 | 234 | 1.27x |
| JC_09_1317 | front, right foot | 4,866 | 244 | 145 | 1.28x |
| JC_10_1318 | front, left foot | 4,841 | 242 | 133 | 1.28x |
| JC_11_1320 | front, upper body | 5,089 | 255 | 129 | 1.25x |
| **Total** | | | **3,227** | **1,719** | |

("—" = median anchor kept, i.e. fewer than half moved.) Batch log:
`work/jump_center/_logs/extract_front_mannequin.log`. Extraction of the 11
clips took ~7 min wall (16 sharpness threads; ~15 s per 2,000 source frames).

**Residual-blur filter (`<clip>_sharp070`)**: after selection, 92 of 3,227
images still had `ratio_local < 0.7` (sharpness under 0.7x the rolling median
of their ±10 neighbors). 54 of them were anchors kept by the 0.8 rule because
their whole ±0.3 s window was blurred (best neighbor only ~1.2x sharper), 29
were already the window maximum — i.e. camera sweeps longer than the window,
not a selection bug. They are spread over 60 short spots (≤1.7 s each) in all
11 clips; the head clips hold 50 (JC_06: 39, JC_05: 11). Removed with
`scripts/filter_frame_cache.py --ratio-threshold 0.7` into hard-linked copies
hard-linked copies which then **replaced the unfiltered caches in place**
(`work/jump_center/<clip>/` now hold 3,135 images; the unfiltered 3,227-image
caches were deleted; all validate with `--sharpness-drop-ratio 0.7`). Log:
`work/jump_center/_logs/filter_front_mannequin_sharp070.log`; QA montages:
`work/jump_center/_qa/` (around-threshold pairs, removed blurry anchors,
residual blur).

**Per-image capture metadata**: every cache image carries EXIF built from its
video's row in `captures/jump_center/annotation.csv`
(`scripts/annotate_frame_cache.py --make Apple --model "iPhone 17 Pro"
--software "Blackmagic Cam 3.4.100022"`): ISO, shutter, f-number,
`FocalLengthIn35mmFilm` (13), fixed white balance, per-frame
`DateTimeOriginal` (+ sub-second), GPS, description, and a JSON `UserComment`
with source video/frame index/timecode/WB Kelvin/tint. Per-video values differ
only in exposure: JC_01 ISO 86 · 1/106, JC_02–03 ISO 101 · 1/122, JC_04–11
ISO 100 · 1/120 (all f/2.2, 3490 K, tint 2). Each cache also has
`image_metadata.csv`; the merged dataset concatenates them.
COLMAP reads the 35 mm-equivalent tag: its focal prior becomes **1,324 px**
(13 mm over the 43.27 mm film diagonal) instead of the default
1.2 × width ≈ 4,608 px — a far better start for the ultra-wide lens.

### `mannequin_front` — the COLMAP dataset (front mannequin)

| Dataset (`work/jump_center/`) | Images | Registered | Points | Notes |
|---|---|---|---|---|
| `mannequin_front` | 3,135 | **3,135 (100%)**, every clip 100% | 2,162,739 | 11 annotated, blur-filtered caches (3 fps · window 0.3 s · keep 0.8 · drop ratio<0.7), `image_metadata.csv`. COLMAP 4.2.0 (cuDSS build), single OPENCV camera, EXIF focal prior 1,324 px → refined **fx 1586 / fy 1585**, cx 1920 cy 1080, k1 0.0098 k2 −0.0062; 20.7 M observations (6,616/image), track length 9.6, **mean reproj 1.452 px**. Log `work/jump_center/_logs/colmap_mannequin_front.log`, QA `work/jump_center/_qa/mannequin_front_sfm_overview.png` |

COLMAP timeline (`mannequin_front`, 2026-09-01/02): extraction 11 min (CPU:
affine/DSP SIFT has no GPU path) + matching 23 min (GPU 2, CUDA wheel) →
CPU mapping with the production schedule (1.1/1.1/5/50): 1,247 frames in
9.3 h, 92 % of it global BA; per-BA time jumped 16 → 199 min past 1,000
images because COLMAP's CPU path switches from the direct sparse solver to
iterative PCG at `max_num_images_direct_sparse_cpu_solver = 1000` (the cuDSS
path keeps the direct solver to 4,000). Stopped; resumed from the 1,002
snapshot with the cuDSS build, GPU BA, 64 threads and the **lazier schedule
1.3 / 1.3 / 2 refinements / 30 iterations**: 1,002 → 3,135 frames in 13.8 h
(09:58 → 23:46). Global BA durations 16 @1284, 23 @1593, 51 @1864, 76 @2208,
87 @2564, 135 @3012 frames, final BA 289 min. 54 Ceres warnings "Linear
solver failure. Failed to compute a step" = cuDSS reported a non-positive-
definite pivot; LM re-damps and retries, harmless (all BAs completed). GPU
utilisation stays near 0 % by design: only the reduced camera-system solve is
on the GPU; Jacobians/Schur elimination/retriangulation over 20 M
observations are CPU (and partly single-threaded). **Levers for the next
objects**: observations, not GPU — `--vocab-tree-num-images 100` and/or
`--feature-max-num-features 4096` roughly halve BA time.

Lessons so far:
- **The ambulance absolute blur floor (`raw < 20`) does not transfer.** On the
  13 mm lens and low-texture content (smooth mannequin skin, plain sheet) most
  frames of the close-up clips JC_07–JC_11 score below 20 even when visibly
  sharp; `raw` there measures texture, not blur. Use relative rules (ratio to
  neighbors / in-window keep-ratio) for this capture; recalibrate any absolute
  floor per lens+scene before reusing `--raw-floor` in
  `scripts/filter_frame_cache.py`.
- Keep-ratio sweep on JC_01 (window 0.3 s): 0.7 moves 25/103 anchors, 0.8
  moves 57, 1.0 (always take the max) moves 99 and makes spacing irregular
  (gaps 1–53 frames vs uniform 20). 0.8 chosen as default: removes most of
  the visibly blurred anchors while keeping sharp anchors on the uniform grid.
- Consecutive picks can land 1–2 frames apart when neighboring anchors move
  toward each other (near-duplicate views). Harmless for SfM; a minimum-gap
  constraint is a possible refinement if frame budget matters.

### COLMAP environment on shenlong-gpu-02 (2026-09-01)

- **No cuDSS pycolmap here**: the ambulance GPU-BA env (`work/_gpu_colmap`,
  source-built on gpu-01) was not carried over; PyPI `pycolmap-cuda12` 4.2.0
  and conda-forge COLMAP 4.1.1 (Ceres 2.2) both lack cuDSS → GPU bundle
  adjustment silently falls back to CPU. cuDSS itself is on PyPI
  (`nvidia-cudss-cu13`) if a source build is wanted later.
- **What we have**: `work/_colmap_cuda/venv` = `pycolmap-cuda12` 4.2.0
  (`has_cuda`): GPU SIFT extraction + GPU matching via
  `prepare_colmap.py --feature-gpu`; retrieval and BA on CPU. Statically links a
  *serial* OpenBLAS → needs the pthread shim like the 3.13 wheel. Vocab tree:
  `work/_colmap_cuda/vocab_tree_faiss_flickr100K_words256K.bin` (COLMAP 3.11.1
  release asset, same tree pycolmap 3.13 downloads).
- **Third BLAS crash mode**: `--num-threads -1` (256) + `OMP_NUM_THREADS=64`
  overflowed the shim's 64-caller limit → SIGSEGV in `sgemm` during vocab-tree
  indexing. 32 + 32 runs clean; `run_object_colmap.sh` caps both at 32
  whenever the shim is on.
- **pycolmap 4.x vs 3.13**: `extract_features` lost the `camera_model` kwarg
  (handled), `ba_global_max_num_iterations` default became -1 → the wrapper
  passes the production schedule 1.1 / 1.1 / 5 / 50 explicitly.
- **Smoke test** (60 consecutive JC_06 images, full recipe, GPU 2, CUDA wheel):
  60/60 registered, 54,790 points, mean reprojection 1.22 px, focal prior
  1,323.7 px from EXIF, 2 min 44 s. Log
  `work/jump_center/_logs/smoke_colmap_cuda_wheel.log`.
- **Non-continuous clips**: the 11 front clips are separate takes of one
  object; vocabulary-tree retrieval (200 neighbors) links them by appearance
  exactly as it did for the 18 ambulance clips, and the 0.8 per-video gate
  reports any clip that fails to join (`reconstruction_stats.json →
  source_registration`). Fallback if a clip fails: rerun mapping with
  `--reuse-matches` and a lower `--min-source-registration-ratio`, or switch
  to exhaustive matching on the GPU (~4.9 M pairs).
- **GPU BA build (2026-09-02)**: `scripts/build_gpu_colmap.sh` recreates the
  cuDSS environment from source (Ceres master = 2.3-dev, the only Ceres with
  cuDSS; COLMAP 4.2.0 so the running job's `database.db` stays reusable via
  `--reuse-matches`; CUDA arch 120). Build log
  `work/_logs/build_gpu_colmap.log`, tmux `gpu-colmap-build`. Once
  `work/_gpu_colmap/BUILD_DONE` exists the wrapper switches to `[cudss]` and
  turns `--ba-use-gpu` on automatically.
  Build lessons (all folded into the script's spec): conda-forge splits cuDSS
  into `libcudss` (runtime) + `libcudss-dev` (headers, `cudss-config.cmake`);
  installing the dev package unpinned swapped the whole stack to **CUDA 12.9**
  (the cuda-12 libcudss variant) — kept, since Blackwell needs only >= 12.8 and
  the 595 driver runs it; `openimageio`'s CMake config additionally requires
  `fmt` and `tbb-devel`. Bash gotcha fixed in the script: a phase function
  called inside `if` ignores `set -e`, which marked a failed COLMAP configure
  as done; phases now run as plain statements under an ERR trap.
- **GPU BA available (2026-09-02 09:28)**: build finished — pycolmap 4.2.0
  (`has_cuda`), Ceres 2.3-dev linked to `libcudss.so.0`, conda pthreads
  OpenBLAS (256 procs, no shim needed), interpreter
  `work/_gpu_colmap/mamba_root/envs/gpubuild/bin/python`, marker
  `work/_gpu_colmap/BUILD_DONE`. Smoke test (60 JC_06 images, full recipe,
  `--feature-gpu --ba-use-gpu`, GPU 2): "Bundle adjustment on GPU 0", no
  cuDSS fallback warning, 2.9 GB GPU memory, 60/60 registered, 54,953 points,
  reproj 1.216 px (identical quality to the CPU-BA smoke; speed gains only
  show at scale). `run_object_colmap.sh --dry-run` now reports
  `[cudss] … bundle adjustment on`. Log
  `work/jump_center/_logs/smoke_colmap_gpu_ba.log`.
- **Handover** (user runs in tmux; GPU 2):
  `bash scripts/run_object_colmap.sh` (defaults = `mannequin_front`,
  GPU 2, OPENCV, production recipe, snapshots every 250 frames; resume with
  `--resume-from-snapshot`). Log: `work/jump_center/_logs/colmap_mannequin_front.log`.
  To move a running CPU-BA job onto GPU BA: stop it (Ctrl-C in its tmux;
  the lock is released), then `--resume-from-snapshot` — the database and
  snapshots were written by COLMAP 4.2.0 and are read by this 4.2.0 build.

Queue (jump_center):
- [x] Front mannequin: frames extracted + validated (3,227 images)
- [x] Front mannequin: residual blur removed, caches replaced in place (3,135 images)
- [x] Front mannequin: per-video EXIF + `image_metadata.csv` embedded; merged →
      `work/jump_center/mannequin_front` (3,135 images), validated
- [x] COLMAP GPU path prepared + smoke-tested (`run_object_colmap.sh`,
      CUDA wheel, shim thread contract)
- [x] Front mannequin: COLMAP on `mannequin_front` — passed, 3,135/3,135,
      reproj 1.45 px (see table)
- [x] Front mannequin: **`mannequin_front_exp1` trained** (2026-09-03, GPU 2,
      exp6/exp11 recipe, 120k steps, 4.2 h): val PSNR **29.36** / SSIM
      **0.951** / LPIPS **0.155** on 392 val images (every 8th), 6.5 M
      Gaussians, 16.1 GB GPU peak, 0.035 s/image render. Per clip PSNR 27.1
      (JC_03 low view) … 31.5 (JC_07 left arm); LPIPS 0.143–0.164, no weak
      clip. Far above the ambulance (24.5 / 0.81 / 0.36): static, well-lit
      object, no operator reflections. Artifacts:
      `ply/point_cloud_119999.ply` (364 MB), `ckpts/ckpt_119999_rank0.pt`,
      `videos/traj_119999.mp4` (first render was cut off by the session ending;
      re-rendered from the checkpoint on 2026-09-03 08:23: 3,124 frames,
      7680×2160). Log
      `outputs/jump_center/_logs/train_mannequin_front_exp1.log`. Background
      note: ~37 % of SfM points / ~50 % of observations are room, so roughly
      half the Gaussian budget sits on the background (see the 2026-09-03
      capacity analysis); masking is the lever if the mannequin needs more.
      Visual check (`work/jump_center/_qa/mannequin_front_exp1_val_canvases.png`,
      GT | render): mannequin, bed and near room indistinguishable from GT at
      half scale; only far background (cables, second bed) slightly softer.
      Worst val images 22.9–23.5 dB are JC_11 upper-body close-ups and a JC_06
      head frame; best 35+ dB on JC_07 arm views.
- Training handover record (kept for reference; the 12k-step variant is
      `--steps-scaler 1 --max-steps 12000 --eval-steps/--save-steps/--ply-steps
      12000 --strategy.refine-stop-iter 11600`) →
      `outputs/jump_center/mannequin_front_exp1`. Smoke-tested 2026-09-03 (30
      steps): gsplat parser reads the 4.2.0 model (OPENCV → on-the-fly
      undistortion), picks up the embedded EXIF exposure for all 3,135 images,
      initializes 2.16 M Gaussians, evaluates and writes PLY. SfM overview
      (`_qa/mannequin_front_sfm_overview.png`): the 11 takes form one tight
      camera ring around the mannequin inside the room point cloud — coherent
      single model, no split clusters. Launch (user, tmux):
      `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
      third-party/gsplat/.venv/bin/python scripts/train_experiment.py mcmc
      --data-dir work/jump_center/mannequin_front --data-factor 1 --result-dir
      outputs/jump_center/mannequin_front_exp1 --strategy.cap-max 6500000
      --steps-scaler 4 --app-opt --strategy.refine-stop-iter 29000
      --opacity-reg 0.0005 --eval-steps 30000 --save-steps 30000 --ply-steps
      30000 --save-ply --disable-viewer` (120k steps; final artifacts at step
      119999).
- [ ] **Back mannequin** (JC_13–JC_19, 13 mm) — run the three commands below
- [ ] Tube (JC_12), Zoll ECG (JC_20), airway opener (JC_21): 24 mm clips,
      one dataset each (`prepare_object_frames.sh` needs ≥2 clips; for a
      single clip use `extract_frames.py` + `filter_frame_cache.py` +
      `annotate_frame_cache.py` directly and point COLMAP at the cache)
- [ ] GPU bundle adjustment: no cuDSS pycolmap on shenlong-gpu-02 (PyPI
      `pycolmap-cuda12` and conda-forge builds lack cuDSS); either copy the old
      `work/_gpu_colmap` env from gpu-01 or source-build; CPU BA otherwise

### Object pipeline — how to run it (e.g. back of the mannequin)

Full runbook: `docs/OBJECT_PIPELINE.md`. One command in tmux (GPU 2):

```bash
bash scripts/run_object_pipeline.sh --project jump_center --name mannequin_back --gpu 2 \
  JC_13_1341 JC_14_1343 JC_15_1344 JC_16_1347 JC_17_1349 JC_18_1351 JC_19_1354
```

= `prepare_object_frames.sh` (3 fps, window 0.3 s, keep 0.8, drop ratio<0.7,
EXIF from `captures/jump_center/annotation.csv`) → `run_object_colmap.sh`
(production recipe, cuDSS GPU build, BA preset *fast* 1.3/1.3/2/30, 64
threads) → `train_experiment.py` (exp6/exp11 recipe, 120k steps). Stages can
be selected (`--stages`), the driver prints each underlying command, and
`--dry-run` shows the plan. Add `--vocab-tree-num-images 100` to halve COLMAP
time. Viewer: training flags + `--ckpt … --view-only --port 8080`.

Retired 2026-09-03 (superseded by the generic scripts above; recipes preserved
in the ledger). Deleted: `sample_ambulance_frames.sh`,
`run_ambulance_colmap.sh`, `train_ambulance_gs.sh`, `run_single_video_gs.sh`,
`run_device_test.sh`, `run_record_pattern_test.sh` and their tests. Moved to
`archive/` (still runnable, unmaintained — see `archive/README.md`): the
curation and hard-example-mining tools `filter_sharp_dataset.py`,
`sample_bootstrap_frames.py`, `register_new_frames.py`,
`score_candidate_frames.py`, the standalone `build_run_manifest.py`, and the
`device_test.md` / `record_pattern_test.md` runbooks. The ambulance datasets
and runs recorded above were produced with these on shenlong-gpu-01.

Housekeeping 2026-09-03: removed COLMAP snapshots (6 GB), build trees and the
micromamba package cache, the superseded CUDA-wheel venv (vocab tree kept at
`work/_colmap_cuda/`), smoke-test datasets/logs and setup scaffolding;
`work/` 33 → 20 GB. Kept: run logs, QA montages (`work/jump_center/_qa/`),
`work/_logs/{setup_gsplat,build_gpu_colmap}.log`.

## Operational contracts

- **Matching env (any pycolmap matching run)**: `LD_PRELOAD` the pthread
  OpenBLAS shim + `OPENBLAS_NUM_THREADS=1` + `OMP_NUM_THREADS=32`, all set AT
  LAUNCH (the preloaded shim reads them before Python starts).
  `archive/scripts/register_new_frames.py` refuses to run otherwise. See
  `third-party/openblas-threaded/README.md`.
- **Shared GPUs**: check `nvidia-smi` for other users before any GPU work.
- **Long jobs**: tmux + `loginctl enable-linger` (login-session teardown killed
  a 76-min BA and all tmux servers on 2026-08-22).
- **Interactive viewer from a checkpoint**: `scripts/train_experiment.py <recipe
  flags> --ckpt <ckpt> --view-only --port 8080` (viser; `ssh -L
  8080:localhost:8080 shenlong-gpu-02` then http://localhost:8080). Restores
  the appearance module, so colors match training; renders live from the
  6.5 M-Gaussian checkpoint on the chosen GPU.
- **Viewer note**: exported PLYs bake colors at neutral appearance (no
  view-dependence, no per-image exposure) — judge color/texture from the
  `videos/traj_*.mp4` renders; use `ply/point_cloud_*_pruned005.ply` exports in
  viewers (identical quality, ~25–45% smaller).
