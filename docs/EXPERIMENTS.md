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

Blur filter: variance-of-Laplacian per frame; `ratio` = raw / rolling median of
±10 same-video neighbors. Lesson: the ratio misses blurry *segments* (whole
fast sweeps where every neighbor is smeared → ratio≈1); the absolute floor
`raw<20` catches those (blind-rating validated). Filtered datasets are built by
`scripts/filter_sharp_dataset.py` — frames are deregistered from a copy of the
COLMAP model, no new bundle adjustment, base never modified.

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
| `exp11_boot_sharp` | merged_gpu_r1_sharp050a20 | bootstrap + aggressive deblur combined | (running) | | | | production-model candidate |

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
- [ ] **exp11** (running): compare vs exp7 and exp10 on shared frames; if it
      wins both → production model, pruned PLY export
- [ ] Advisor asks: wandb tracking for gsplat training; web viz of sparse SfM
      points + camera poses (rerun.io or custom)
- [ ] Round 2 (decision-gated): dense 60fps sampling of the two c004 door
      windows (~300 frames) → register into exp11's dataset → retrain
- [ ] Ceiling capture pass (when filming is possible)
- [ ] Reserve lever: `--steps-scaler 8` (240k steps) on the final dataset
- [ ] VR export (`exp4b .../point_cloud_119999_vr500k.ply`, 28 MB) render-check
      still pending a free GPU

## Operational contracts

- **Matching env (any pycolmap matching run)**: `LD_PRELOAD` the pthread
  OpenBLAS shim + `OPENBLAS_NUM_THREADS=1` + `OMP_NUM_THREADS=32`, all set AT
  LAUNCH (the preloaded shim reads them before Python starts).
  `register_new_frames.py` refuses to run otherwise. See
  `third-party/openblas-threaded/README.md`.
- **Shared GPUs**: check `nvidia-smi` for other users before any GPU work.
- **Long jobs**: tmux + `loginctl enable-linger` (login-session teardown killed
  a 76-min BA and all tmux servers on 2026-08-22).
- **Viewer note**: exported PLYs bake colors at neutral appearance (no
  view-dependence, no per-image exposure) — judge color/texture from the
  `videos/traj_*.mp4` renders; use `ply/point_cloud_*_pruned005.ply` exports in
  viewers (identical quality, ~25–45% smaller).
