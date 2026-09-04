#!/usr/bin/env python3
"""Instrumented gsplat training for experiment runs.

A thin wrapper around third-party/gsplat/examples/simple_trainer.py that adds
the diagnostics repeated tuning rounds have needed, without modifying the
gsplat submodule (which must stay clean for manifest provenance):

1. Per-source validation metrics: PSNR/SSIM/LPIPS bucketed by the merged
   dataset's ``NN_<video>_`` image-name prefix, so exposure or quality problems
   can be traced to individual capture videos
   (``stats/val_by_source_step*.json``).
2. Per-image validation metrics (``stats/val_per_image_step*.csv``) to locate
   *where* in the scene quality is bad.
3. Opacity and scale distribution scalars written to TensorBoard every 1,000
   steps (``dist/*``), making fog formation and reset-recovery visible live
   instead of requiring post-hoc checkpoint analysis.

W&B mirroring: set ``WANDB_PROJECT=ambulance-3dgs`` in the environment to
mirror every TensorBoard scalar plus up to four validation canvases per eval
to Weights & Biases (add ``WANDB_MODE=offline`` to record locally and upload
later with ``wandb sync``). No new CLI flags; without ``WANDB_PROJECT`` set,
behavior is unchanged.

Deviation from upstream eval: side-by-side render PNGs are written for every
8th validation image instead of all of them (a full pass writes gigabytes).

Usage is identical to simple_trainer.py, e.g.:
  python scripts/train_experiment.py mcmc --strategy.cap-max 6500000 ...
  python scripts/train_experiment.py default --strategy.absgrad ...

Checkpoint modes (pass the same recipe flags as the training run):
  --ckpt CKPT             render ``videos/traj_<step>.mp4`` from a checkpoint
                          (restores the ``--app-opt`` module too, unlike upstream)
  --ckpt CKPT --view-only serve the interactive viser viewer on ``--port``
                          (default 8080) from a checkpoint and wait; no eval,
                          no trajectory render. Forward the port over SSH.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence

_EXAMPLES = Path(__file__).resolve().parents[1] / "third-party" / "gsplat" / "examples"
sys.path.insert(0, str(_EXAMPLES))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

import simple_trainer as st  # noqa: E402
from simple_trainer import Config, Runner  # noqa: E402
from gsplat.distributed import cli  # noqa: E402
from gsplat.strategy import DefaultStrategy, MCMCStrategy  # noqa: E402


def _source_prefix(image_name: str) -> str:
    """'05_a001_08071524_c009_000123.jpg' -> '05_a001_08071524_c009_'."""
    stem = image_name.rsplit("_", 1)
    return stem[0] + "_" if len(stem) == 2 else image_name


class ExperimentRunner(Runner):
    #: wandb module handle when mirroring is active on this rank (see main()).
    _wandb: ModuleType | None = None

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Upstream eval plus per-image and per-source metric dumps."""
        print("Running instrumented evaluation...")
        cfg = self.cfg
        device = self.device
        if self.world_rank != 0:
            return super().eval(step=step, stage=stage)

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0.0
        metrics = defaultdict(list)
        per_image: list[dict[str, object]] = []
        mirror_canvases: list[str] = []
        import imageio.v2 as imageio

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            if i % 8 == 0:
                canvas = (
                    torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                )
                canvas_path = f"{self.render_dir}/{stage}_step{step}_{i:04d}.png"
                imageio.imwrite(canvas_path, (canvas * 255).astype(np.uint8))
                if i % 64 == 0:
                    mirror_canvases.append(canvas_path)

            pixels_p = pixels.permute(0, 3, 1, 2)
            colors_p = colors.permute(0, 3, 1, 2)
            psnr = self.psnr(colors_p, pixels_p)
            ssim = self.ssim(colors_p, pixels_p)
            lpips = self.lpips(colors_p, pixels_p)
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["lpips"].append(lpips)

            name = self.valset.parser.image_names[self.valset.indices[i]]
            per_image.append(
                {
                    "index": i,
                    "name": name,
                    "source": _source_prefix(name),
                    "psnr": float(psnr),
                    "ssim": float(ssim),
                    "lpips": float(lpips),
                }
            )

        ellipse_time /= max(len(valloader), 1)
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update(
            {"ellipse_time": ellipse_time, "num_GS": len(self.splats["means"])}
        )
        print(
            f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, "
            f"LPIPS: {stats['lpips']:.3f} Time: {stats['ellipse_time']:.3f}s/image "
            f"Number of GS: {stats['num_GS']}"
        )
        with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        for k, v in stats.items():
            self.writer.add_scalar(f"{stage}/{k}", v, step)

        by_source: dict[str, dict[str, float]] = {}
        for source in sorted({row["source"] for row in per_image}):
            rows = [r for r in per_image if r["source"] == source]
            by_source[source] = {
                "count": len(rows),
                "psnr": float(np.mean([r["psnr"] for r in rows])),
                "ssim": float(np.mean([r["ssim"] for r in rows])),
                "lpips": float(np.mean([r["lpips"] for r in rows])),
            }
            self.writer.add_scalar(
                f"{stage}_source_psnr/{source}", by_source[source]["psnr"], step
            )
        with open(
            f"{self.stats_dir}/{stage}_by_source_step{step:04d}.json", "w"
        ) as f:
            json.dump(by_source, f, indent=2)
        with open(
            f"{self.stats_dir}/{stage}_per_image_step{step:04d}.csv", "w", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=["index", "name", "source", "psnr", "ssim", "lpips"]
            )
            writer.writeheader()
            writer.writerows(per_image)

        worst = sorted(by_source.items(), key=lambda kv: kv[1]["psnr"])[:3]
        print(
            "Worst sources: "
            + "; ".join(f"{k} {v['psnr']:.2f}dB (n={v['count']})" for k, v in worst)
        )
        if self._wandb is not None:
            _log_eval_renders(self._wandb, stage, step, mirror_canvases)
        self.writer.flush()


def _wrap_strategy_distribution_logging(runner: Runner) -> None:
    """Log opacity/scale distributions to TensorBoard every 1,000 steps."""
    strategy = runner.cfg.strategy
    original = strategy.step_post_backward

    def wrapped(params, optimizers, state, step, *args, **kwargs):
        if step % 1000 == 0:
            with torch.no_grad():
                opacity = torch.sigmoid(params["opacities"].detach())
                scale = torch.exp(params["scales"].detach()).max(dim=-1).values
                quantiles = torch.tensor([0.1, 0.5, 0.9], device=opacity.device)
                op_q = torch.quantile(opacity.float(), quantiles)
                writer = runner.writer
                writer.add_scalar("dist/opacity_p10", op_q[0].item(), step)
                writer.add_scalar("dist/opacity_p50", op_q[1].item(), step)
                writer.add_scalar("dist/opacity_p90", op_q[2].item(), step)
                writer.add_scalar(
                    "dist/opacity_frac_below_0.1",
                    (opacity < 0.1).float().mean().item(),
                    step,
                )
                writer.add_scalar(
                    "dist/opacity_frac_above_0.9",
                    (opacity > 0.9).float().mean().item(),
                    step,
                )
                writer.add_scalar(
                    "dist/scale_p50",
                    torch.quantile(scale.float(), 0.5).item(),
                    step,
                )
        return original(params, optimizers, state, step, *args, **kwargs)

    strategy.step_post_backward = wrapped


def _sanitize_config(config: dict[str, object]) -> dict[str, object]:
    """Coerce config values W&B cannot JSON-serialize (Paths, dataclasses)."""

    def sanitize(value: object) -> object:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): sanitize(item) for key, item in value.items()}
        return str(value)

    return {str(key): sanitize(value) for key, value in config.items()}


def _mirror_writer_to_wandb(runner: Runner, wandb_module: ModuleType) -> None:
    """Send every TensorBoard scalar to W&B as well (same tag, same step)."""
    writer = runner.writer
    original = writer.add_scalar

    def wrapped(tag, scalar_value, global_step=None, *args, **kwargs):
        result = original(tag, scalar_value, global_step, *args, **kwargs)
        try:
            value = float(scalar_value)
        except (TypeError, ValueError):
            return result
        if math.isfinite(value):
            wandb_module.log({tag: value}, step=global_step)
        return result

    writer.add_scalar = wrapped


def _log_eval_renders(
    wandb_module: ModuleType,
    stage: str,
    step: int,
    canvas_paths: Sequence[str],
) -> None:
    """Log up to four just-written side-by-side canvases for one eval pass."""
    if not canvas_paths:
        return
    images = [wandb_module.Image(path) for path in canvas_paths[:4]]
    wandb_module.log({f"{stage}_renders": images}, step=step)


def _maybe_activate_wandb(
    runner: Runner, cfg: Config, world_rank: int
) -> ModuleType | None:
    """Mirror the TensorBoard stream to W&B when WANDB_PROJECT is set.

    Activation is environment-driven (``wandb.init`` natively picks up
    WANDB_PROJECT, WANDB_MODE, ...), so the tyro-generated CLI stays
    unchanged. Returns the wandb module when mirroring is active.
    """
    if world_rank != 0 or not os.environ.get("WANDB_PROJECT"):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WANDB_PROJECT is set but the wandb package is not installed; "
            "install it in the training environment (pip install wandb) "
            "or unset WANDB_PROJECT"
        ) from exc
    wandb.init(name=Path(cfg.result_dir).name, config=_sanitize_config(vars(cfg)))
    _mirror_writer_to_wandb(runner, wandb)
    runner._wandb = wandb
    return wandb


@dataclass
class ExperimentConfig(Config):
    """Upstream Config plus the checkpoint viewer switch."""

    # With --ckpt: start the viser viewer from the checkpoint and wait instead
    # of rendering the trajectory video. Requires the viewer to be enabled.
    view_only: bool = False


def _restore_checkpoint(runner: Runner, cfg: Config) -> int:
    """Load splats and the ``--app-opt`` module from ``cfg.ckpt``; return step.

    Upstream's eval-only branch restores the splats but silently leaves an
    ``--app-opt`` appearance module randomly initialized, which garbles the
    colors. This restores both.
    """
    ckpts = [
        torch.load(file, map_location=runner.device, weights_only=True)
        for file in cfg.ckpt
    ]
    for key in runner.splats.keys():
        runner.splats[key].data = torch.cat(
            [ckpt["splats"][key] for ckpt in ckpts]
        )
    runner.scene = st.GaussianScene.from_splats(runner.splats, id="scene")
    runner.splats = runner.scene.splats
    runner.stage = st.Stage()
    runner.stage.add_scene(runner.scene, runner.rasterize_splats)
    app_state = ckpts[0].get("app_module")
    app_module = getattr(runner, "app_module", None)
    if app_module is not None:
        if app_state is None:
            raise RuntimeError(
                "--app-opt is set but the checkpoint has no app_module state"
            )
        module = getattr(app_module, "module", app_module)
        module.load_state_dict(app_state)
    return int(ckpts[0]["step"])


def _render_traj_from_checkpoint(runner: Runner, cfg: Config) -> None:
    """Load a training checkpoint and render only the trajectory video."""
    step = _restore_checkpoint(runner, cfg)
    print(f"Rendering trajectory from checkpoint at step {step}")
    runner.render_traj(step=step)


def _serve_viewer_from_checkpoint(
    runner: Runner,
    cfg: Config,
    sleep: "Callable[[float], None]" = time.sleep,
) -> None:
    """Load a checkpoint and keep the interactive viser viewer running."""
    if cfg.disable_viewer or getattr(runner, "viewer", None) is None:
        raise RuntimeError("--view-only needs the viewer enabled (drop --disable-viewer)")
    step = _restore_checkpoint(runner, cfg)
    runner.viewer.complete()
    print(
        f"Viewing checkpoint at step {step}: open http://localhost:{cfg.port} "
        "(forward the port over SSH, e.g. ssh -L "
        f"{cfg.port}:localhost:{cfg.port} <host>). Ctrl+C to exit.",
        flush=True,
    )
    while True:
        sleep(3600)


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config) -> None:
    if cfg.post_processing is not None:
        raise NotImplementedError(
            "post_processing is not wired through the experiment wrapper"
        )
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True

    runner = ExperimentRunner(local_rank, world_rank, world_size, cfg)
    if world_rank == 0:
        _wrap_strategy_distribution_logging(runner)
    if cfg.ckpt is not None:
        if getattr(cfg, "view_only", False):
            _serve_viewer_from_checkpoint(runner, cfg)
        else:
            _render_traj_from_checkpoint(runner, cfg)
        return
    wandb_module = _maybe_activate_wandb(runner, cfg, world_rank)
    try:
        runner.train()
    finally:
        if wandb_module is not None:
            wandb_module.finish()


if __name__ == "__main__":
    configs = {
        "default": (
            "Densification heuristics from the original paper (instrumented).",
            ExperimentConfig(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "MCMC strategy with a fixed Gaussian budget (instrumented).",
            ExperimentConfig(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
