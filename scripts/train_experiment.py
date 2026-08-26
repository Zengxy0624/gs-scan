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

Deviation from upstream eval: side-by-side render PNGs are written for every
8th validation image instead of all of them (a full pass writes gigabytes).

Usage is identical to simple_trainer.py, e.g.:
  python scripts/train_experiment.py mcmc --strategy.cap-max 6500000 ...
  python scripts/train_experiment.py default --strategy.absgrad ...
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

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
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    (canvas * 255).astype(np.uint8),
                )

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
        raise NotImplementedError("eval-only mode: use simple_trainer.py directly")
    runner.train()


if __name__ == "__main__":
    configs = {
        "default": (
            "Densification heuristics from the original paper (instrumented).",
            Config(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "MCMC strategy with a fixed Gaussian budget (instrumented).",
            Config(
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
