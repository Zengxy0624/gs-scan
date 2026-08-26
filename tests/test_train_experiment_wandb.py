from __future__ import annotations

import dataclasses
import importlib
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / "third-party" / "gsplat" / ".venv" / "bin" / "python"


def _heavy_module_stubs() -> dict[str, types.ModuleType]:
    """Stand-ins for the GPU-stack imports at the top of train_experiment."""
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: (lambda function: function)

    numpy = types.ModuleType("numpy")

    tyro = types.ModuleType("tyro")
    tyro.extras = SimpleNamespace()

    simple_trainer = types.ModuleType("simple_trainer")
    simple_trainer.Config = type("Config", (), {})
    simple_trainer.Runner = type("Runner", (), {})

    gsplat = types.ModuleType("gsplat")
    distributed = types.ModuleType("gsplat.distributed")
    distributed.cli = lambda *args, **kwargs: None
    strategy = types.ModuleType("gsplat.strategy")
    strategy.DefaultStrategy = type("DefaultStrategy", (), {})
    strategy.MCMCStrategy = type("MCMCStrategy", (), {})
    gsplat.distributed = distributed
    gsplat.strategy = strategy

    return {
        "torch": torch,
        "numpy": numpy,
        "tyro": tyro,
        "simple_trainer": simple_trainer,
        "gsplat": gsplat,
        "gsplat.distributed": distributed,
        "gsplat.strategy": strategy,
    }


def _import_train_experiment() -> types.ModuleType:
    with patch.dict(sys.modules, _heavy_module_stubs()):
        sys.modules.pop("scripts.train_experiment", None)
        module = importlib.import_module("scripts.train_experiment")
    return module


train_experiment = _import_train_experiment()


class FakeWandb(types.ModuleType):
    """Records init/log/finish calls; never touches the network."""

    def __init__(self) -> None:
        super().__init__("wandb")
        self.init_calls: list[dict[str, object]] = []
        self.log_calls: list[tuple[dict[str, object], object]] = []
        self.finish_count = 0

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)

    def log(self, payload: dict[str, object], step: object = None) -> None:
        self.log_calls.append((payload, step))

    def Image(self, path: object) -> tuple[str, str]:  # noqa: N802
        return ("wandb-image", str(path))

    def finish(self) -> None:
        self.finish_count += 1


class FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, object, object]] = []

    def add_scalar(self, tag, value, step=None, *args, **kwargs) -> str:
        self.scalars.append((tag, value, step))
        return "tb-result"


@dataclasses.dataclass
class FakeStrategy:
    cap_max: int = 6_500_000


@dataclasses.dataclass
class FakeConfig:
    result_dir: str = "outputs/ambulance/exp9_wandb"
    data_dir: Path = dataclasses.field(default_factory=lambda: Path("work/merged"))
    max_steps: int = 30_000
    strategy: FakeStrategy = dataclasses.field(default_factory=FakeStrategy)


def _fake_runner() -> SimpleNamespace:
    return SimpleNamespace(writer=FakeWriter())


class ActivationTests(unittest.TestCase):
    def test_without_wandb_project_writer_is_untouched(self) -> None:
        runner = _fake_runner()
        fake = FakeWandb()
        with patch.dict(os.environ):
            os.environ.pop("WANDB_PROJECT", None)
            with patch.dict(sys.modules, {"wandb": fake}):
                result = train_experiment._maybe_activate_wandb(
                    runner, FakeConfig(), 0
                )
        self.assertIsNone(result)
        # No wrapper installed: still the plain class method, not an
        # instance attribute (bound methods are recreated per access, so
        # identity of runner.writer.add_scalar itself cannot be compared).
        self.assertNotIn("add_scalar", vars(runner.writer))
        self.assertIsNone(getattr(runner, "_wandb", None))
        self.assertEqual(fake.init_calls, [])
        self.assertEqual(fake.log_calls, [])

    def test_nonzero_rank_never_activates(self) -> None:
        runner = _fake_runner()
        fake = FakeWandb()
        with patch.dict(os.environ, {"WANDB_PROJECT": "ambulance-3dgs"}):
            with patch.dict(sys.modules, {"wandb": fake}):
                result = train_experiment._maybe_activate_wandb(
                    runner, FakeConfig(), 1
                )
        self.assertIsNone(result)
        self.assertNotIn("add_scalar", vars(runner.writer))
        self.assertEqual(fake.init_calls, [])

    def test_activation_inits_run_and_mirrors_scalars(self) -> None:
        runner = _fake_runner()
        fake = FakeWandb()
        with patch.dict(os.environ, {"WANDB_PROJECT": "ambulance-3dgs"}):
            with patch.dict(sys.modules, {"wandb": fake}):
                result = train_experiment._maybe_activate_wandb(
                    runner, FakeConfig(), 0
                )
        self.assertIs(result, fake)
        self.assertIs(runner._wandb, fake)

        self.assertEqual(len(fake.init_calls), 1)
        init = fake.init_calls[0]
        self.assertEqual(init["name"], "exp9_wandb")
        json.dumps(init["config"])  # must be fully JSON-serializable
        self.assertEqual(init["config"]["data_dir"], "work/merged")

        returned = runner.writer.add_scalar("val/psnr", 24.56, 7000)
        self.assertEqual(returned, "tb-result")
        self.assertEqual(runner.writer.scalars, [("val/psnr", 24.56, 7000)])
        self.assertEqual(fake.log_calls, [({"val/psnr": 24.56}, 7000)])

    def test_missing_wandb_package_raises_clear_error(self) -> None:
        runner = _fake_runner()
        with patch.dict(os.environ, {"WANDB_PROJECT": "ambulance-3dgs"}):
            with patch.dict(sys.modules, {"wandb": None}):
                with self.assertRaises(RuntimeError) as context:
                    train_experiment._maybe_activate_wandb(
                        runner, FakeConfig(), 0
                    )
        message = str(context.exception)
        self.assertIn("WANDB_PROJECT", message)
        self.assertIn("install", message)
        self.assertIn("wandb", message)

    def test_experiment_runner_mirroring_defaults_off(self) -> None:
        self.assertIsNone(train_experiment.ExperimentRunner._wandb)


class MirrorWriterTests(unittest.TestCase):
    def test_non_finite_values_reach_tensorboard_but_not_wandb(self) -> None:
        runner = _fake_runner()
        fake = FakeWandb()
        train_experiment._mirror_writer_to_wandb(runner, fake)

        runner.writer.add_scalar("train/loss", float("nan"), 10)
        runner.writer.add_scalar("train/loss", float("inf"), 20)
        runner.writer.add_scalar("train/loss", float("-inf"), 30)
        runner.writer.add_scalar("train/loss", 0.5, 40)

        self.assertEqual(len(runner.writer.scalars), 4)
        self.assertEqual(fake.log_calls, [({"train/loss": 0.5}, 40)])

    def test_non_numeric_value_still_forwarded_to_tensorboard(self) -> None:
        runner = _fake_runner()
        fake = FakeWandb()
        train_experiment._mirror_writer_to_wandb(runner, fake)

        runner.writer.add_scalar("train/tag", "not-a-number", 5)

        self.assertEqual(
            runner.writer.scalars, [("train/tag", "not-a-number", 5)]
        )
        self.assertEqual(fake.log_calls, [])


class SanitizeConfigTests(unittest.TestCase):
    def test_paths_dataclasses_and_containers(self) -> None:
        sanitized = train_experiment._sanitize_config(
            {
                "result_dir": Path("/outputs/run"),
                "strategy": FakeStrategy(cap_max=3),
                "steps": [1, 2, (3, 4)],
                "nested": {"lr": 0.01, "device": Path("cuda")},
                "flag": True,
                "nothing": None,
            }
        )
        json.dumps(sanitized)  # everything JSON-serializable
        self.assertEqual(sanitized["result_dir"], "/outputs/run")
        self.assertEqual(sanitized["strategy"], str(FakeStrategy(cap_max=3)))
        self.assertEqual(sanitized["steps"], [1, 2, [3, 4]])
        self.assertEqual(sanitized["nested"], {"lr": 0.01, "device": "cuda"})
        self.assertIs(sanitized["flag"], True)
        self.assertIsNone(sanitized["nothing"])


class EvalRenderMirrorTests(unittest.TestCase):
    def test_caps_at_four_images_under_stage_key(self) -> None:
        fake = FakeWandb()
        paths = [
            f"/renders/val_step7000_{index:04d}.png"
            for index in (0, 64, 128, 192, 256)
        ]
        train_experiment._log_eval_renders(fake, "val", 7000, paths)

        self.assertEqual(len(fake.log_calls), 1)
        payload, step = fake.log_calls[0]
        self.assertEqual(step, 7000)
        self.assertEqual(list(payload), ["val_renders"])
        self.assertEqual(len(payload["val_renders"]), 4)
        self.assertEqual(payload["val_renders"][0], ("wandb-image", paths[0]))
        self.assertEqual(payload["val_renders"][3], ("wandb-image", paths[3]))

    def test_no_canvases_logs_nothing(self) -> None:
        fake = FakeWandb()
        train_experiment._log_eval_renders(fake, "val", 7000, [])
        self.assertEqual(fake.log_calls, [])


class CliHelpTests(unittest.TestCase):
    def test_help_exits_zero_with_tyro_cli_unchanged(self) -> None:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key != "WANDB_PROJECT"
        }
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [str(VENV_PYTHON), "scripts/train_experiment.py", "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mcmc", result.stdout)
        self.assertNotIn("wandb", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
