from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import scripts.train_experiment as te


class FakeParam:
    def __init__(self, data: object) -> None:
        self.data = data


class FakeRunner:
    def __init__(self, with_app: bool, splat_keys=("means", "opacities")) -> None:
        self.device = "cpu"
        self.splats = {k: FakeParam(None) for k in splat_keys}
        self.rasterize_splats = object()
        self.rendered_at: int | None = None
        if with_app:
            self.app_loaded = None
            runner = self

            class App:
                def load_state_dict(self, state):
                    runner.app_loaded = state

            self.app_module = App()

    def render_traj(self, step: int) -> None:
        self.rendered_at = step


class RenderTrajFromCheckpointTests(unittest.TestCase):
    def _patched(self, ckpt: dict):
        fake_torch = types.SimpleNamespace(
            load=lambda *a, **k: ckpt,
            cat=lambda tensors: ("cat", tuple(tensors)),
        )
        fake_scene = types.SimpleNamespace(splats={"means": FakeParam(1)})
        fake_st = types.SimpleNamespace(
            GaussianScene=types.SimpleNamespace(
                from_splats=lambda splats, id: fake_scene
            ),
            Stage=lambda: types.SimpleNamespace(add_scene=lambda *a, **k: None),
        )
        return (
            patch.object(te, "torch", fake_torch),
            patch.object(te, "st", fake_st),
        )

    def test_loads_splats_and_app_module_then_renders(self) -> None:
        ckpt = {
            "step": 119999,
            "splats": {"means": "m", "opacities": "o"},
            "app_module": {"weight": 42},
        }
        runner = FakeRunner(with_app=True)
        cfg = types.SimpleNamespace(ckpt=["ckpt.pt"])
        p1, p2 = self._patched(ckpt)
        with p1, p2:
            te._render_traj_from_checkpoint(runner, cfg)
        self.assertEqual(runner.app_loaded, {"weight": 42})
        self.assertEqual(runner.rendered_at, 119999)

    def test_app_opt_without_saved_state_is_an_error(self) -> None:
        ckpt = {"step": 5, "splats": {"means": "m", "opacities": "o"}}
        runner = FakeRunner(with_app=True)
        cfg = types.SimpleNamespace(ckpt=["ckpt.pt"])
        p1, p2 = self._patched(ckpt)
        with p1, p2:
            with self.assertRaisesRegex(RuntimeError, "app_module"):
                te._render_traj_from_checkpoint(runner, cfg)
        self.assertIsNone(runner.rendered_at)

    def test_no_app_module_runner_renders_without_state(self) -> None:
        ckpt = {"step": 7, "splats": {"means": "m", "opacities": "o"}}
        runner = FakeRunner(with_app=False)
        cfg = types.SimpleNamespace(ckpt=["ckpt.pt"])
        p1, p2 = self._patched(ckpt)
        with p1, p2:
            te._render_traj_from_checkpoint(runner, cfg)
        self.assertEqual(runner.rendered_at, 7)


class ViewOnlyTests(unittest.TestCase):
    def test_experiment_config_defaults_to_render_mode(self) -> None:
        cfg = te.ExperimentConfig()
        self.assertFalse(cfg.view_only)
        self.assertEqual(cfg.port, 8080)
        self.assertIsNone(cfg.ckpt)

    def test_serve_viewer_restores_checkpoint_and_waits(self) -> None:
        ckpt = {"step": 119999, "splats": {"means": "m", "opacities": "o"}, "app_module": {"w": 1}}
        runner = FakeRunner(with_app=True)
        completed = []
        runner.viewer = types.SimpleNamespace(complete=lambda: completed.append(True))
        cfg = types.SimpleNamespace(ckpt=["ckpt.pt"], disable_viewer=False, port=8090)
        sleeps: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            raise KeyboardInterrupt  # end the wait loop in the test

        fake_torch = types.SimpleNamespace(load=lambda *a, **k: ckpt, cat=lambda tensors: ("cat", tuple(tensors)))
        fake_scene = types.SimpleNamespace(splats={"means": FakeParam(1)})
        fake_st = types.SimpleNamespace(
            GaussianScene=types.SimpleNamespace(from_splats=lambda splats, id: fake_scene),
            Stage=lambda: types.SimpleNamespace(add_scene=lambda *a, **k: None),
        )
        with patch.object(te, "torch", fake_torch), patch.object(te, "st", fake_st):
            with self.assertRaises(KeyboardInterrupt):
                te._serve_viewer_from_checkpoint(runner, cfg, sleep=fake_sleep)

        self.assertEqual(completed, [True])
        self.assertEqual(runner.app_loaded, {"w": 1})
        self.assertIsNone(runner.rendered_at)
        self.assertEqual(sleeps, [3600])

    def test_serve_viewer_refuses_when_viewer_disabled(self) -> None:
        runner = FakeRunner(with_app=False)
        cfg = types.SimpleNamespace(ckpt=["ckpt.pt"], disable_viewer=True, port=8080)
        with self.assertRaisesRegex(RuntimeError, "view-only"):
            te._serve_viewer_from_checkpoint(runner, cfg, sleep=lambda s: None)


if __name__ == "__main__":
    unittest.main()
