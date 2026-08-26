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


if __name__ == "__main__":
    unittest.main()
