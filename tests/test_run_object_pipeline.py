from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_object_pipeline.sh"


class ObjectPipelineDriverTests(unittest.TestCase):
    def run_script(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments], cwd=PROJECT_ROOT, env=dict(os.environ),
            check=False, text=True, capture_output=True,
        )

    def commands(self, result: subprocess.CompletedProcess[str]) -> list[str]:
        return [line for line in result.stdout.splitlines() if line.startswith("+ ")]

    def test_dry_run_chains_three_stages_with_production_training(self) -> None:
        result = self.run_script("--dry-run", "--project", "jump_center", "--name", "mannequin_back", "--gpu", "2", "A", "B")
        self.assertEqual(result.returncode, 0, result.stderr)
        cmds = self.commands(result)
        self.assertEqual(len(cmds), 3, "\n".join(cmds))
        prepare, colmap, train = cmds
        self.assertIn("prepare_object_frames.sh --project jump_center --name mannequin_back A B", prepare)
        self.assertIn(f"run_object_colmap.sh --dataset {PROJECT_ROOT}/work/jump_center/mannequin_back --gpu 2", colmap)
        for flag in ("--camera-model OPENCV", "--vocab-tree-num-images 200", "--feature-max-num-features 8192",
                     "--ba-global-frames-ratio 1.3", "--ba-global-max-refinements 2", "--ba-global-max-num-iterations 30", "--colmap-threads 64"):
            self.assertIn(flag, colmap, flag)
        for flag in ("CUDA_VISIBLE_DEVICES=2", "train_experiment.py mcmc", f"--data-dir {PROJECT_ROOT}/work/jump_center/mannequin_back",
                     f"--result-dir {PROJECT_ROOT}/outputs/jump_center/mannequin_back_exp1", "--strategy.cap-max 6500000",
                     "--steps-scaler 4", "--app-opt", "--strategy.refine-stop-iter 29000", "--opacity-reg 0.0005",
                     "--max-steps 30000", "--eval-steps 30000", "--save-ply", "--disable-viewer"):
            self.assertIn(flag, train, flag)
        self.assertIn("final step 119999", result.stdout)

    def test_stage_selection_presets_and_short_training(self) -> None:
        result = self.run_script(
            "--dry-run", "--project", "p", "--name", "o", "--stages", "colmap,train", "--ba-preset", "production",
            "--train-steps", "12000", "--tag", "quick", "--camera-model", "OPENCV_FISHEYE", "--vocab-tree-num-images", "100",
            "--colmap-arg", "--resume-from-snapshot", "--cap-max", "3000000",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        cmds = self.commands(result)
        self.assertEqual(len(cmds), 2)
        colmap, train = cmds
        self.assertNotIn("prepare_object_frames", "\n".join(cmds))
        self.assertNotIn("--ba-global-frames-ratio", colmap)      # production = wrapper defaults
        self.assertIn("--camera-model OPENCV_FISHEYE", colmap)
        self.assertIn("--vocab-tree-num-images 100", colmap)
        self.assertIn("--resume-from-snapshot", colmap)
        self.assertIn("--steps-scaler 1", train); self.assertIn("--max-steps 12000", train)
        self.assertIn("--strategy.refine-stop-iter 11600", train); self.assertIn("--strategy.cap-max 3000000", train)
        self.assertIn("/outputs/p/o_quick", train)
        self.assertIn("final step 11999", result.stdout)

    def test_prepare_args_pass_through(self) -> None:
        result = self.run_script("--dry-run", "--project", "p", "--name", "o", "--stages", "frames",
                                 "--prepare-arg", "--drop-ratio", "--prepare-arg", "0.6", "A", "B")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--drop-ratio 0.6 A B", self.commands(result)[0])

    def test_rejects_bad_arguments(self) -> None:
        for arguments in (
            ["--dry-run", "--name", "o", "A", "B"],                         # missing project
            ["--dry-run", "--project", "p", "A", "B"],                       # missing name
            ["--dry-run", "--project", "p", "--name", "o", "A"],             # one clip for frames stage
            ["--dry-run", "--project", "p", "--name", "o", "--stages", "colmap,export"],
            ["--dry-run", "--project", "p", "--name", "o", "--ba-preset", "slow", "A", "B"],
            ["--dry-run", "--project", "p", "--name", "o", "--train-steps", "0", "A", "B"],
            ["--dry-run", "--project", "p", "--name", "o", "--gpu", "two", "A", "B"],
            ["--dry-run", "--project", "../p", "--name", "o", "A", "B"],
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.run_script(*arguments).returncode, 2, arguments)


if __name__ == "__main__":
    unittest.main()
