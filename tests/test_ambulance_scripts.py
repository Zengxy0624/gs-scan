from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLMAP_SCRIPT = PROJECT_ROOT / "scripts" / "run_ambulance_colmap.sh"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_ambulance_gs.sh"


class AmbulanceScriptTests(unittest.TestCase):
    def run_script(
        self,
        script: Path,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), *arguments],
            cwd=PROJECT_ROOT,
            env={**os.environ, **(environment or {})},
            check=False,
            text=True,
            capture_output=True,
        )

    def test_colmap_dry_run_prints_quality_recipe_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_root = root / "frames"
            dataset = root / "merged"
            result = self.run_script(
                COLMAP_SCRIPT,
                "--dry-run",
                environment={
                    "CAPTURE_ROOT": str(root / "captures"),
                    "FRAME_ROOT": str(frame_root),
                    "DATASET_ROOT": str(dataset),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [
                line for line in result.stdout.splitlines() if line.startswith("+ ")
            ]
            self.assertEqual(len(commands), 2)
            self.assertIn("merge_frame_datasets.py", commands[0])
            self.assertIn("--link-mode hardlink", commands[0])
            self.assertEqual(commands[0].count("/frames/A001_"), 18)
            self.assertIn("prepare_colmap.py", commands[1])
            for expected in (
                "LD_PRELOAD=",
                "third-party/openblas-threaded/libopenblas-pthread.so.0",
                "OMP_NUM_THREADS=64",
                "OPENBLAS_NUM_THREADS=1",
                "--matching vocab_tree",
                "--vocab-tree-num-images 200",
                "--guided-matching",
                "--sift-affine-shape",
                "--sift-domain-size-pooling",
                "--feature-max-image-size 3840",
                "--feature-max-num-features 8192",
                "--init-min-tri-angle 8",
                "--camera-model OPENCV",
                "--min-registration-ratio 0.8",
                "--min-source-registration-ratio 0.8",
            ):
                self.assertIn(expected, commands[1])
            self.assertFalse(frame_root.exists())
            self.assertFalse(dataset.exists())

    def test_colmap_dry_run_accepts_local_vocab_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = Path(temporary) / "tree.bin"
            result = self.run_script(
                COLMAP_SCRIPT,
                "--dry-run",
                "--vocab-tree",
                str(tree),
                "--vocab-tree-num-images",
                "250",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--vocab-tree-num-images 250", result.stdout)
            self.assertIn(f"--vocab-tree {tree}", result.stdout)

    def test_training_dry_run_matches_old_full_quality_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            dataset = root / "dataset"
            result = self.run_script(
                TRAIN_SCRIPT,
                "--profile",
                "pilot",
                "--gpu",
                "7",
                "--dry-run",
                environment={
                    "CAPTURE_ROOT": str(root / "captures"),
                    "DATASET_ROOT": str(dataset),
                    "OUTPUT_ROOT": str(output),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [
                line for line in result.stdout.splitlines() if line.startswith("+ ")
            ]
            self.assertEqual(len(commands), 4)
            self.assertIn("build_run_manifest.py build", commands[0])
            self.assertEqual(commands[0].count("--source "), 18)
            trainer = commands[-1]
            for expected in (
                "CUDA_VISIBLE_DEVICES=7",
                "simple_trainer.py default",
                "--data-factor 1",
                "--max-steps 7000",
                "--eval-steps 7000",
                "--save-steps 7000",
                "--ply-steps 7000",
                "--save-ply",
                "--disable-viewer",
            ):
                self.assertIn(expected, trainer)
            self.assertFalse(output.exists())
            self.assertFalse(dataset.exists())

    def test_reuse_matches_and_mapper_tuning_forward_to_preparer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script(
                COLMAP_SCRIPT,
                "--dry-run",
                "--reuse-matches",
                "--ba-global-frames-ratio",
                "1.3",
                "--ba-global-points-ratio",
                "1.3",
                "--ba-global-max-refinements",
                "2",
                "--ba-global-max-num-iterations",
                "40",
                "--snapshot-frames-freq",
                "250",
                environment={
                    "CAPTURE_ROOT": str(root / "captures"),
                    "FRAME_ROOT": str(root / "frames"),
                    "DATASET_ROOT": str(root / "merged"),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            colmap = [
                line
                for line in result.stdout.splitlines()
                if "prepare_colmap.py" in line
            ][0]
            for expected in (
                "--reuse-matches",
                "--ba-global-frames-ratio 1.3",
                "--ba-global-points-ratio 1.3",
                "--ba-global-max-refinements 2",
                "--ba-global-max-num-iterations 40",
                "--snapshot-frames-freq 250",
            ):
                self.assertIn(expected, colmap)

    def test_gpu_ba_switches_interpreter_and_forwards_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_python = root / "cuda-python"
            fake_python.write_text("#!/bin/sh\n")
            fake_python.chmod(0o755)
            result = self.run_script(
                COLMAP_SCRIPT,
                "--dry-run",
                "--ba-use-gpu",
                "--ba-gpu-index",
                "6",
                environment={
                    "CAPTURE_ROOT": str(root / "captures"),
                    "FRAME_ROOT": str(root / "frames"),
                    "DATASET_ROOT": str(root / "merged"),
                    "COLMAP_PYTHON": str(fake_python),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            colmap = [
                line
                for line in result.stdout.splitlines()
                if "prepare_colmap.py" in line
            ][0]
            self.assertIn(str(fake_python), colmap)
            self.assertIn("--ba-use-gpu", colmap)
            self.assertIn("--ba-gpu-index 6", colmap)
            self.assertNotIn("LD_PRELOAD", colmap)
            self.assertNotIn("uv run", colmap)

    def test_rejects_invalid_options(self) -> None:
        cases = (
            (COLMAP_SCRIPT, ("--link-mode", "symlink"), "--link-mode"),
            (COLMAP_SCRIPT, ("--vocab-tree-num-images", "0"), "positive"),
            (COLMAP_SCRIPT, ("--ba-global-max-refinements", "0"), "positive"),
            (COLMAP_SCRIPT, ("--snapshot-frames-freq", "-5"), "positive"),
            (TRAIN_SCRIPT, ("--profile", "medium"), "--profile"),
            (TRAIN_SCRIPT, ("--gpu", "0,1"), "--gpu"),
        )
        for script, arguments, message in cases:
            with self.subTest(script=script.name, arguments=arguments):
                result = self.run_script(script, *arguments)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
