from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_single_video_gs.sh"
STEM = "A001_08071524_C009"


class SingleVideoScriptTests(unittest.TestCase):
    def run_script(
        self,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            cwd=PROJECT_ROOT,
            env={**os.environ, **(environment or {})},
            check=False,
            text=True,
            capture_output=True,
        )

    def isolated_environment(self, root: Path) -> dict[str, str]:
        return {
            "CAPTURE_ROOT": str(root / "captures"),
            "WORK_ROOT": str(root / "work"),
            "OUTPUT_ROOT": str(root / "outputs"),
        }

    def commands(self, stdout: str) -> list[str]:
        return [line for line in stdout.splitlines() if line.startswith("+ ")]

    def test_dry_run_prints_full_pipeline_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script(
                "--video",
                STEM,
                "--profile",
                "pilot",
                "--gpu",
                "6",
                "--dry-run",
                environment=self.isolated_environment(root),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = self.commands(result.stdout)
            self.assertEqual(len(commands), 6)
            self.assertIn("extract_frames.py", commands[0])
            self.assertIn("--sample-fps 3", commands[0])
            self.assertIn("prepare_colmap.py", commands[1])
            for expected in (
                "OMP_NUM_THREADS=64",
                "--matching sequential",
                "--overlap 15",
                "--camera-model OPENCV",
                "--min-registration-ratio 0.8",
            ):
                self.assertIn(expected, commands[1])
            self.assertNotIn("LD_PRELOAD", commands[1])
            self.assertNotIn("--min-source-registration-ratio", commands[1])
            self.assertIn("build_run_manifest.py build", commands[2])
            self.assertIn(f"--scene {STEM}", commands[2])
            self.assertEqual(commands[2].count("--source "), 1)
            trainer = commands[-1]
            for expected in (
                "CUDA_VISIBLE_DEVICES=6",
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
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "outputs").exists())

    def test_preprocess_only_stops_after_colmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script(
                "--video",
                STEM,
                "--preprocess-only",
                "--dry-run",
                environment=self.isolated_environment(root),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = self.commands(result.stdout)
            self.assertEqual(len(commands), 2)
            self.assertIn("extract_frames.py", commands[0])
            self.assertIn("prepare_colmap.py", commands[1])
            self.assertNotIn("simple_trainer.py", result.stdout)

    def test_vocab_tree_matching_forwards_its_own_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = root / "tree.bin"
            result = self.run_script(
                "--video",
                STEM,
                "--matching",
                "vocab_tree",
                "--vocab-tree-num-images",
                "200",
                "--vocab-tree",
                str(tree),
                "--preprocess-only",
                "--dry-run",
                environment=self.isolated_environment(root),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            colmap = self.commands(result.stdout)[1]
            self.assertIn("--matching vocab_tree", colmap)
            self.assertIn("--vocab-tree-num-images 200", colmap)
            self.assertIn(f"--vocab-tree {tree}", colmap)
            self.assertIn("LD_PRELOAD=", colmap)
            self.assertIn(
                "third-party/openblas-threaded/libopenblas-pthread.so.0",
                colmap,
            )
            self.assertIn("OPENBLAS_NUM_THREADS=1", colmap)

    def test_sequential_run_key_and_paths_include_the_video_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script(
                "--video",
                STEM,
                "--dry-run",
                environment=self.isolated_environment(root),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"/work/{STEM}", result.stdout)
            self.assertIn(
                f"/outputs/{STEM}/sequential_single_opencv_30000steps",
                result.stdout,
            )

    def test_rejects_invalid_options(self) -> None:
        cases = (
            ((), "--video is required"),
            (("--video", "../escape"), "--video"),
            (("--video", STEM, "--profile", "medium"), "--profile"),
            (("--video", STEM, "--matching", "tree"), "--matching"),
            (("--video", STEM, "--gpu", "0,1"), "--gpu"),
            (("--video", STEM, "--overlap", "0"), "positive"),
            (("--video", STEM, "--camera-model", "FISHEYE"), "--camera-model"),
            (("--video", STEM, "--vocab-tree", "tree.bin"), "--vocab-tree"),
            (
                ("--video", STEM, "--min-registration-ratio", "1.5"),
                "--min-registration-ratio",
            ),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = self.run_script(*arguments)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
