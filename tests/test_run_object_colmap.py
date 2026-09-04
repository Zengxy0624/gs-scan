from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_object_colmap.sh"


def write_dataset(root: Path) -> Path:
    dataset = root / "mannequin_front"
    (dataset / "images").mkdir(parents=True)
    videos = []
    for index in range(2):
        video = root / "captures" / f"JC_0{index + 1}_0000.mov"
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(b"video")
        videos.append(str(video.resolve()))
    metadata = {
        "dataset_type": "video_set",
        "dataset_instance_id": str(uuid.uuid4()),
        "sources": [{"source_video": video} for video in videos],
        "sample_fps": 3.0,
        "max_frames": 100000,
        "sharpness_window_seconds": 0.3,
        "sharpness_keep_ratio": 0.8,
        "sharpness_drop_ratio": 0.7,
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return dataset


class ObjectColmapScriptTests(unittest.TestCase):
    def run_script(self, *arguments: str, environment: dict[str, str] | None = None):
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            cwd=PROJECT_ROOT,
            env={**os.environ, **(environment or {})},
            check=False,
            text=True,
            capture_output=True,
        )

    def commands(self, result: subprocess.CompletedProcess[str]) -> list[str]:
        return [line for line in result.stdout.splitlines() if line.startswith("+ ")]

    def test_dry_run_with_cuda_wheel_uses_shim_gpu_features_and_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_dataset(root)
            cuda_python = root / "cuda-venv" / "bin" / "python"
            cuda_python.parent.mkdir(parents=True)
            cuda_python.write_text("#!/bin/sh\n"); cuda_python.chmod(0o755)
            vocab = root / "tree.bin"; vocab.write_bytes(b"tree")

            result = self.run_script(
                "--dry-run", "--dataset", str(dataset), "--gpu", "2", "--vocab-tree", str(vocab),
                environment={
                    "GPU_COLMAP_PYTHON": str(root / "missing-cudss-python"),
                    "CUDA_COLMAP_PYTHON": str(cuda_python),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[cuda-wheel]", result.stdout)
            self.assertIn("features on, bundle adjustment off", result.stdout)
            commands = self.commands(result)
            self.assertEqual(len(commands), 2)
            colmap = commands[0]
            for expected in (
                "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=2", "OMP_NUM_THREADS=32",
                "LD_PRELOAD=", "libopenblas-pthread.so.0", "OPENBLAS_NUM_THREADS=1",
                str(cuda_python), "prepare_colmap.py", str(dataset.resolve()),
                "--matching vocab_tree", "--vocab-tree-num-images 200", "--guided-matching",
                "--sift-affine-shape", "--sift-domain-size-pooling",
                "--feature-max-image-size 3840", "--feature-max-num-features 8192",
                "--init-min-tri-angle 8", "--camera-model OPENCV", "--num-threads 32",
                "--ba-global-frames-ratio 1.1", "--ba-global-points-ratio 1.1",
                "--ba-global-max-refinements 5", "--ba-global-max-num-iterations 50",
                "--min-registration-ratio 0.8", "--min-source-registration-ratio 0.8",
                f"--vocab-tree {vocab.resolve()}", "--feature-gpu --feature-gpu-index 0",
                "--snapshot-frames-freq 250",
            ):
                self.assertIn(expected, colmap, expected)
            self.assertNotIn("--ba-use-gpu", colmap)
            self.assertNotIn("--reuse-matches", colmap)
            validate = commands[1]
            for expected in (
                "validate_dataset_cache.py", "JC_01_0000.mov", "JC_02_0000.mov",
                "--sample-fps 3.0", "--max-frames 100000", "--sharpness-window 0.3",
                "--sharpness-keep-ratio 0.8", "--sharpness-drop-ratio 0.7",
                "--camera-model OPENCV", "--matching-strategy vocab_tree", "--require-colmap",
            ):
                self.assertIn(expected, validate, expected)
            self.assertFalse((dataset / "database.db").exists())
            self.assertFalse((root / ".mannequin_front-colmap-lock").exists())

    def test_cudss_interpreter_enables_gpu_ba_and_drops_shim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_dataset(root)
            cudss_python = root / "gpubuild" / "bin" / "python"
            cudss_python.parent.mkdir(parents=True)
            cudss_python.write_text("#!/bin/sh\n"); cudss_python.chmod(0o755)
            marker = root / "BUILD_DONE"

            # interpreter present but build not finished: not selected
            result = self.run_script(
                "--dry-run", "--dataset", str(dataset),
                environment={
                    "GPU_COLMAP_PYTHON": str(cudss_python),
                    "GPU_COLMAP_READY_MARKER": str(marker),
                    "CUDA_COLMAP_PYTHON": str(root / "no-cuda"),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[cpu-wheel]", result.stdout)

            marker.write_text("done\n")
            result = self.run_script(
                "--dry-run", "--dataset", str(dataset), "--camera-model", "OPENCV_FISHEYE",
                "--ba-global-frames-ratio", "1.5", "--ba-global-max-refinements", "2",
                "--mapping-blas-threads", "32", "--snapshot-frames-freq", "0",
                environment={"GPU_COLMAP_PYTHON": str(cudss_python), "GPU_COLMAP_READY_MARKER": str(marker)},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[cudss]", result.stdout)
            colmap = self.commands(result)[0]
            self.assertIn("--ba-use-gpu --ba-gpu-index 0", colmap)
            self.assertIn("--feature-gpu --feature-gpu-index 0", colmap)
            self.assertNotIn("LD_PRELOAD", colmap)
            self.assertNotIn("OPENBLAS_NUM_THREADS", colmap)
            self.assertIn("--camera-model OPENCV_FISHEYE", colmap)
            self.assertIn("--ba-global-frames-ratio 1.5", colmap)
            self.assertIn("--ba-global-max-refinements 2", colmap)
            self.assertIn("--mapping-blas-threads 32", colmap)
            self.assertNotIn("--snapshot-frames-freq", colmap)
            # no shim for the cuDSS build: the thread count is not capped
            result = self.run_script(
                "--dry-run", "--dataset", str(dataset), "--colmap-threads", "-1",
                environment={"GPU_COLMAP_PYTHON": str(cudss_python), "GPU_COLMAP_READY_MARKER": str(marker)},
            )
            self.assertIn("--num-threads -1", self.commands(result)[0])
            self.assertIn("OMP_NUM_THREADS=64", self.commands(result)[0])

    def test_cpu_fallback_and_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_dataset(root)
            missing = {
                "GPU_COLMAP_PYTHON": str(root / "no-cudss"),
                "CUDA_COLMAP_PYTHON": str(root / "no-cuda"),
            }
            result = self.run_script("--dry-run", "--dataset", str(dataset), environment=missing)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[cpu-wheel]", result.stdout)
            colmap = self.commands(result)[0]
            self.assertNotIn("--feature-gpu", colmap)
            self.assertIn("LD_PRELOAD=", colmap)
            # shim on: thread requests above the safe limit are capped
            result = self.run_script(
                "--dry-run", "--dataset", str(dataset), "--colmap-threads", "128", environment=missing,
            )
            self.assertIn("capped at 32", result.stdout)
            self.assertIn("--num-threads 32", self.commands(result)[0])
            self.assertIn("OMP_NUM_THREADS=32", self.commands(result)[0])

            result = self.run_script(
                "--dry-run", "--dataset", str(dataset), "--feature-gpu", "on", "--ba-use-gpu", "on",
                "--blas-shim", "off", "--resume-from-snapshot", environment=missing,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            colmap = self.commands(result)[0]
            self.assertIn("--feature-gpu", colmap)
            self.assertIn("--ba-use-gpu", colmap)
            self.assertNotIn("LD_PRELOAD", colmap)
            self.assertIn("--reuse-matches", colmap)
            self.assertIn("--resume-model", colmap)

    def test_rejects_invalid_options_and_missing_dataset(self) -> None:
        for arguments in (
            ["--gpu", "x"], ["--camera-model", "FISHEYE"], ["--feature-gpu", "yes"],
            ["--min-registration-ratio", "1.5"], ["--vocab-tree-num-images", "0"],
            ["--snapshot-frames-freq", "-1"], ["--unknown"],
        ):
            with self.subTest(arguments=arguments):
                result = self.run_script("--dry-run", "--dataset", "x", *arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--dataset is required", result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_script("--dataset", str(Path(temporary) / "nope"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("metadata not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
