from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_device_test.sh"
CAPTURE_NAMES = (
    "insta_normal.mp4",
    "iphone_18mm.mov",
    "iphone_24mm.mov",
    "osmo_rotate.MP4",
    "osmo_stable.MP4",
)


class DeviceBatchTests(unittest.TestCase):
    def prepare_captures(self, root: Path, missing: str | None = None) -> Path:
        captures = root / "captures"
        captures.mkdir(exist_ok=True)
        for name in CAPTURE_NAMES:
            capture = captures / name
            if name != missing and not capture.exists():
                capture.write_bytes(b"placeholder")
        return captures

    def run_script(
        self,
        root: Path,
        *arguments: str,
        missing: str | None = None,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        captures = self.prepare_captures(root, missing)

        environment = {
            **os.environ,
            "CAPTURE_ROOT": str(captures),
            "WORK_ROOT": str(root / "work"),
            "OUTPUT_ROOT": str(root / "outputs"),
            "PROJECT_PYTHON": sys.executable,
            **(extra_environment or {}),
        }
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )

    def make_fake_uv(self, root: Path) -> tuple[Path, Path]:
        fake_bin = root / "bin"
        fake_bin.mkdir()
        log = root / "uv-calls.log"
        uv = fake_bin / "uv"
        uv.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

def option(arguments, name):
    return arguments[arguments.index(name) + 1]

def identity(path):
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def images(directory, count=2):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:06d}.jpg").write_bytes(b"jpeg")

def sparse(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (directory / name).write_bytes(name.encode("ascii"))

arguments = sys.argv[1:]
script_index = next(
    index for index, value in enumerate(arguments)
    if value.endswith(".py")
)
script = Path(arguments[script_index]).name
stage = arguments[script_index + 1:]
with Path(os.environ["STUB_UV_LOG"]).open("a", encoding="utf-8") as output:
    output.write(script + "\\n")

if script == "extract_frames.py":
    source = Path(stage[0]).resolve()
    dataset = Path(stage[1])
    dataset.mkdir(parents=True)
    images(dataset / "images")
    source_id = identity(source)
    metadata = {
        "dataset_type": "single_video",
        "source_video": source_id["path"],
        "source_size_bytes": source_id["size_bytes"],
        "source_mtime_ns": source_id["mtime_ns"],
        "sample_fps": float(option(stage, "--sample-fps")),
        "max_frames": int(option(stage, "--max-frames")),
        "selected_indices": [0, 1],
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata))
elif script == "prepare_colmap.py":
    dataset = Path(stage[0])
    (dataset / "database.db").write_bytes(b"database")
    sparse(dataset / "sparse" / "0")
    (dataset / "reconstruction_stats.json").write_text(
        '{"status": "passed"}'
    )
else:
    raise SystemExit(f"unexpected stubbed script: {script}")
""",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        return fake_bin, log

    def test_preprocess_only_lists_every_input_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_script(
                Path(temporary),
                "--dry-run",
                "--preprocess-only",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        for name in CAPTURE_NAMES:
            self.assertIn(name, result.stdout)
        self.assertEqual(result.stdout.count("=== "), 5)
        self.assertNotIn("simple_trainer.py", result.stdout)

    def test_rejects_unknown_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_script(
                Path(temporary),
                "--dry-run",
                "--scene",
                "not-a-scene",
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown --scene value", result.stderr)

    def test_rejects_invalid_numeric_options(self) -> None:
        cases = (
            (("--sample-fps", "0.0"), "--sample-fps must be positive"),
            (
                ("--min-registration-ratio", "1.1"),
                "--min-registration-ratio must be greater than 0",
            ),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                with tempfile.TemporaryDirectory() as temporary:
                    result = self.run_script(
                        Path(temporary),
                        "--dry-run",
                        *arguments,
                    )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)

    def test_output_namespace_includes_preprocessing_parameters_not_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_script(
                Path(temporary),
                "--dry-run",
                "--scene",
                "iphone_18mm",
                "--sample-fps",
                "5",
                "--max-frames",
                "600",
                "--overlap",
                "20",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "iphone_18mm/"
            "fps5_max600_overlap20_reg0.5_30000steps",
            result.stdout,
        )
        self.assertNotIn("iphone_18mm/full", result.stdout)
        self.assertNotIn("--disable-video", result.stdout)

    def test_missing_standard_video_fails_that_scene_but_continues_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_script(
                Path(temporary),
                "--dry-run",
                "--preprocess-only",
                missing="iphone_24mm.mov",
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("capture not found", result.stderr)
        self.assertIn("=== osmo_stable ===", result.stdout)
        self.assertIn("Failed:  iphone_24mm", result.stdout)
        self.assertIn("Passed:  insta_normal", result.stdout)

    def test_script_does_not_install_or_change_cluster_environment(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("bash scripts/setup_gsplat.sh", script)
        self.assertNotIn("module load", script)
        self.assertNotIn("uv sync", script)
        self.assertIn("--no-sync", script)

    def test_real_mode_preprocess_resumes_without_repeating_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin, log = self.make_fake_uv(root)
            environment = {
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "STUB_UV_LOG": str(log),
            }
            first = self.run_script(
                root,
                "--preprocess-only",
                "--scene",
                "iphone_18mm",
                extra_environment=environment,
            )
            second = self.run_script(
                root,
                "--preprocess-only",
                "--scene",
                "iphone_18mm",
                extra_environment=environment,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                log.read_text(encoding="utf-8").splitlines(),
                ["extract_frames.py", "prepare_colmap.py"],
            )
            self.assertIn("Skipping completed extraction", second.stdout)
            self.assertIn("Skipping completed COLMAP", second.stdout)
            self.assertFalse(
                (
                    root
                    / "work"
                    / "fps6_max100000_overlap15_reg0.5"
                    / ".batch-lock"
                ).exists()
            )

    def test_retry_archives_partial_colmap_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = self.prepare_captures(root)
            fake_bin, log = self.make_fake_uv(root)
            key = "fps6_max100000_overlap15_reg0.5"
            scene_work = root / "work" / key / "iphone_18mm"
            dataset = scene_work / "dataset"
            images = dataset / "images"
            images.mkdir(parents=True)
            for index in range(2):
                (images / f"{index:06d}.jpg").write_bytes(b"jpeg")
            source = (captures / "iphone_18mm.mov").resolve()
            stat = source.stat()
            (dataset / "metadata.json").write_text(
                (
                    "{"
                    '"dataset_type":"single_video",'
                    f'"source_video":"{source}",'
                    f'"source_size_bytes":{stat.st_size},'
                    f'"source_mtime_ns":{stat.st_mtime_ns},'
                    '"sample_fps":6.0,'
                    '"max_frames":100000,'
                    '"selected_indices":[0,1]'
                    "}"
                ),
                encoding="utf-8",
            )
            (dataset / "database.db").write_bytes(b"partial database")
            (scene_work / "old-state-marker").write_text(
                "partial",
                encoding="utf-8",
            )

            result = self.run_script(
                root,
                "--preprocess-only",
                "--scene",
                "iphone_18mm",
                "--retry-incomplete",
                extra_environment={
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                    "STUB_UV_LOG": str(log),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                log.read_text(encoding="utf-8").splitlines(),
                ["extract_frames.py", "prepare_colmap.py"],
            )
            archived_markers = list(
                (root / "work" / key / "_failed").glob(
                    "iphone_18mm-*/old-state-marker"
                )
            )
            self.assertEqual(len(archived_markers), 1)
            self.assertTrue((dataset / "metadata.json").is_file())
            self.assertTrue((dataset / "sparse" / "0").is_dir())


if __name__ == "__main__":
    unittest.main()
