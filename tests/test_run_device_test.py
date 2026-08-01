from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_device_test.sh"
TRAINER = PROJECT_ROOT / "third-party" / "gsplat" / "examples" / "simple_trainer.py"
CAPTURE_NAMES = (
    "insta_normal.mp4",
    "iphone_18mm.mov",
    "iphone_24mm.mov",
    "osmo_rotate.MP4",
    "osmo_stable.MP4",
)
DEFAULT_PREPROCESS_KEY = "fps6_max100000_camopencv_overlap15_reg0.5"


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
import hashlib
import os
import sys
import uuid
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
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

def images(directory, count=2):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:06d}.jpg").write_bytes(b"jpeg")

def tree_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\\0")
    return digest.hexdigest()

def sparse(directory):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        content = name.encode("ascii")
        (directory / name).write_bytes(content)
        result[name] = {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return result

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
        "dataset_instance_id": str(uuid.uuid4()),
        "source_video": source_id["path"],
        "source_size_bytes": source_id["size_bytes"],
        "source_mtime_ns": source_id["mtime_ns"],
        "source_sha256": source_id["sha256"],
        "images_sha256": tree_digest(dataset / "images"),
        "sample_fps": float(option(stage, "--sample-fps")),
        "max_frames": int(option(stage, "--max-frames")),
        "selected_indices": [0, 1],
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata))
elif script == "prepare_colmap.py":
    dataset = Path(stage[0])
    (dataset / "database.db").write_bytes(b"database")
    model_files = sparse(dataset / "sparse" / "0")
    (dataset / "reconstruction_stats.json").write_text(
        json.dumps({
            "camera_model": option(stage, "--camera-model"),
            "matching_overlap": int(option(stage, "--overlap")),
            "matching_quadratic_overlap": True,
            "minimum_registration_ratio": float(
                option(stage, "--min-registration-ratio")
            ),
            "num_input_images": 2,
            "selected_model_id": 0,
            "selected_model_files": model_files,
            "selected_model_path": "sparse/0",
            "status": "passed",
        })
    )
elif script == "simple_trainer.py":
    result_dir = Path(option(stage, "--result-dir"))
    final_step = int(option(stage, "--max-steps")) - 1
    artifacts = {
        result_dir / "ckpts" / f"ckpt_{final_step}_rank0.pt": b"checkpoint",
        result_dir / "ply" / f"point_cloud_{final_step}.ply": b"ply",
        result_dir / "stats" / f"val_step{final_step}.json": b"{}",
    }
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
else:
    raise SystemExit(f"unexpected stubbed script: {script}")
""",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        nvidia_smi = fake_bin / "nvidia-smi"
        nvidia_smi.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
        nvidia_smi.chmod(0o755)
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

    def test_output_namespace_includes_reconstruction_parameters_not_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script(
                root,
                "--dry-run",
                "--scene",
                "iphone_18mm",
                "--sample-fps",
                "05.00",
                "--max-frames",
                "0600",
                "--overlap",
                "020",
                "--min-registration-ratio",
                "0.50",
                "--gpu",
                "06",
                "--colmap-threads",
                "08",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "iphone_18mm/"
            "fps5_max600_camopencv_overlap20_reg0.5_30000steps",
            result.stdout,
        )
        self.assertNotIn("iphone_18mm/full", result.stdout)
        self.assertNotIn("--disable-video", result.stdout)
        commands = [
            shlex.split(line.removeprefix("+ "))
            for line in result.stdout.splitlines()
            if line.startswith("+ ")
        ]
        trainer_command = next(
            command
            for command in commands
            if str(TRAINER) in command and "default" in command
        )
        key = "fps5_max600_camopencv_overlap20_reg0.5"
        self.assertEqual(
            trainer_command,
            [
                "env",
                "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES=6",
                "uv",
                "run",
                "--project",
                str(PROJECT_ROOT / "third-party" / "gsplat"),
                "--no-sync",
                "python",
                str(TRAINER),
                "default",
                "--data-dir",
                str(root / "work" / key / "iphone_18mm" / "dataset"),
                "--data-factor",
                "1",
                "--result-dir",
                str(root / "outputs" / "iphone_18mm" / f"{key}_30000steps"),
                "--max-steps",
                "30000",
                "--eval-steps",
                "30000",
                "--save-steps",
                "30000",
                "--ply-steps",
                "30000",
                "--save-ply",
                "--disable-viewer",
            ],
        )
        manifest_builder = PROJECT_ROOT / "scripts" / "build_run_manifest.py"
        manifest_command = next(
            command for command in commands if str(manifest_builder) in command
        )
        self.assertEqual(
            manifest_command,
            [
                sys.executable,
                str(manifest_builder),
                "build",
                "--scene",
                "iphone_18mm",
                "--dataset",
                str(root / "work" / key / "iphone_18mm" / "dataset"),
                "--source",
                str(root / "captures" / "iphone_18mm.mov"),
                "--gsplat-root",
                str(PROJECT_ROOT / "third-party" / "gsplat"),
                "--sample-fps",
                "5",
                "--max-frames",
                "600",
                "--overlap",
                "20",
                "--min-registration-ratio",
                "0.5",
                "--max-steps",
                "30000",
                "--camera-model",
                "OPENCV",
                "--colmap-threads",
                "8",
                "--gpu-id",
                "6",
                "--output",
                str(
                    root
                    / "work"
                    / key
                    / "iphone_18mm"
                    / "training_manifest_30000.json"
                ),
            ],
        )

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

    def test_pilot_profile_propagates_steps_and_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_script(
                Path(temporary),
                "--dry-run",
                "--scene",
                "iphone_18mm",
                "--profile",
                "pilot",
                "--gpu",
                "7",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        trainer_command = next(
            shlex.split(line.removeprefix("+ "))
            for line in result.stdout.splitlines()
            if line.startswith("+ ") and "simple_trainer.py" in line
        )
        self.assertIn("CUDA_VISIBLE_DEVICES=7", trainer_command)
        for option in ("--max-steps", "--eval-steps", "--save-steps", "--ply-steps"):
            self.assertEqual(trainer_command[trainer_command.index(option) + 1], "7000")
        self.assertIn(
            "_7000steps",
            trainer_command[trainer_command.index("--result-dir") + 1],
        )

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
                    / DEFAULT_PREPROCESS_KEY
                    / ".batch-lock"
                ).exists()
            )

    def test_real_mode_runs_training_manifest_and_resumes_completed_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin, log = self.make_fake_uv(root)
            environment = {
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "STUB_UV_LOG": str(log),
            }
            first = self.run_script(
                root,
                "--scene",
                "iphone_18mm",
                extra_environment=environment,
            )
            second = self.run_script(
                root,
                "--scene",
                "iphone_18mm",
                extra_environment=environment,
            )
            dry_run = self.run_script(
                root,
                "--dry-run",
                "--scene",
                "iphone_18mm",
                extra_environment=environment,
            )

            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(dry_run.returncode, 0, dry_run.stdout + dry_run.stderr)
            self.assertEqual(
                log.read_text(encoding="utf-8").splitlines(),
                ["extract_frames.py", "prepare_colmap.py", "simple_trainer.py"],
            )
            result_dir = (
                root
                / "outputs"
                / "iphone_18mm"
                / f"{DEFAULT_PREPROCESS_KEY}_30000steps"
            )
            manifest = json.loads(
                (result_dir / "pipeline_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["identity"]["trainer"]["config"]["max_steps"],
                30000,
            )
            self.assertIn("Skipping completed training", second.stdout)
            self.assertIn(str(TRAINER), dry_run.stdout)
            self.assertNotIn("Skipping completed training", dry_run.stdout)

    def test_rejects_cached_colmap_with_wrong_camera_model(self) -> None:
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
            stats_path = (
                root
                / "work"
                / DEFAULT_PREPROCESS_KEY
                / "iphone_18mm"
                / "dataset"
                / "reconstruction_stats.json"
            )
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["camera_model"] = "SIMPLE_RADIAL"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")

            second = self.run_script(
                root,
                "--preprocess-only",
                "--scene",
                "iphone_18mm",
                extra_environment=environment,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 1)
            self.assertIn("stale or incomplete COLMAP state", second.stdout)
            self.assertEqual(
                log.read_text(encoding="utf-8").splitlines(),
                ["extract_frames.py", "prepare_colmap.py"],
            )

    def test_does_not_steal_live_lock_owned_by_another_user(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin, log = self.make_fake_uv(root)
            key = DEFAULT_PREPROCESS_KEY
            lock = root / "work" / key / ".batch-lock"
            lock.mkdir(parents=True)
            (lock / "hostname").write_text(
                subprocess.check_output(["hostname"], text=True).strip(),
                encoding="utf-8",
            )
            (lock / "pid").write_text(str(os.getpid()), encoding="utf-8")
            (lock / "uid").write_text(str(os.getuid() + 1), encoding="utf-8")

            result = self.run_script(
                root,
                "--preprocess-only",
                "--scene",
                "iphone_18mm",
                extra_environment={
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                    "STUB_UV_LOG": str(log),
                },
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("another user's batch is active", result.stderr)
            self.assertTrue(lock.is_dir())
            self.assertEqual(list(lock.parent.glob(".batch-lock.stale-*")), [])
            self.assertFalse(log.exists())

    def test_permission_denied_process_is_still_treated_as_alive(self) -> None:
        try:
            os.kill(1, 0)
        except PermissionError:
            pass
        except OSError as error:
            self.skipTest(f"PID 1 cannot exercise EPERM here: {error}")
        else:
            self.skipTest("Current user may signal PID 1; no EPERM to exercise")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin, log = self.make_fake_uv(root)
            key = DEFAULT_PREPROCESS_KEY
            lock = root / "work" / key / ".batch-lock"
            lock.mkdir(parents=True)
            (lock / "hostname").write_text(
                subprocess.check_output(["hostname"], text=True).strip(),
                encoding="utf-8",
            )
            (lock / "pid").write_text("1", encoding="utf-8")
            (lock / "uid").write_text("0", encoding="utf-8")
            result = self.run_script(
                root,
                "--preprocess-only",
                "--scene",
                "iphone_18mm",
                extra_environment={
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                    "STUB_UV_LOG": str(log),
                },
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("another user's batch is active", result.stderr)
            self.assertTrue(lock.is_dir())
            self.assertFalse(log.exists())

    def test_archives_dead_same_host_lock_and_continues(self) -> None:
        dead_pid = 2_147_483_647
        try:
            os.kill(dead_pid, 0)
        except ProcessLookupError:
            pass
        except OSError as error:
            self.skipTest(f"Cannot obtain a guaranteed dead PID: {error}")
        else:
            self.skipTest("Chosen dead PID unexpectedly exists")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin, log = self.make_fake_uv(root)
            key = DEFAULT_PREPROCESS_KEY
            lock = root / "work" / key / ".batch-lock"
            lock.mkdir(parents=True)
            (lock / "hostname").write_text(
                subprocess.check_output(["hostname"], text=True).strip(),
                encoding="utf-8",
            )
            (lock / "pid").write_text(str(dead_pid), encoding="utf-8")
            (lock / "uid").write_text(str(os.getuid()), encoding="utf-8")
            result = self.run_script(
                root,
                "--preprocess-only",
                "--scene",
                "iphone_18mm",
                extra_environment={
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                    "STUB_UV_LOG": str(log),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                len(list(lock.parent.glob(".batch-lock.stale-*"))),
                1,
            )
            self.assertEqual(
                log.read_text(encoding="utf-8").splitlines(),
                ["extract_frames.py", "prepare_colmap.py"],
            )

    def test_retry_archives_partial_colmap_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = self.prepare_captures(root)
            fake_bin, log = self.make_fake_uv(root)
            key = DEFAULT_PREPROCESS_KEY
            scene_work = root / "work" / key / "iphone_18mm"
            dataset = scene_work / "dataset"
            images = dataset / "images"
            images.mkdir(parents=True)
            for index in range(2):
                (images / f"{index:06d}.jpg").write_bytes(b"jpeg")
            source = (captures / "iphone_18mm.mov").resolve()
            stat = source.stat()
            images_digest = hashlib.sha256()
            image_content_digest = hashlib.sha256(b"jpeg").hexdigest()
            for index in range(2):
                images_digest.update(f"{index:06d}.jpg".encode("utf-8"))
                images_digest.update(b"\0")
                images_digest.update(image_content_digest.encode("ascii"))
                images_digest.update(b"\0")
            (dataset / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_type": "single_video",
                        "dataset_instance_id": str(uuid.uuid4()),
                        "source_video": str(source),
                        "source_size_bytes": stat.st_size,
                        "source_mtime_ns": stat.st_mtime_ns,
                        "source_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "images_sha256": images_digest.hexdigest(),
                        "sample_fps": 6.0,
                        "max_frames": 100000,
                        "selected_indices": [0, 1],
                    }
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
