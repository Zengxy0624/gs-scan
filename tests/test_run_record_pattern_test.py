from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_record_pattern_test.sh"
GSPLAT_ROOT = PROJECT_ROOT / "third-party" / "gsplat"
EXTRACTOR = PROJECT_ROOT / "scripts" / "extract_frames.py"
RECONSTRUCTOR = PROJECT_ROOT / "scripts" / "prepare_colmap.py"
MANIFEST_BUILDER = PROJECT_ROOT / "scripts" / "build_run_manifest.py"
TRAINER = GSPLAT_ROOT / "examples" / "simple_trainer.py"

INWARD_KEY = "fps6_max100000_camopencv_sequential15_reg0.8"
COMBINED_KEY = "fps6_max100000_camopencv_exhaustive_reg0.8_source0.8"


class RecordPatternBatchTests(unittest.TestCase):
    def prepare_captures(self, root: Path) -> Path:
        captures = root / "captures"
        captures.mkdir()
        (captures / "inward_orbit.mov").write_bytes(b"inward")
        (captures / "upward.mov").write_bytes(b"upward")
        return captures

    def run_script(
        self,
        root: Path,
        *arguments: str,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        captures = self.prepare_captures(root)
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
            timeout=20,
        )

    def expected_commands(
        self,
        root: Path,
        *,
        scene: str,
        key: str,
        matching: str,
        gpu: str,
        sources: list[Path],
    ) -> list[list[str]]:
        dataset = root / "work" / scene / key / "dataset"
        desired_manifest = (
            root / "work" / scene / key / "training_manifest_30000.json"
        )
        result_dir = root / "outputs" / scene / f"{key}_30000steps"

        extraction = [
            "uv",
            "run",
            "--project",
            str(GSPLAT_ROOT),
            "--no-sync",
            "python",
            str(EXTRACTOR),
            str(sources[0]),
            str(dataset),
        ]
        for source in sources[1:]:
            extraction.extend(["--additional-video", str(source)])
        extraction.extend(["--sample-fps", "6", "--max-frames", "100000"])

        reconstruction = [
            "uv",
            "run",
            "--project",
            str(GSPLAT_ROOT),
            "--no-sync",
            "python",
            str(RECONSTRUCTOR),
            str(dataset),
            "--matching",
            matching,
            "--overlap",
            "15",
            "--camera-model",
            "OPENCV",
            "--num-threads",
            "8",
            "--min-registration-ratio",
            "0.8",
        ]
        if matching == "exhaustive":
            reconstruction.extend(
                ["--min-source-registration-ratio", "0.8"]
            )

        manifest = [
            sys.executable,
            str(MANIFEST_BUILDER),
            "build",
            "--scene",
            scene,
            "--dataset",
            str(dataset),
        ]
        for source in sources:
            manifest.extend(["--source", str(source)])
        manifest.extend(
            [
                "--gsplat-root",
                str(GSPLAT_ROOT),
                "--sample-fps",
                "6",
                "--max-frames",
                "100000",
                "--overlap",
                "15",
                "--matching-strategy",
                matching,
                "--min-registration-ratio",
                "0.8",
                "--max-steps",
                "30000",
                "--camera-model",
                "OPENCV",
                "--colmap-threads",
                "8",
                "--gpu-id",
                gpu,
                "--output",
                str(desired_manifest),
            ]
        )
        if matching == "exhaustive":
            manifest.extend(["--min-source-registration-ratio", "0.8"])

        trainer = [
            "env",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID",
            f"CUDA_VISIBLE_DEVICES={gpu}",
            "uv",
            "run",
            "--project",
            str(GSPLAT_ROOT),
            "--no-sync",
            "python",
            str(TRAINER),
            "default",
            "--data-dir",
            str(dataset),
            "--data-factor",
            "1",
            "--result-dir",
            str(result_dir),
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
        ]
        return [
            extraction,
            reconstruction,
            manifest,
            ["mkdir", "-p", str(result_dir)],
            ["cp", str(desired_manifest), str(result_dir / "pipeline_manifest.json")],
            trainer,
        ]

    def test_dry_run_prints_exact_plans_with_expected_gpu_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script(root, "--dry-run")

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            captures = root / "captures"
            expected = self.expected_commands(
                root,
                scene="inward_orbit",
                key=INWARD_KEY,
                matching="sequential",
                gpu="6",
                sources=[captures / "inward_orbit.mov"],
            )
            expected.extend(
                self.expected_commands(
                    root,
                    scene="inward_orbit_upward",
                    key=COMBINED_KEY,
                    matching="exhaustive",
                    gpu="7",
                    sources=[
                        captures / "inward_orbit.mov",
                        captures / "upward.mov",
                    ],
                )
            )

            # Each complete command is rendered by one printf call, so the
            # two concurrent plans cannot merge command lines.
            for command in expected:
                with self.subTest(command=command):
                    rendered_body = " " + shlex.join(command)
                    self.assertIn(rendered_body, result.stdout)

            self.assertEqual(result.stdout.count(str(EXTRACTOR)), 2)
            self.assertEqual(result.stdout.count(str(RECONSTRUCTOR)), 2)
            self.assertEqual(result.stdout.count(str(MANIFEST_BUILDER)), 2)
            self.assertEqual(result.stdout.count(str(TRAINER)), 2)
            self.assertEqual(result.stdout.count("CUDA_VISIBLE_DEVICES=6"), 1)
            self.assertEqual(result.stdout.count("CUDA_VISIBLE_DEVICES=7"), 1)
            self.assertEqual(result.stdout.count("--no-sync"), 6)
            self.assertNotIn("setup_gsplat.sh", result.stdout)
            self.assertNotIn("module load", result.stdout)
            self.assertNotIn("uv sync", result.stdout)
            self.assertIn("Dry-run summary (no files written)", result.stdout)
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "outputs").exists())

    def test_rejects_same_gpu_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script(
                root,
                "--dry-run",
                "--gpu-inward",
                "06",
                "--gpu-combined",
                "6",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("require distinct GPU indices", result.stderr)
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "outputs").exists())

    def test_incomplete_and_live_output_locks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete_lock = root / "work" / ".batch-lock"
            incomplete_lock.mkdir(parents=True)
            result = self.run_script(root, "--preprocess-only")
            self.assertEqual(result.returncode, 1)
            self.assertIn("lock metadata is incomplete", result.stderr)
            self.assertNotIn("Launched ", result.stdout)
            self.assertTrue(incomplete_lock.is_dir())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_lock = root / "outputs" / ".batch-lock"
            output_lock.mkdir(parents=True)
            (output_lock / "pid").write_text(f"{os.getpid()}\n")
            (output_lock / "hostname").write_text(
                subprocess.check_output(["hostname"], text=True).strip() + "\n"
            )
            (output_lock / "uid").write_text(f"{os.getuid()}\n")
            (output_lock / "pgids").write_text("")
            result = self.run_script(root, "--preprocess-only")
            self.assertEqual(result.returncode, 1)
            self.assertIn("another batch is active", result.stderr)
            self.assertNotIn("Launched ", result.stdout)
            self.assertFalse((root / "work" / ".batch-lock").exists())
            self.assertTrue(output_lock.is_dir())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dead_process = subprocess.Popen(["true"])
            dead_process.wait()
            launch_lock = root / "work" / ".batch-lock"
            launch_lock.mkdir(parents=True)
            (launch_lock / "pid").write_text(f"{dead_process.pid}\n")
            (launch_lock / "hostname").write_text(
                subprocess.check_output(["hostname"], text=True).strip() + "\n"
            )
            (launch_lock / "uid").write_text(f"{os.getuid()}\n")
            (launch_lock / "pgids").write_text("")
            (launch_lock / "launch-in-progress").write_text(
                f"{dead_process.pid}\n"
            )
            result = self.run_script(root, "--preprocess-only")
            self.assertEqual(result.returncode, 1)
            self.assertIn("incomplete process-group launch", result.stderr)
            self.assertNotIn("Launched ", result.stdout)
            self.assertTrue(launch_lock.is_dir())

    def make_overlap_uv(self, root: Path) -> Path:
        fake_bin = root / "bin"
        fake_bin.mkdir()
        uv = fake_bin / "uv"
        uv.write_text(
            """#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

def option(arguments, name):
    return arguments[arguments.index(name) + 1]

arguments = sys.argv[1:]
script_index = next(
    index for index, value in enumerate(arguments) if value.endswith(".py")
)
script = Path(arguments[script_index]).name
stage = arguments[script_index + 1:]
state = Path(os.environ["STUB_STATE"])
state.mkdir(parents=True, exist_ok=True)

if script == "extract_frames.py":
    dataset = Path(stage[1])
    images = dataset / "images"
    images.mkdir(parents=True)
    (images / "000000.jpg").write_bytes(b"jpeg")
    (dataset / "metadata.json").write_text("{}", encoding="utf-8")
elif script == "prepare_colmap.py":
    matching = option(stage, "--matching")
    (state / f"{matching}.started").write_text(
        str(time.monotonic_ns()), encoding="utf-8"
    )
    deadline = time.monotonic() + 5
    expected = [state / "sequential.started", state / "exhaustive.started"]
    while not all(path.exists() for path in expected):
        if time.monotonic() >= deadline:
            raise SystemExit("other COLMAP phase did not start concurrently")
        time.sleep(0.01)
    time.sleep(0.1)
    (state / f"{matching}.ended").write_text(
        str(time.monotonic_ns()), encoding="utf-8"
    )
else:
    raise SystemExit(f"unexpected stubbed script: {script}")
""",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        return fake_bin

    def test_preprocess_only_overlaps_both_colmap_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = self.make_overlap_uv(root)
            state = root / "stub-state"
            result = self.run_script(
                root,
                "--preprocess-only",
                extra_environment={
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                    "STUB_STATE": str(state),
                },
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            starts = [
                int((state / f"{matching}.started").read_text())
                for matching in ("sequential", "exhaustive")
            ]
            ends = [
                int((state / f"{matching}.ended").read_text())
                for matching in ("sequential", "exhaustive")
            ]
            self.assertLess(max(starts), min(ends))
            self.assertEqual(result.stdout.count("Preprocessing complete for"), 2)
            self.assertFalse((root / "work" / ".batch-lock").exists())
            self.assertFalse((root / "outputs" / ".batch-lock").exists())

    def make_interrupt_uv(self, root: Path) -> Path:
        fake_bin = root / "interrupt-bin"
        fake_bin.mkdir()
        uv = fake_bin / "uv"
        uv.write_text(
            """#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

state = Path(os.environ["STUB_STATE"])
state.mkdir(parents=True, exist_ok=True)
own_pid = os.getpid()

def terminate(signum, frame):
    (state / f"uv-{own_pid}.term").write_text(str(signum))
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, terminate)
child_code = r'''\
import os
import signal
import sys
import time
from pathlib import Path
state = Path(sys.argv[1])
pid = os.getpid()
def terminate(signum, frame):
    (state / f"child-{pid}.term").write_text(str(signum))
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, terminate)
(state / f"child-{pid}.ready").write_text("ready")
while True:
    time.sleep(1)
'''
child = subprocess.Popen([sys.executable, "-c", child_code, str(state)])
deadline = time.monotonic() + 5
while not (state / f"child-{child.pid}.ready").exists():
    if time.monotonic() >= deadline:
        raise SystemExit("grandchild did not become ready")
    time.sleep(0.01)
(state / f"uv-{own_pid}.ready").write_text(str(child.pid))
child.wait()
""",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        return fake_bin

    def make_pair_guard_uv(self, root: Path) -> Path:
        fake_bin = root / "guard-bin"
        fake_bin.mkdir()
        uv = fake_bin / "uv"
        uv.write_text(
            """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
script_index = next(
    index for index, value in enumerate(arguments) if value.endswith(".py")
)
script = Path(arguments[script_index]).name
stage = arguments[script_index + 1:]
state = Path(os.environ["STUB_STATE"])
state.mkdir(parents=True, exist_ok=True)
if script == "extract_frames.py":
    dataset = Path(stage[1])
    images = dataset / "images"
    images.mkdir(parents=True)
    count = 3 if "--additional-video" in stage else 2
    for index in range(count):
        (images / f"{index:06d}.jpg").write_bytes(b"jpeg")
    (dataset / "metadata.json").write_text("{}")
elif script == "prepare_colmap.py":
    matching = stage[stage.index("--matching") + 1]
    (state / f"{matching}.prepare").write_text("called")
else:
    raise SystemExit(f"unexpected script: {script}")
""",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        return fake_bin

    def test_exhaustive_pair_guard_stops_only_oversized_combined_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = self.make_pair_guard_uv(root)
            state = root / "guard-state"
            result = self.run_script(
                root,
                "--preprocess-only",
                "--max-exhaustive-pairs",
                "1",
                extra_environment={
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                    "STUB_STATE": str(state),
                },
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("3 candidate pairs", result.stdout)
            self.assertIn("exceeds the safety cap of 1", result.stdout)
            self.assertTrue((state / "sequential.prepare").is_file())
            self.assertFalse((state / "exhaustive.prepare").exists())
            self.assertFalse((root / "work" / ".batch-lock").exists())
            self.assertFalse((root / "outputs" / ".batch-lock").exists())

    def test_interrupt_terminates_grandchildren_before_unlocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = self.prepare_captures(root)
            fake_bin = self.make_interrupt_uv(root)
            state = root / "interrupt-state"
            environment = {
                **os.environ,
                "CAPTURE_ROOT": str(captures),
                "WORK_ROOT": str(root / "work"),
                "OUTPUT_ROOT": str(root / "outputs"),
                "PROJECT_PYTHON": sys.executable,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "STUB_STATE": str(state),
            }
            process = subprocess.Popen(
                ["bash", str(SCRIPT), "--preprocess-only"],
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while len(list(state.glob("uv-*.ready"))) < 2:
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    self.fail(stdout + stderr)
                if time.monotonic() >= deadline:
                    process.kill()
                    process.communicate()
                    self.fail("stubbed uv processes did not become ready")
                time.sleep(0.02)

            process.terminate()
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 143, stdout + stderr)
            self.assertEqual(len(list(state.glob("uv-*.term"))), 2)
            self.assertEqual(len(list(state.glob("child-*.term"))), 2)
            self.assertFalse((root / "work" / ".batch-lock").exists())
            self.assertFalse((root / "outputs" / ".batch-lock").exists())

    def test_dead_launcher_lock_rejects_live_orphaned_process_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = self.prepare_captures(root)
            fake_bin = self.make_interrupt_uv(root)
            state = root / "orphan-state"
            environment = {
                **os.environ,
                "CAPTURE_ROOT": str(captures),
                "WORK_ROOT": str(root / "work"),
                "OUTPUT_ROOT": str(root / "outputs"),
                "PROJECT_PYTHON": sys.executable,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "STUB_STATE": str(state),
            }
            command = ["bash", str(SCRIPT), "--preprocess-only"]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process_groups: list[int] = []
            try:
                deadline = time.monotonic() + 10
                while len(list(state.glob("uv-*.ready"))) < 2:
                    if process.poll() is not None:
                        self.fail("launcher exited before orphan test was ready")
                    if time.monotonic() >= deadline:
                        self.fail("stubbed uv processes did not become ready")
                    time.sleep(0.02)

                process.kill()
                self.assertEqual(process.wait(timeout=5), -9)
                pgids_path = root / "work" / ".batch-lock" / "pgids"
                process_groups = [
                    int(value)
                    for value in pgids_path.read_text().splitlines()
                    if value
                ]
                self.assertEqual(len(process_groups), 2)
                self.assertNotIn(os.getpgrp(), process_groups)

                retry = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(retry.returncode, 1)
                self.assertIn(
                    "orphaned batch process group",
                    retry.stderr,
                )
                self.assertTrue((root / "work" / ".batch-lock").is_dir())
                self.assertTrue((root / "outputs" / ".batch-lock").is_dir())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                for group_id in process_groups:
                    try:
                        os.killpg(group_id, 15)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    alive = []
                    for group_id in process_groups:
                        try:
                            os.killpg(group_id, 0)
                        except ProcessLookupError:
                            continue
                        alive.append(group_id)
                    if not alive:
                        break
                    time.sleep(0.02)
                for group_id in (alive if process_groups else []):
                    try:
                        os.killpg(group_id, 9)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
