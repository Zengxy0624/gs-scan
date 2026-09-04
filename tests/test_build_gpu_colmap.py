from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_gpu_colmap.sh"


class BuildGpuColmapScriptTests(unittest.TestCase):
    def run_script(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            cwd=PROJECT_ROOT,
            env=dict(os.environ),
            check=False,
            text=True,
            capture_output=True,
        )

    def test_dry_run_lists_all_phases_with_gpu_flags_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "gpu"
            result = self.run_script("--dry-run", "--root", str(root), "--jobs", "8", "--cuda-arch", "90;120")

            self.assertEqual(result.returncode, 0, result.stderr)
            out = result.stdout
            for expected in (
                "phase env", "phase ceres", "phase colmap", "phase pycolmap", "phase verify",
                "micromamba create -y -n gpubuild -c conda-forge --override-channels",
                "gpubuild nvcc --version", "-DWITH_CUDA=ON", "-DWITH_SUITESPARSE=ON",
                f"-Dcudss_DIR={root}/mamba_root/envs/gpubuild/lib/cmake/cudss",
                "patch_ceres_arch", "-DCUDA_ENABLED=ON", "-DCMAKE_CUDA_ARCHITECTURES=90\;120",
                "-DGUI_ENABLED=OFF", "-DONNX_ENABLED=OFF", "-DFETCH_FAISS=OFF",
                "ninja -C", "-j 8", "install-build-requirements.py",
                "pip install --no-build-isolation --no-deps -v",
                "Dry run complete",
            ):
                self.assertIn(expected, out, expected)
            self.assertFalse(root.exists())
            self.assertFalse((PROJECT_ROOT / "work" / "_logs" / "build_gpu_colmap.log.dry").exists())

    def test_phase_selection_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "gpu"
            result = self.run_script("--dry-run", "--root", str(root), "--phases", "colmap,verify")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("phase env", result.stdout)
            self.assertNotIn("phase ceres", result.stdout)
            self.assertIn("phase colmap", result.stdout)
            self.assertIn("phase verify", result.stdout)
            self.assertIn("colmap=4.2.0 ceres=master", result.stdout)
            self.assertIn("arch=120 cuda=12.9", result.stdout)
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120", result.stdout)

    def test_rejects_invalid_options(self) -> None:
        for arguments in (
            ["--jobs", "0"], ["--cuda-arch", "sm_120"], ["--cuda-version", "12"],
            ["--phases", "env,gui"], ["--nice", "25"], ["--bogus"],
        ):
            with self.subTest(arguments=arguments):
                result = self.run_script("--dry-run", *arguments)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
