from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_object_frames.sh"


class PrepareObjectFramesTests(unittest.TestCase):
    def run_script(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments], cwd=PROJECT_ROOT, env=dict(os.environ),
            check=False, text=True, capture_output=True,
        )

    def commands(self, result: subprocess.CompletedProcess[str]) -> list[str]:
        return [line for line in result.stdout.splitlines() if line.startswith("+ ")]

    def test_dry_run_chains_extract_filter_annotate_merge_per_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "captures").mkdir(); (root / "captures" / "annotation.csv").write_text("clip,file\n")
            result = self.run_script(
                "--dry-run", "--name", "mannequin_back", "--capture-root", str(root / "captures"),
                "--work-root", str(root / "work"), "--make", "Apple", "--model", "iPhone 17 Pro",
                "--software", "Blackmagic Cam", "JC_13_1341", "JC_14_1343",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            cmds = self.commands(result)
            # per clip: extract -> filter -> rm staging -> annotate ; then one merge
            self.assertEqual(len(cmds), 4 * 2 + 1, "\n".join(cmds))
            extract, filt, rm, annotate = cmds[0:4]
            self.assertIn("extract_frames.py", extract)
            for flag in ("--sample-fps 3", "--sharpness-window 0.3", "--sharpness-keep-ratio 0.8", "--sharpness-workers 16"):
                self.assertIn(flag, extract, flag)
            self.assertIn("/work/.JC_13_1341.unfiltered", extract)
            self.assertIn("filter_frame_cache.py", filt); self.assertIn("--ratio-threshold 0.7", filt)
            self.assertIn(f"{root}/work/JC_13_1341", filt)
            self.assertTrue(rm.startswith("+ rm -rf"))
            self.assertIn("annotate_frame_cache.py", annotate)
            for flag in ("--make Apple", "--model iPhone\\ 17\\ Pro", "--software Blackmagic\\ Cam", "annotation.csv"):
                self.assertIn(flag, annotate, flag)
            merge = cmds[-1]
            self.assertIn("merge_frame_datasets.py", merge)
            self.assertIn(f"{root}/work/mannequin_back", merge)
            self.assertIn("JC_13_1341", merge); self.assertIn("JC_14_1343", merge)
            self.assertIn("--link-mode hardlink", merge)
            self.assertIn("run_object_colmap.sh --dataset", result.stdout)
            self.assertFalse((root / "work").exists())

    def test_drop_ratio_zero_skips_the_filter_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script(
                "--dry-run", "--name", "tube", "--capture-root", str(root), "--work-root", str(root / "w"),
                "--drop-ratio", "0", "--sample-fps", "2", "--keep-ratio", "1.0", "A", "B",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            cmds = self.commands(result)
            self.assertEqual(len(cmds), 2 * 2 + 1)
            self.assertNotIn("filter_frame_cache.py", "\n".join(cmds))
            self.assertNotIn("--make", cmds[1])  # no camera identity unless given (or in the CSV)
            self.assertIn(f"{root}/w/A", cmds[0]); self.assertIn("--sample-fps 2", cmds[0]); self.assertIn("--sharpness-keep-ratio 1.0", cmds[0])

    def test_project_sets_default_roots(self) -> None:
        result = self.run_script("--dry-run", "--project", "demo", "--name", "obj", "A", "B")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{PROJECT_ROOT}/work/demo/obj", result.stdout)
        self.assertIn(f"{PROJECT_ROOT}/captures/demo/annotation.csv", result.stdout)
        self.assertIn(f"{PROJECT_ROOT}/captures/demo/A.mov", self.commands(result)[0])

    def test_rejects_bad_arguments(self) -> None:
        for arguments in (
            ["--dry-run", "A", "B"],                       # missing --name
            ["--dry-run", "--name", "x", "A", "B"],         # missing --project / roots
            ["--dry-run", "--project", "../p", "--name", "x", "A", "B"],
            ["--dry-run", "--project", "p", "--name", "x", "A"],              # one clip
            ["--dry-run", "--project", "p", "--name", "../x", "A", "B"],      # unsafe name
            ["--dry-run", "--project", "p", "--name", "x", "--drop-ratio", "abc", "A", "B"],
            ["--dry-run", "--project", "p", "--name", "x", "--sharpness-workers", "0", "A", "B"],
            ["--dry-run", "--project", "p", "--name", "x", "--bogus", "A", "B"],
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.run_script(*arguments).returncode, 2)

    def test_real_run_requires_videos_and_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_script("--name", "x", "--capture-root", str(root), "--work-root", str(root / "w"), "A", "B")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not found", result.stderr)
            self.assertFalse((root / "w").exists())


if __name__ == "__main__":
    unittest.main()
