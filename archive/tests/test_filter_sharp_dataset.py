from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from archive.scripts.filter_sharp_dataset import (
    _removal_set,
    build_parser,
    filter_sharp_dataset,
)


BASE_NAMES = tuple(f"00_vid_{index:06d}.jpg" for index in range(6))
CANDIDATE_NAMES = ("00_vid_m000050.jpg", "00_vid_m000150.jpg")


class FakeImage:
    def __init__(self, name: str, frame_id: int, has_pose: bool = True) -> None:
        self.name = name
        self.frame_id = frame_id
        self.has_pose = has_pose


class FakeReconstruction:
    def __init__(self, path: str) -> None:
        self.loaded_from = path
        names = [*BASE_NAMES, *CANDIDATE_NAMES]
        self.images = {
            index: FakeImage(name, frame_id=100 + index)
            for index, name in enumerate(names)
        }
        self.deregistered: list[int] = []
        self.written_to: Path | None = None

    def num_reg_images(self) -> int:
        return len(self.images) - len(self.deregistered)

    def num_points3D(self) -> int:
        return 5000 - 100 * len(self.deregistered)

    def deregister_frame(self, frame_id: int) -> None:
        self.deregistered.append(frame_id)

    def write(self, path: str) -> None:
        self.written_to = Path(path)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (self.written_to / name).write_bytes(name.encode("ascii"))


def _make_base(root: Path) -> Path:
    base = root / "base"
    images = base / "images"
    images.mkdir(parents=True)
    for name in (*BASE_NAMES, *CANDIDATE_NAMES):
        (images / name).write_bytes(b"jpeg " + name.encode("ascii"))
    model = base / "sparse" / "0"
    model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(b"model")
    (base / "metadata.json").write_text(json.dumps({"dataset_type": "video_set"}))
    (base / "reconstruction_stats.json").write_text(
        json.dumps({"status": "passed", "num_registered_images": 8})
    )
    return base


def _write_scores(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "scores.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "raw", "ratio_local"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _default_scores(root: Path) -> Path:
    # 000000 fails the ratio rule, 000001 fails the raw floor, rest pass.
    rows = []
    for index, name in enumerate(BASE_NAMES):
        rows.append(
            {
                "name": name,
                "raw": 5.0 if index == 1 else 80.0,
                "ratio_local": 0.2 if index == 0 else 0.9,
            }
        )
    return _write_scores(root, rows)


def _write_protect(root: Path) -> Path:
    path = root / "protect.txt"
    path.write_text("\n".join(CANDIDATE_NAMES) + "\n")
    return path


class RemovalRuleTests(unittest.TestCase):
    def test_ratio_and_raw_floor_are_independent_triggers(self) -> None:
        rows = [
            {"name": "a.jpg", "raw": "100", "ratio_local": "0.30"},
            {"name": "b.jpg", "raw": "10", "ratio_local": "0.90"},
            {"name": "c.jpg", "raw": "100", "ratio_local": "0.90"},
        ]
        self.assertEqual(_removal_set(rows, 0.5, 20.0), {"a.jpg", "b.jpg"})
        self.assertEqual(_removal_set(rows, 0.5, 0.0), {"a.jpg"})

    def test_non_finite_scores_are_rejected(self) -> None:
        rows = [{"name": "a.jpg", "raw": "nan", "ratio_local": "0.9"}]
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            _removal_set(rows, 0.5, 20.0)


class FilterDatasetTests(unittest.TestCase):
    def _run(self, root: Path, **kwargs):
        fake_module = types.SimpleNamespace(Reconstruction=FakeReconstruction)
        with patch.dict(sys.modules, {"pycolmap": fake_module}):
            return filter_sharp_dataset(
                _make_base(root),
                root / "filtered",
                _default_scores(root),
                protect_names=_write_protect(root),
                **kwargs,
            )

    def test_removes_failing_frames_and_keeps_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = self._run(root)
            kept = sorted(p.name for p in (output / "images").iterdir())
            self.assertNotIn(BASE_NAMES[0], kept)
            self.assertNotIn(BASE_NAMES[1], kept)
            for name in CANDIDATE_NAMES:
                self.assertIn(name, kept)
            self.assertEqual(len(kept), len(BASE_NAMES) - 2 + len(CANDIDATE_NAMES))
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                self.assertTrue((output / "sparse" / "0" / name).is_file())
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["sharpness_filter"]["num_removed"], 2)
            stats = json.loads((output / "reconstruction_stats.json").read_text())
            self.assertEqual(stats["num_registered_images"], 6)
            self.assertEqual(stats["num_points3D"], 4800)
            with (output / "sharpness_scores.csv").open() as handle:
                rows = {r["name"]: r["removed"] for r in csv.DictReader(handle)}
            self.assertEqual(rows[BASE_NAMES[0]], "True")
            self.assertEqual(rows[BASE_NAMES[2]], "False")

    def test_unscored_unprotected_image_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            (base / "images" / "00_vid_999999.jpg").write_bytes(b"stray")
            scores = _default_scores(root)
            with self.assertRaisesRegex(ValueError, "neither scored nor protected"):
                filter_sharp_dataset(
                    base, root / "filtered", scores,
                    protect_names=_write_protect(root),
                )
            self.assertFalse((root / "filtered").exists())

    def test_rejects_existing_output_and_removing_everything(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            scores = _write_scores(
                root,
                [
                    {"name": n, "raw": 1.0, "ratio_local": 0.01}
                    for n in (*BASE_NAMES, *CANDIDATE_NAMES)
                ],
            )
            (root / "exists").mkdir()
            with self.assertRaises(FileExistsError):
                filter_sharp_dataset(base, root / "exists", scores)
            with self.assertRaisesRegex(ValueError, "every image"):
                filter_sharp_dataset(base, root / "filtered", scores)
            self.assertFalse(
                any(p.name.startswith(".filtered") for p in root.iterdir())
            )

    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args(
            ["base", "out", "--scores", "s.csv"]
        )
        self.assertEqual(args.ratio_threshold, 0.50)
        self.assertEqual(args.raw_floor, 20.0)
        self.assertIsNone(args.protect_names)


if __name__ == "__main__":
    unittest.main()
