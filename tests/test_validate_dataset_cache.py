from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_dataset_cache import validate_cache


def source_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def create_numbered_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for index in range(count):
        (directory / f"{index:06d}.jpg").write_bytes(b"jpeg")


class DatasetCacheValidationTests(unittest.TestCase):
    def test_accepts_matching_single_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.mov"
            source.write_bytes(b"video")
            dataset = root / "dataset"
            dataset.mkdir()
            identity = source_metadata(source)
            (dataset / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_type": "single_video",
                        "source_video": identity["path"],
                        "source_size_bytes": identity["size_bytes"],
                        "source_mtime_ns": identity["mtime_ns"],
                        "sample_fps": 6.0,
                        "max_frames": 100000,
                        "selected_indices": [0, 5],
                    }
                ),
                encoding="utf-8",
            )
            create_numbered_images(dataset / "images", 2)

            validate_cache(dataset, source, 6.0, 100000)

    def test_rejects_changed_single_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.mov"
            source.write_bytes(b"old")
            dataset = root / "dataset"
            dataset.mkdir()
            identity = source_metadata(source)
            (dataset / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_type": "single_video",
                        "source_video": identity["path"],
                        "source_size_bytes": identity["size_bytes"],
                        "source_mtime_ns": identity["mtime_ns"],
                        "sample_fps": 6.0,
                        "max_frames": 100000,
                        "selected_indices": [0, 5],
                    }
                ),
                encoding="utf-8",
            )
            create_numbered_images(dataset / "images", 2)
            source.write_bytes(b"new and different")

            with self.assertRaisesRegex(ValueError, "changed"):
                validate_cache(dataset, source, 6.0, 100000)

    def test_rejects_missing_single_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.mov"
            source.write_bytes(b"video")
            identity = source_metadata(source)
            single = root / "single"
            single.mkdir()
            (single / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_type": "single_video",
                        "source_video": identity["path"],
                        "source_size_bytes": identity["size_bytes"],
                        "source_mtime_ns": identity["mtime_ns"],
                        "sample_fps": 6.0,
                        "max_frames": 100000,
                        "selected_indices": [0, 5],
                    }
                ),
                encoding="utf-8",
            )
            create_numbered_images(single / "images", 1)
            with self.assertRaisesRegex(ValueError, "frame set"):
                validate_cache(single, source, 6.0, 100000)

    def test_rejects_old_or_parameter_mismatched_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.mov"
            source.write_bytes(b"video")
            dataset = root / "dataset"
            dataset.mkdir()
            metadata_path = dataset / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "source_video": str(source.resolve()),
                        "sample_fps": 6.0,
                        "max_frames": 100000,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "predates"):
                validate_cache(dataset, source, 6.0, 100000)

            metadata_path.write_text(
                json.dumps(
                    {
                        "dataset_type": "single_video",
                        "source_video": str(source.resolve()),
                        "source_size_bytes": source.stat().st_size,
                        "source_mtime_ns": source.stat().st_mtime_ns,
                        "sample_fps": 3.0,
                        "max_frames": 100000,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample_fps"):
                validate_cache(dataset, source, 6.0, 100000)


if __name__ == "__main__":
    unittest.main()
