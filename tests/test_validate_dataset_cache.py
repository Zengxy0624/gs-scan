from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.validate_dataset_cache import validate_cache


def source_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def create_numbered_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for index in range(count):
        (directory / f"{index:06d}.jpg").write_bytes(b"jpeg")


def numbered_images_digest(count: int) -> str:
    digest = hashlib.sha256()
    content_digest = hashlib.sha256(b"jpeg").hexdigest().encode("ascii")
    for index in range(count):
        digest.update(f"{index:06d}.jpg".encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest)
        digest.update(b"\0")
    return digest.hexdigest()


def create_sparse_model(directory: Path) -> dict[str, dict[str, object]]:
    directory.mkdir(parents=True)
    identities = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        content = name.encode("ascii")
        (directory / name).write_bytes(content)
        identities[name] = {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return identities


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
                        "dataset_instance_id": str(uuid.uuid4()),
                        "source_video": identity["path"],
                        "source_size_bytes": identity["size_bytes"],
                        "source_mtime_ns": identity["mtime_ns"],
                        "source_sha256": identity["sha256"],
                        "images_sha256": numbered_images_digest(2),
                        "sample_fps": 6.0,
                        "max_frames": 100000,
                        "selected_indices": [0, 5],
                    }
                ),
                encoding="utf-8",
            )
            create_numbered_images(dataset / "images", 2)
            model_files = create_sparse_model(dataset / "sparse" / "0")
            (dataset / "reconstruction_stats.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "camera_model": "SIMPLE_RADIAL",
                        "matching_overlap": 15,
                        "matching_quadratic_overlap": True,
                        "minimum_registration_ratio": 0.5,
                        "num_input_images": 2,
                        "selected_model_files": model_files,
                        "selected_model_path": "sparse/0",
                    }
                ),
                encoding="utf-8",
            )
            relocated = root / "relocated.mov"
            relocated.write_bytes(source.read_bytes())

            validate_cache(
                dataset,
                relocated,
                6.0,
                100000,
                camera_model="SIMPLE_RADIAL",
                overlap=15,
                quadratic_overlap=True,
                require_colmap=True,
            )
            with self.assertRaisesRegex(ValueError, "camera_model"):
                validate_cache(
                    dataset,
                    relocated,
                    6.0,
                    100000,
                    camera_model="OPENCV",
                    overlap=15,
                    quadratic_overlap=True,
                    require_colmap=True,
                )
            same_size_replacement = root / "same-size.mov"
            same_size_replacement.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                validate_cache(
                    dataset,
                    same_size_replacement,
                    6.0,
                    100000,
                )

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
                        "dataset_instance_id": str(uuid.uuid4()),
                        "source_video": identity["path"],
                        "source_size_bytes": identity["size_bytes"],
                        "source_mtime_ns": identity["mtime_ns"],
                        "source_sha256": identity["sha256"],
                        "images_sha256": numbered_images_digest(2),
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
                        "dataset_instance_id": str(uuid.uuid4()),
                        "source_video": identity["path"],
                        "source_size_bytes": identity["size_bytes"],
                        "source_mtime_ns": identity["mtime_ns"],
                        "source_sha256": identity["sha256"],
                        "images_sha256": numbered_images_digest(2),
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
                        "dataset_instance_id": str(uuid.uuid4()),
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
