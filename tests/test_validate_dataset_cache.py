from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.validate_dataset_cache import _tree_sha256, build_parser, validate_cache


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


def create_video_set_cache(
    root: Path,
    sources: list[Path],
) -> Path:
    dataset = root / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    prefixes = ("00_inward_orbit_", "01_upward_")
    counts = (2, 3)
    records = []
    for source, prefix, count in zip(sources, prefixes, counts, strict=True):
        identity = source_metadata(source)
        for index in range(count):
            (images / f"{prefix}{index:06d}.jpg").write_bytes(b"jpeg")
        records.append(
            {
                "source_video": identity["path"],
                "source_size_bytes": identity["size_bytes"],
                "source_mtime_ns": identity["mtime_ns"],
                "source_sha256": identity["sha256"],
                "image_prefix": prefix,
                "selected_indices": list(range(count)),
                "output_frame_width": 64,
                "output_frame_height": 48,
            }
        )
    (dataset / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_type": "video_set",
                "dataset_instance_id": str(uuid.uuid4()),
                "sources": records,
                "total_selected_frames": sum(counts),
                "output_frame_width": 64,
                "output_frame_height": 48,
                "images_sha256": _tree_sha256(images),
                "sample_fps": 6.0,
                "max_frames": 100000,
            }
        ),
        encoding="utf-8",
    )
    model_files = create_sparse_model(dataset / "sparse" / "0")
    (dataset / "reconstruction_stats.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "camera_model": "OPENCV",
                "matching_overlap": 15,
                "matching_quadratic_overlap": True,
                "minimum_registration_ratio": 0.5,
                "num_input_images": sum(counts),
                "selected_model_files": model_files,
                "selected_model_path": "sparse/0",
            }
        ),
        encoding="utf-8",
    )
    return dataset


class DatasetCacheValidationTests(unittest.TestCase):
    def test_accepts_ordered_video_set_and_cli_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inward = root / "inward_orbit.mov"
            upward = root / "upward.mov"
            inward.write_bytes(b"inward")
            upward.write_bytes(b"upward-longer")
            dataset = create_video_set_cache(root, [inward, upward])

            validate_cache(
                dataset,
                [inward, upward],
                6.0,
                100000,
                camera_model="OPENCV",
                overlap=15,
                quadratic_overlap=True,
                require_colmap=True,
            )
            args = build_parser().parse_args(
                [
                    str(dataset),
                    str(inward),
                    str(upward),
                    "--sample-fps",
                    "6",
                    "--max-frames",
                    "100000",
                ]
            )
            self.assertEqual(args.source, [inward, upward])

    def test_video_set_rejects_source_order_and_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inward = root / "inward_orbit.mov"
            upward = root / "upward.mov"
            inward.write_bytes(b"inward")
            upward.write_bytes(b"upward-longer")
            dataset = create_video_set_cache(root, [inward, upward])

            with self.assertRaisesRegex(ValueError, "source capture 1 changed"):
                validate_cache(
                    dataset,
                    [upward, inward],
                    6.0,
                    100000,
                )

            upward.write_bytes(b"changed-upward-video")
            with self.assertRaisesRegex(ValueError, "source capture 2 changed"):
                validate_cache(
                    dataset,
                    [inward, upward],
                    6.0,
                    100000,
                )

    def test_video_set_rejects_frame_set_and_colmap_input_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inward = root / "inward_orbit.mov"
            upward = root / "upward.mov"
            inward.write_bytes(b"inward")
            upward.write_bytes(b"upward")
            dataset = create_video_set_cache(root, [inward, upward])

            missing = dataset / "images" / "01_upward_000002.jpg"
            content = missing.read_bytes()
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "frame set"):
                validate_cache(dataset, [inward, upward], 6.0, 100000)

            missing.write_bytes(content)
            stats_path = dataset / "reconstruction_stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["num_input_images"] = 2
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input-image count"):
                validate_cache(dataset, [inward, upward], 6.0, 100000)

    def test_validates_matching_strategy_and_per_source_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inward = root / "inward_orbit.mov"
            upward = root / "upward.mov"
            inward.write_bytes(b"inward")
            upward.write_bytes(b"upward")
            dataset = create_video_set_cache(root, [inward, upward])
            stats_path = dataset / "reconstruction_stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats.update(
                {
                    "matching_strategy": "exhaustive",
                    "matching_quadratic_overlap": False,
                    "minimum_source_registration_ratio": 0.8,
                    "source_registration": [
                        {
                            "source_index": 0,
                            "image_prefix": "00_inward_orbit_",
                            "num_input_images": 2,
                            "num_registered_images": 2,
                            "registration_ratio": 1.0,
                            "minimum_registration_ratio": 0.8,
                            "status": "passed",
                        },
                        {
                            "source_index": 1,
                            "image_prefix": "01_upward_",
                            "num_input_images": 3,
                            "num_registered_images": 3,
                            "registration_ratio": 1.0,
                            "minimum_registration_ratio": 0.8,
                            "status": "passed",
                        },
                    ],
                }
            )
            stats_path.write_text(json.dumps(stats), encoding="utf-8")

            validate_cache(
                dataset,
                [inward, upward],
                6.0,
                100000,
                require_colmap=True,
                matching_strategy="exhaustive",
                min_source_registration_ratio=0.8,
            )
            args = build_parser().parse_args(
                [
                    str(dataset),
                    str(inward),
                    str(upward),
                    "--sample-fps",
                    "6",
                    "--max-frames",
                    "100000",
                    "--matching-strategy",
                    "exhaustive",
                    "--min-source-registration-ratio",
                    "0.8",
                ]
            )
            self.assertEqual(args.matching_strategy, "exhaustive")
            self.assertEqual(args.min_source_registration_ratio, 0.8)

            with self.assertRaisesRegex(ValueError, "matching_strategy"):
                validate_cache(
                    dataset,
                    [inward, upward],
                    6.0,
                    100000,
                    matching_strategy="sequential",
                )
            stats["source_registration"][1]["status"] = "failed"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "per-source diagnostics"):
                validate_cache(
                    dataset,
                    [inward, upward],
                    6.0,
                    100000,
                    min_source_registration_ratio=0.8,
                )

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
