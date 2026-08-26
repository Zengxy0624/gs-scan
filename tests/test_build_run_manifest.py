from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.build_run_manifest import (
    build_manifest,
    build_parser,
    manifest_identity_matches,
)


REVISION = "a" * 40


def image_tree_sha256(images: Path) -> str:
    digest = hashlib.sha256()
    for image in sorted(images.iterdir()):
        digest.update(image.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            hashlib.sha256(image.read_bytes()).hexdigest().encode("ascii")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def create_dataset(root: Path, source: Path) -> Path:
    dataset = root / "dataset"
    images = dataset / "images"
    model = dataset / "sparse" / "0"
    images.mkdir(parents=True)
    model.mkdir(parents=True)
    for index in range(2):
        (images / f"{index:06d}.jpg").write_bytes(f"image-{index}".encode())
    model_files = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        content = name.encode("ascii")
        (model / name).write_bytes(content)
        model_files[name] = {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    stat = source.stat()
    (dataset / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_type": "single_video",
                "dataset_instance_id": str(uuid.uuid4()),
                "source_video": str(source.resolve()),
                "source_size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "images_sha256": image_tree_sha256(images),
                "sample_fps": 6.0,
                "max_frames": 100000,
                "jpeg_quality": 95,
                "selected_indices": [0, 5],
            }
        ),
        encoding="utf-8",
    )
    (dataset / "reconstruction_stats.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "camera_model": "SIMPLE_RADIAL",
                "matching_overlap": 15,
                "matching_quadratic_overlap": True,
                "minimum_registration_ratio": 0.5,
                "num_input_images": 2,
                "num_registered_images": 2,
                "selected_model_files": model_files,
                "selected_model_path": "sparse/0",
            }
        ),
        encoding="utf-8",
    )
    return dataset


def source_metadata(
    source: Path,
    image_prefix: str,
    selected_indices: list[int],
) -> dict[str, object]:
    stat = source.stat()
    return {
        "source_video": str(source.resolve()),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "image_prefix": image_prefix,
        "selected_indices": selected_indices,
    }


def create_video_set_dataset(root: Path, sources: list[Path]) -> Path:
    dataset = create_dataset(root, sources[0])
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_type"] = "video_set"
    for key in (
        "source_video",
        "source_size_bytes",
        "source_mtime_ns",
        "source_sha256",
        "selected_indices",
    ):
        metadata.pop(key)
    metadata["sources"] = [
        source_metadata(source, f"source_{index:03d}", [index])
        for index, source in enumerate(sources)
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    stats_path = dataset / "reconstruction_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["minimum_source_registration_ratio"] = 0.4
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return dataset


def manifest(
    dataset: Path,
    source: Path | list[Path],
    *,
    camera_model: str = "SIMPLE_RADIAL",
    max_steps: int = 30000,
    revision: str = REVISION,
    overlap: int = 15,
    matching_strategy: str = "sequential",
    min_source_registration_ratio: float | None = None,
) -> dict[str, object]:
    stats_path = dataset / "reconstruction_stats.json"
    original_stats = stats_path.read_text(encoding="utf-8")
    stats = json.loads(original_stats)
    stats["camera_model"] = camera_model
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    try:
        return build_manifest(
            "scene",
            dataset,
            source,
            6.0,
            100000,
            overlap,
            0.5,
            max_steps,
            camera_model,
            revision,
            8,
            "6",
            matching_strategy,
            min_source_registration_ratio,
        )
    finally:
        stats_path.write_text(original_stats, encoding="utf-8")


class RunManifestTests(unittest.TestCase):
    def test_semantic_identity_is_stable_but_tracks_result_affecting_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.mov"
            source.write_bytes(b"source")
            dataset = create_dataset(root, source)

            baseline = manifest(dataset, source)
            self.assertEqual(baseline["schema_version"], 2)
            self.assertIn("source", baseline["provenance"])
            self.assertNotIn("sources", baseline["provenance"])
            self.assertEqual(
                baseline["identity"]["dataset"]["reconstruction"],
                {
                    "recipe_version": 3,
                    "camera_mode": "SINGLE",
                    "camera_model": "SIMPLE_RADIAL",
                    "matching": "sequential",
                    "overlap": 15,
                    "quadratic_overlap": True,
                },
            )
            self.assertEqual(
                baseline["identity"]["trainer"]["implementation"]["commit"],
                REVISION,
            )
            self.assertEqual(
                baseline["identity"]["trainer"]["config"],
                {
                    "preset": "default",
                    "data_type": "colmap",
                    "data_factor": 1,
                    "camera_model": "pinhole",
                    "max_steps": 30000,
                    "eval_steps": [30000],
                    "save_steps": [30000],
                    "ply_steps": [30000],
                    "save_ply": True,
                    "disable_viewer": True,
                },
            )
            self.assertEqual(
                baseline["identity"]["dataset"]["extraction"][
                    "images_sha256"
                ],
                baseline["provenance"]["dataset_snapshot"]["images"][
                    "sha256"
                ],
            )
            source.touch()
            touched = manifest(dataset, source)
            self.assertEqual(
                baseline["identity_sha256"],
                touched["identity_sha256"],
            )

            sparse_before = baseline["provenance"]["dataset_snapshot"]
            points_path = dataset / "sparse" / "0" / "points3D.bin"
            points_path.write_bytes(
                b"byte-different reconstruction"
            )
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                manifest(dataset, source)

            stats_path = dataset / "reconstruction_stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["selected_model_files"]["points3D.bin"] = {
                "size_bytes": points_path.stat().st_size,
                "sha256": hashlib.sha256(points_path.read_bytes()).hexdigest(),
            }
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            metadata_path = dataset / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["dataset_instance_id"] = str(uuid.uuid4())
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            changed_snapshot = manifest(dataset, source)
            self.assertEqual(
                baseline["identity_sha256"],
                changed_snapshot["identity_sha256"],
            )
            self.assertNotEqual(
                sparse_before,
                changed_snapshot["provenance"]["dataset_snapshot"],
            )
            rebuilt_path = root / "rebuilt.json"
            rebuilt_path.write_text(
                json.dumps(changed_snapshot),
                encoding="utf-8",
            )

            for changed in (
                manifest(dataset, source, camera_model="OPENCV"),
                manifest(dataset, source, max_steps=7000),
                manifest(dataset, source, revision="b" * 40),
            ):
                self.assertNotEqual(
                    baseline["identity_sha256"],
                    changed["identity_sha256"],
                )

            expected_path = root / "expected.json"
            touched_path = root / "touched.json"
            changed_path = root / "changed.json"
            expected_path.write_text(json.dumps(baseline), encoding="utf-8")
            touched_path.write_text(json.dumps(touched), encoding="utf-8")
            changed_path.write_text(
                json.dumps(manifest(dataset, source, max_steps=7000)),
                encoding="utf-8",
            )
            changed_image = dataset / "images" / "000000.jpg"
            changed_image.write_bytes(b"different extracted pixels")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["dataset_instance_id"] = str(uuid.uuid4())
            metadata["images_sha256"] = image_tree_sha256(dataset / "images")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            changed_images_path = root / "changed-images.json"
            changed_images_path.write_text(
                json.dumps(manifest(dataset, source)),
                encoding="utf-8",
            )
            tampered = json.loads(json.dumps(baseline))
            tampered["identity"]["trainer"]["config"]["max_steps"] = 1
            tampered_path = root / "tampered.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            wrong_schema = json.loads(json.dumps(baseline))
            wrong_schema["schema_version"] = 999
            wrong_schema_path = root / "wrong-schema.json"
            wrong_schema_path.write_text(
                json.dumps(wrong_schema),
                encoding="utf-8",
            )
            changed_same_instance = json.loads(json.dumps(baseline))
            changed_same_instance["provenance"]["dataset_snapshot"][
                "reconstruction_stats"
            ]["num_points3D"] = 999
            changed_same_instance_path = root / "changed-snapshot.json"
            changed_same_instance_path.write_text(
                json.dumps(changed_same_instance),
                encoding="utf-8",
            )
            self.assertTrue(manifest_identity_matches(expected_path, touched_path))
            self.assertTrue(manifest_identity_matches(expected_path, rebuilt_path))
            self.assertFalse(
                manifest_identity_matches(
                    expected_path,
                    rebuilt_path,
                    strict_snapshot=True,
                )
            )
            self.assertFalse(manifest_identity_matches(expected_path, changed_path))
            self.assertFalse(
                manifest_identity_matches(expected_path, changed_images_path)
            )
            self.assertFalse(manifest_identity_matches(expected_path, tampered_path))
            self.assertFalse(
                manifest_identity_matches(expected_path, wrong_schema_path)
            )
            self.assertFalse(
                manifest_identity_matches(
                    expected_path,
                    changed_same_instance_path,
                )
            )

    def test_video_set_identity_tracks_source_content_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "inward.mov"
            second = root / "upward.mov"
            first.write_bytes(b"inward-source")
            second.write_bytes(b"upward-source")
            dataset = create_video_set_dataset(root, [first, second])

            baseline = manifest(
                dataset,
                [first, second],
                min_source_registration_ratio=0.4,
            )
            expected_hashes = [
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (first, second)
            ]
            self.assertEqual(
                baseline["identity"]["dataset"]["source_sha256"],
                expected_hashes,
            )
            self.assertNotIn("source", baseline["provenance"])
            self.assertEqual(
                [
                    snapshot["image_prefix"]
                    for snapshot in baseline["provenance"]["sources"]
                ],
                ["source_000", "source_001"],
            )
            self.assertEqual(
                baseline["provenance"]["sources"][1]["selected_indices"],
                [1],
            )

            with self.assertRaisesRegex(ValueError, "content no longer matches"):
                manifest(
                    dataset,
                    [second, first],
                    min_source_registration_ratio=0.4,
                )

            second.write_bytes(b"changed-upward-source")
            with self.assertRaisesRegex(ValueError, "content no longer matches"):
                manifest(
                    dataset,
                    [first, second],
                    min_source_registration_ratio=0.4,
                )

            metadata_path = dataset / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["sources"][1] = source_metadata(
                second,
                "source_001",
                [1],
            )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            changed = manifest(
                dataset,
                [first, second],
                min_source_registration_ratio=0.4,
            )
            self.assertNotEqual(
                baseline["identity_sha256"],
                changed["identity_sha256"],
            )

            metadata["sources"].reverse()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            reversed_sources = manifest(
                dataset,
                [second, first],
                min_source_registration_ratio=0.4,
            )
            self.assertEqual(
                reversed_sources["identity"]["dataset"]["source_sha256"],
                list(
                    reversed(
                        changed["identity"]["dataset"]["source_sha256"]
                    )
                ),
            )
            self.assertNotEqual(
                changed["identity_sha256"],
                reversed_sources["identity_sha256"],
            )

            with self.assertRaisesRegex(
                ValueError,
                "minimum_source_registration_ratio",
            ):
                manifest(
                    dataset,
                    [second, first],
                    min_source_registration_ratio=0.8,
                )

    def test_matching_strategy_is_validated_and_identified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.mov"
            source.write_bytes(b"source")
            dataset = create_dataset(root, source)

            # Legacy reconstruction stats have no matching_strategy field and
            # remain valid as sequential reconstructions.
            sequential = manifest(dataset, source)

            stats_path = dataset / "reconstruction_stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["matching_strategy"] = "exhaustive"
            # prepare_colmap records the requested overlap for diagnostics even
            # though exhaustive matching does not use it.
            stats["matching_overlap"] = 15
            stats["matching_quadratic_overlap"] = False
            stats_path.write_text(json.dumps(stats), encoding="utf-8")

            exhaustive = manifest(
                dataset,
                source,
                matching_strategy="exhaustive",
            )
            exhaustive_other_overlap = manifest(
                dataset,
                source,
                overlap=999,
                matching_strategy="exhaustive",
            )
            self.assertNotEqual(
                sequential["identity_sha256"],
                exhaustive["identity_sha256"],
            )
            self.assertEqual(
                exhaustive["identity_sha256"],
                exhaustive_other_overlap["identity_sha256"],
            )
            self.assertEqual(
                exhaustive["identity"]["dataset"]["reconstruction"],
                {
                    "recipe_version": 3,
                    "camera_mode": "SINGLE",
                    "camera_model": "SIMPLE_RADIAL",
                    "matching": "exhaustive",
                    "overlap": None,
                    "quadratic_overlap": None,
                },
            )
            with self.assertRaisesRegex(ValueError, "matching_strategy"):
                manifest(dataset, source)

    def test_vocab_tree_identity_tracks_full_reconstruction_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.mov"
            source.write_bytes(b"source")
            vocab_path = root / "vocab.bin"
            vocab_path.write_bytes(b"vocabulary tree")
            dataset = create_dataset(root, source)
            stats_path = dataset / "reconstruction_stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats.update(
                {
                    "camera_mode": "SINGLE",
                    "matching_strategy": "vocab_tree",
                    "matching_quadratic_overlap": False,
                    "vocab_tree": {
                        "num_images": 200,
                        "num_nearest_neighbors": 5,
                        "num_checks": 256,
                        "num_images_after_verification": 0,
                        "max_num_features": -1,
                        "path": str(vocab_path.resolve()),
                        "sha256": hashlib.sha256(
                            vocab_path.read_bytes()
                        ).hexdigest(),
                    },
                    "feature_extraction": {
                        "max_image_size": 3200,
                        "max_num_features": 8192,
                        "estimate_affine_shape": True,
                        "domain_size_pooling": True,
                    },
                    "feature_matching": {"guided_matching": True},
                    "mapper": {"init_min_tri_angle": 8.0},
                }
            )
            stats_path.write_text(json.dumps(stats), encoding="utf-8")

            baseline = manifest(
                dataset,
                source,
                matching_strategy="vocab_tree",
            )
            other_overlap = manifest(
                dataset,
                source,
                overlap=999,
                matching_strategy="vocab_tree",
            )
            reconstruction = baseline["identity"]["dataset"]["reconstruction"]
            self.assertEqual(reconstruction["recipe_version"], 3)
            self.assertEqual(reconstruction["matching"], "vocab_tree")
            self.assertIsNone(reconstruction["overlap"])
            self.assertIsNone(reconstruction["quadratic_overlap"])
            for name in (
                "vocab_tree",
                "feature_extraction",
                "feature_matching",
                "mapper",
            ):
                self.assertEqual(reconstruction[name], stats[name])
            self.assertEqual(
                baseline["identity_sha256"],
                other_overlap["identity_sha256"],
            )

            stats["feature_matching"]["guided_matching"] = False
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            changed = manifest(
                dataset,
                source,
                matching_strategy="vocab_tree",
            )
            self.assertNotEqual(
                baseline["identity_sha256"],
                changed["identity_sha256"],
            )

            stats["vocab_tree"]["sha256"] = "not-a-digest"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                manifest(
                    dataset,
                    source,
                    matching_strategy="vocab_tree",
                )

    def test_build_parser_accepts_ordered_repeatable_sources(self) -> None:
        args = build_parser().parse_args(
            [
                "build",
                "--scene",
                "scene",
                "--dataset",
                "dataset",
                "--source",
                "inward.mov",
                "--source",
                "upward.mov",
                "--gsplat-root",
                "gsplat",
                "--sample-fps",
                "6",
                "--max-frames",
                "100000",
                "--overlap",
                "15",
                "--matching-strategy",
                "vocab_tree",
                "--min-registration-ratio",
                "0.5",
                "--min-source-registration-ratio",
                "0.4",
                "--max-steps",
                "30000",
                "--camera-model",
                "OPENCV",
                "--colmap-threads",
                "8",
                "--gpu-id",
                "7",
                "--output",
                "run_manifest.json",
            ]
        )
        self.assertEqual(
            args.source,
            [Path("inward.mov"), Path("upward.mov")],
        )
        self.assertEqual(args.matching_strategy, "vocab_tree")
        self.assertEqual(args.min_source_registration_ratio, 0.4)


if __name__ == "__main__":
    unittest.main()
