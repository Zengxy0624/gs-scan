from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.build_run_manifest import build_manifest, manifest_identity_matches


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


def manifest(
    dataset: Path,
    source: Path,
    *,
    camera_model: str = "SIMPLE_RADIAL",
    max_steps: int = 30000,
    revision: str = REVISION,
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
            15,
            0.5,
            max_steps,
            camera_model,
            revision,
            8,
            "6",
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


if __name__ == "__main__":
    unittest.main()
