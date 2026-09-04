from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.extract_frames import _tree_sha256
from scripts.merge_frame_datasets import merge_frame_datasets
from scripts.validate_dataset_cache import validate_cache


class MergeFrameDatasetTests(unittest.TestCase):
    def make_source(
        self,
        root: Path,
        name: str,
        count: int,
        *,
        sample_fps: float = 3.0,
        dimensions: tuple[int, int] = (64, 48),
        sharpness_window: float | None = None,
        drop_ratio: float | None = None,
        image_metadata: bool = False,
    ) -> tuple[Path, Path]:
        capture = root / "captures" / f"{name}.mov"
        capture.parent.mkdir(exist_ok=True)
        capture.write_bytes(f"video-{name}".encode("utf-8"))
        capture_stat = capture.stat()

        dataset = root / "frames" / name
        images = dataset / "images"
        images.mkdir(parents=True)
        for index in range(count):
            (images / f"{index:06d}.jpg").write_bytes(
                f"{name}-frame-{index}".encode("utf-8")
            )
        metadata = {
            "dataset_type": "single_video",
            "dataset_instance_id": str(uuid.uuid4()),
            "source_video": str(capture.resolve()),
            "source_size_bytes": capture_stat.st_size,
            "source_mtime_ns": capture_stat.st_mtime_ns,
            "source_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
            "images_sha256": _tree_sha256(images),
            "sample_fps": sample_fps,
            "max_frames": 100000,
            "jpeg_quality": 95,
            "source_fps": 30.0,
            "total_source_frames": count * 10,
            "duration_seconds": count / 3,
            "selected_indices": [index * 10 for index in range(count)],
            "selected_timestamps_seconds": [
                index / 3 for index in range(count)
            ],
            "source_frame_width": dimensions[0],
            "source_frame_height": dimensions[1],
            "output_frame_width": dimensions[0],
            "output_frame_height": dimensions[1],
        }
        if sharpness_window is not None:
            metadata["sharpness_window_seconds"] = sharpness_window
            metadata["sharpness_keep_ratio"] = 0.8
            metadata["sharpness_window_frames"] = round(sharpness_window * 30)
            metadata["anchor_indices"] = [index * 10 for index in range(count)]
            metadata["anchor_sharpness"] = [5.0] * count
            metadata["selected_sharpness"] = [9.0] * count
        if drop_ratio is not None:
            metadata["sharpness_drop_ratio"] = drop_ratio
            metadata["sharpness_drop_raw_floor"] = 0.0
        if image_metadata:
            metadata["capture_metadata"] = {"iso": 100, "lens_mm": 13.0, "clip": name}
            metadata["exif_embedded"] = True
            with (dataset / "image_metadata.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["name", "iso", "capture_datetime"])
                for index in range(count):
                    writer.writerow([f"{index:06d}.jpg", 100, f"2026-08-27T12:00:{index:02d}"])
        (dataset / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return dataset, capture

    def test_carries_capture_metadata_and_concatenates_image_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, first_video = self.make_source(root, "JC_01", 2, image_metadata=True)
            second, second_video = self.make_source(root, "JC_02", 3, image_metadata=True)
            plain, _ = self.make_source(root, "plain", 2)
            output = root / "merged"

            merge_frame_datasets([first, second], output)

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [source["capture_metadata"]["clip"] for source in metadata["sources"]],
                ["JC_01", "JC_02"],
            )
            self.assertTrue(all(source["exif_embedded"] for source in metadata["sources"]))
            with (output / "image_metadata.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["name"] for row in rows],
                ["00_jc_01_000000.jpg", "00_jc_01_000001.jpg", "01_jc_02_000000.jpg",
                 "01_jc_02_000001.jpg", "01_jc_02_000002.jpg"],
            )
            self.assertEqual(rows[0]["source_dataset"], "JC_01")
            self.assertEqual(rows[4]["capture_datetime"], "2026-08-27T12:00:02")
            self.assertEqual(list(rows[0]), ["name", "source_dataset", "iso", "capture_datetime"])
            # the merged images directory still validates (CSV lives beside it)
            validate_cache(output, [first_video, second_video], sample_fps=3.0, max_frames=100000)

            with self.assertRaisesRegex(ValueError, "annotate all or none"):
                merge_frame_datasets([first, plain], root / "mixed")
            self.assertFalse((root / "mixed").exists())

    def test_filtered_and_unfiltered_caches_do_not_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plain, _ = self.make_source(root, "plain", 2, sharpness_window=0.3)
            filtered, filtered_video = self.make_source(
                root, "filtered", 3, sharpness_window=0.3, drop_ratio=0.7
            )
            other, other_video = self.make_source(
                root, "other", 2, sharpness_window=0.3, drop_ratio=0.7
            )
            output = root / "merged"

            with self.assertRaisesRegex(ValueError, "one extraction recipe"):
                merge_frame_datasets([plain, filtered], output)
            self.assertFalse(output.exists())

            merge_frame_datasets([filtered, other], output)
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["sharpness_drop_ratio"], 0.7)
            self.assertEqual(metadata["sharpness_drop_raw_floor"], 0.0)
            validate_cache(
                output,
                [filtered_video, other_video],
                sample_fps=3.0,
                max_frames=100000,
                sharpness_drop_ratio=0.7,
            )
            with self.assertRaisesRegex(ValueError, "sharpness_drop_ratio"):
                validate_cache(
                    output,
                    [filtered_video, other_video],
                    sample_fps=3.0,
                    max_frames=100000,
                    sharpness_drop_ratio=0.0,
                )

    def test_sharpness_window_is_part_of_the_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plain, plain_video = self.make_source(root, "plain", 2)
            windowed, windowed_video = self.make_source(
                root, "windowed", 3, sharpness_window=0.3
            )
            output = root / "merged"

            with self.assertRaisesRegex(ValueError, "one extraction recipe"):
                merge_frame_datasets([plain, windowed], output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".merged.*")), [])

            other, other_video = self.make_source(
                root, "other", 2, sharpness_window=0.3
            )
            merge_frame_datasets([windowed, other], output)
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["sharpness_window_seconds"], 0.3)
            self.assertEqual(metadata["sharpness_keep_ratio"], 0.8)
            self.assertEqual(
                [source["sharpness_window_seconds"] for source in metadata["sources"]],
                [0.3, 0.3],
            )
            self.assertEqual(
                [source["sharpness_keep_ratio"] for source in metadata["sources"]],
                [0.8, 0.8],
            )
            self.assertEqual(
                metadata["sources"][0]["anchor_indices"], [0, 10, 20]
            )
            validate_cache(
                output,
                [windowed_video, other_video],
                sample_fps=3.0,
                max_frames=100000,
                sharpness_window=0.3,
                sharpness_keep_ratio=0.8,
            )
            with self.assertRaisesRegex(ValueError, "sharpness_keep_ratio"):
                validate_cache(
                    output,
                    [windowed_video, other_video],
                    sample_fps=3.0,
                    max_frames=100000,
                    sharpness_keep_ratio=0.0,
                )
            with self.assertRaisesRegex(ValueError, "sharpness_window_seconds"):
                validate_cache(
                    output,
                    [windowed_video, other_video],
                    sample_fps=3.0,
                    max_frames=100000,
                    sharpness_window=0.0,
                )

    def test_merges_with_unique_names_metadata_and_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inward, inward_video = self.make_source(root, "Inward Orbit", 2)
            upward, upward_video = self.make_source(root, "upward.v1", 3)
            output = root / "merged"

            merge_frame_datasets([inward, upward], output)

            expected_names = [
                "00_inward_orbit_000000.jpg",
                "00_inward_orbit_000001.jpg",
                "01_upward_v1_000000.jpg",
                "01_upward_v1_000001.jpg",
                "01_upward_v1_000002.jpg",
            ]
            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                expected_names,
            )
            self.assertEqual(
                (inward / "images" / "000000.jpg").stat().st_ino,
                (output / "images" / expected_names[0]).stat().st_ino,
            )
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["dataset_type"], "video_set")
            self.assertEqual(metadata["total_selected_frames"], 5)
            self.assertEqual(metadata["sample_fps"], 3.0)
            self.assertEqual(
                [source["image_prefix"] for source in metadata["sources"]],
                ["00_inward_orbit_", "01_upward_v1_"],
            )
            self.assertEqual(
                metadata["images_sha256"],
                _tree_sha256(output / "images"),
            )
            uuid.UUID(metadata["dataset_instance_id"])
            validate_cache(
                output,
                [inward_video, upward_video],
                sample_fps=3.0,
                max_frames=100000,
            )

    def test_recipe_mismatch_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = self.make_source(root, "first", 2)
            second, _ = self.make_source(
                root,
                "second",
                2,
                sample_fps=6.0,
            )
            output = root / "merged"

            with self.assertRaisesRegex(ValueError, "one extraction recipe"):
                merge_frame_datasets([first, second], output)

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".merged.*")), [])

    def test_changed_frame_is_rejected_without_publishing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = self.make_source(root, "capture", 2)
            (dataset / "images" / "000001.jpg").write_bytes(b"changed")
            output = root / "merged"

            with self.assertRaisesRegex(ValueError, "no longer matches"):
                merge_frame_datasets([dataset], output)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
