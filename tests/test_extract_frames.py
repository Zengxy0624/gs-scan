from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from scripts.extract_frames import (
    VideoSample,
    _tree_sha256,
    build_parser,
    create_dataset,
    create_video_set_dataset,
    select_frame_indices,
)


class FrameSelectionTests(unittest.TestCase):
    def test_spans_full_video_and_caps_count(self) -> None:
        indices = select_frame_indices(
            total_frames=9_000,
            source_fps=30.0,
            sample_fps=1.0,
            max_frames=150,
        )

        self.assertEqual(len(indices), 150)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 8_999)
        self.assertEqual(indices, sorted(set(indices)))

    def test_obeys_sampling_rate_for_short_video(self) -> None:
        indices = select_frame_indices(
            total_frames=301,
            source_fps=30.0,
            sample_fps=1.0,
            max_frames=150,
        )

        self.assertEqual(len(indices), 11)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 300)

    def test_rejects_invalid_values(self) -> None:
        cases = (
            (0, 30.0, 1.0, 10),
            (10, 0.0, 1.0, 10),
            (10, 30.0, 0.0, 10),
            (10, 30.0, 1.0, 1),
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                select_frame_indices(*values)


class DatasetCreationTests(unittest.TestCase):
    @staticmethod
    def fake_sample(
        video: Path,
        images_dir: Path,
        sample_fps: float,
        max_frames: int,
        jpeg_quality: int,
        *,
        dimensions: tuple[int, int] = (64, 48),
    ) -> VideoSample:
        del sample_fps, max_frames, jpeg_quality
        count = 2 if video.stem == "inward_orbit" else 3
        images_dir.mkdir(parents=True)
        for index in range(count):
            (images_dir / f"{index:06d}.jpg").write_bytes(
                f"{video.stem}-{index}".encode("utf-8")
            )
        return VideoSample(
            source_fps=30.0,
            total_source_frames=30,
            duration_seconds=29 / 30,
            selected_indices=list(range(count)),
            selected_timestamps_seconds=[index / 30 for index in range(count)],
            source_frame_width=dimensions[0],
            source_frame_height=dimensions[1],
            output_frame_width=dimensions[0],
            output_frame_height=dimensions[1],
        )

    def test_rejects_existing_dataset_without_changing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            video.write_bytes(b"placeholder")
            dataset = root / "dataset"
            dataset.mkdir()
            marker = dataset / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                create_dataset(video, dataset, 3.0, 1000, 95)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_rejects_invalid_cli_values_before_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            video.write_bytes(b"placeholder")
            cases = (
                (math.nan, 1000, 95),
                (0.0, 1000, 95),
                (3.0, 1, 95),
                (3.0, 1000, 0),
                (3.0, 1000, 101),
            )

            for index, (sample_fps, max_frames, quality) in enumerate(cases):
                dataset = root / f"dataset-{index}"
                with (
                    self.subTest(
                        sample_fps=sample_fps,
                        max_frames=max_frames,
                        quality=quality,
                    ),
                    self.assertRaises(ValueError),
                ):
                    create_dataset(
                        video,
                        dataset,
                        sample_fps,
                        max_frames,
                        quality,
                    )
                self.assertFalse(dataset.exists())

    def test_creates_ordered_video_set_with_flat_prefixed_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inward = root / "inward_orbit.mov"
            upward = root / "upward.mov"
            inward.write_bytes(b"inward-video")
            upward.write_bytes(b"upward-video")
            dataset = root / "dataset"

            with mock.patch(
                "scripts.extract_frames.extract_frames",
                side_effect=self.fake_sample,
            ):
                create_video_set_dataset(
                    [inward, upward],
                    dataset,
                    6.0,
                    100000,
                    95,
                )

            images = sorted(path.name for path in (dataset / "images").iterdir())
            self.assertEqual(
                images,
                [
                    "00_inward_orbit_000000.jpg",
                    "00_inward_orbit_000001.jpg",
                    "01_upward_000000.jpg",
                    "01_upward_000001.jpg",
                    "01_upward_000002.jpg",
                ],
            )
            metadata = json.loads(
                (dataset / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["dataset_type"], "video_set")
            self.assertEqual(metadata["sample_fps"], 6.0)
            self.assertEqual(metadata["max_frames"], 100000)
            self.assertEqual(metadata["total_selected_frames"], 5)
            self.assertEqual(
                (metadata["output_frame_width"], metadata["output_frame_height"]),
                (64, 48),
            )
            self.assertEqual(
                [record["source_video"] for record in metadata["sources"]],
                [str(inward.resolve()), str(upward.resolve())],
            )
            self.assertEqual(
                [record["image_prefix"] for record in metadata["sources"]],
                ["00_inward_orbit_", "01_upward_"],
            )
            self.assertEqual(
                [record["selected_indices"] for record in metadata["sources"]],
                [[0, 1], [0, 1, 2]],
            )
            self.assertEqual(
                [record["source_sha256"] for record in metadata["sources"]],
                [
                    hashlib.sha256(inward.read_bytes()).hexdigest(),
                    hashlib.sha256(upward.read_bytes()).hexdigest(),
                ],
            )
            self.assertEqual(
                metadata["images_sha256"],
                _tree_sha256(dataset / "images"),
            )
            uuid.UUID(metadata["dataset_instance_id"])

    def test_video_set_dimension_mismatch_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inward = root / "inward_orbit.mov"
            upward = root / "upward.mov"
            inward.write_bytes(b"inward")
            upward.write_bytes(b"upward")
            dataset = root / "dataset"

            def mismatched_sample(*args: object, **kwargs: object) -> VideoSample:
                video = args[0]
                assert isinstance(video, Path)
                dimensions = (
                    (64, 48)
                    if video.name == "inward_orbit.mov"
                    else (48, 64)
                )
                return self.fake_sample(*args, **kwargs, dimensions=dimensions)

            with (
                mock.patch(
                    "scripts.extract_frames.extract_frames",
                    side_effect=mismatched_sample,
                ),
                self.assertRaisesRegex(ValueError, "same output dimensions"),
            ):
                create_video_set_dataset(
                    [inward, upward],
                    dataset,
                    6.0,
                    100000,
                    95,
                )

            self.assertFalse(dataset.exists())
            self.assertEqual(list(root.glob(".dataset.*")), [])

    def test_cli_preserves_primary_video_and_additional_source_order(self) -> None:
        args = build_parser().parse_args(
            [
                "inward_orbit.mov",
                "dataset",
                "--additional-video",
                "upward.mov",
                "--additional-video",
                "ceiling.mov",
            ]
        )

        self.assertEqual(args.video, Path("inward_orbit.mov"))
        self.assertEqual(
            args.additional_video,
            [Path("upward.mov"), Path("ceiling.mov")],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("cv2") is not None,
        "OpenCV is installed by scripts/setup_gsplat.sh",
    )
    def test_extracts_real_video_into_colmap_layout(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            try:
                for index in range(30):
                    frame = np.zeros((48, 64, 3), dtype=np.uint8)
                    frame[:, :, 0] = index * 7
                    frame[:, :, 1] = np.arange(64, dtype=np.uint8)
                    frame[:, :, 2] = np.arange(48, dtype=np.uint8)[:, None]
                    writer.write(frame)
            finally:
                writer.release()

            dataset = root / "dataset"
            create_dataset(video, dataset, 10.0, 5, 95)

            images = sorted((dataset / "images").glob("*.jpg"))
            metadata = json.loads(
                (dataset / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual([path.name for path in images], [
                "000000.jpg",
                "000001.jpg",
                "000002.jpg",
                "000003.jpg",
                "000004.jpg",
            ])
            self.assertTrue(all(path.stat().st_size > 0 for path in images))
            self.assertEqual(metadata["selected_indices"], [0, 7, 14, 22, 29])
            self.assertEqual(metadata["dataset_type"], "single_video")
            self.assertEqual(metadata["source_video"], str(video.resolve()))
            self.assertEqual(
                metadata["source_size_bytes"],
                video.stat().st_size,
            )
            self.assertEqual(
                metadata["source_mtime_ns"],
                video.stat().st_mtime_ns,
            )
            self.assertEqual(
                metadata["source_sha256"],
                hashlib.sha256(video.read_bytes()).hexdigest(),
            )
            uuid.UUID(metadata["dataset_instance_id"])
            self.assertEqual(
                metadata["images_sha256"],
                _tree_sha256(dataset / "images"),
            )
            self.assertEqual(
                (
                    metadata["output_frame_width"],
                    metadata["output_frame_height"],
                ),
                (64, 48),
            )


if __name__ == "__main__":
    unittest.main()
