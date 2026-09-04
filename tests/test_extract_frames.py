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
    local_sharpness_ratios,
    select_frame_indices,
    select_sharp_indices,
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


class SharpSelectionTests(unittest.TestCase):
    def test_zero_window_returns_anchors(self) -> None:
        anchors = [0, 4, 8, 11]
        self.assertEqual(
            select_sharp_indices(anchors, [1.0] * 12, 0, keep_ratio=1.0),
            anchors,
        )

    def test_picks_sharpest_neighbor_within_window(self) -> None:
        sharpness = [0.0] * 30
        sharpness[3] = 9.0  # sharper neighbor of anchor 0 (in window)
        sharpness[19] = 9.0  # sharper neighbor of anchor 20 (in window)
        sharpness[27] = 9.0  # outside anchor 20's window; inside anchor 29's

        chosen = select_sharp_indices([0, 10, 20, 29], sharpness, 3, keep_ratio=1.0)

        self.assertEqual(chosen, [3, 10, 19, 27])

    def test_overlapping_windows_stay_unique_and_increasing(self) -> None:
        sharpness = [0.0] * 11
        sharpness[4] = 9.0
        sharpness[6] = 9.0

        chosen = select_sharp_indices([0, 5, 10], sharpness, 4, keep_ratio=1.0)

        self.assertEqual(chosen, [4, 6, 10])
        self.assertEqual(chosen, sorted(set(chosen)))

    def test_reserves_frames_for_later_anchors(self) -> None:
        # Anchors packed at the end: the first pick may not consume frames the
        # remaining anchors need, so every anchor still yields one image.
        chosen = select_sharp_indices([0, 2, 3], [1.0, 5.0, 9.0, 9.0], 5, keep_ratio=1.0)

        self.assertEqual(chosen, [1, 2, 3])

    def test_ties_prefer_closest_then_earliest_frame(self) -> None:
        flat = [1.0] * 9
        self.assertEqual(select_sharp_indices([4], flat, 3), [4])
        peaked = [1.0] * 9
        peaked[1] = peaked[7] = 5.0  # equidistant from anchor 4
        self.assertEqual(select_sharp_indices([4], peaked, 3, keep_ratio=1.0), [1])

    def test_keep_ratio_retains_anchors_that_are_sharp_enough(self) -> None:
        sharpness = [0.0] * 30
        sharpness[10] = 9.0  # anchor 10 is the window maximum: kept
        sharpness[20] = 8.0  # anchor 20 at 80% of neighbor 22: kept at 0.8
        sharpness[22] = 10.0
        sharpness[0] = 1.0  # anchor 0 far below neighbor 2: replaced
        sharpness[2] = 10.0

        self.assertEqual(
            select_sharp_indices([0, 10, 20], sharpness, 3, keep_ratio=0.8),
            [2, 10, 20],
        )
        self.assertEqual(
            select_sharp_indices([0, 10, 20], sharpness, 3, keep_ratio=0.9),
            [2, 10, 22],
        )
        # An anchor already passed by the previous pick cannot be kept.
        self.assertEqual(
            select_sharp_indices([0, 3], [0.0, 0.0, 0.0, 5.0, 9.0, 0.0], 3, 0.8),
            [3, 4],
        )
        # 0 disables replacement entirely; 1 always takes the sharpest frame.
        self.assertEqual(
            select_sharp_indices([0, 10, 20], sharpness, 3, keep_ratio=0.0),
            [0, 10, 20],
        )
        self.assertEqual(
            select_sharp_indices([0, 10, 20], sharpness, 3, keep_ratio=1.0),
            [2, 10, 22],
        )

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            select_sharp_indices([0, 5], [1.0] * 10, 1, keep_ratio=1.5)
        with self.assertRaises(ValueError):
            select_sharp_indices([0, 5], [1.0] * 10, -1)
        with self.assertRaises(ValueError):
            select_sharp_indices([], [1.0] * 10, 1)
        with self.assertRaises(ValueError):
            select_sharp_indices([5, 0], [1.0] * 10, 1)
        with self.assertRaises(ValueError):
            select_sharp_indices([0, 10], [1.0] * 10, 1)

    def test_local_ratio_uses_rolling_median(self) -> None:
        raw = [10.0, 10.0, 2.0, 10.0, 10.0]
        ratios = local_sharpness_ratios(raw, neighbors=1)
        self.assertAlmostEqual(ratios[2], 0.2)
        self.assertAlmostEqual(ratios[0], 1.0)
        self.assertEqual(local_sharpness_ratios([0.0, 0.0]), [1.0, 1.0])


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
        **extraction_options: object,
    ) -> VideoSample:
        del sample_fps, max_frames, jpeg_quality, extraction_options
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

    def test_cli_parses_sharpness_options_with_safe_defaults(self) -> None:
        defaults = build_parser().parse_args(["in.mov", "dataset"])
        self.assertEqual(defaults.sharpness_window, 0.0)
        self.assertGreaterEqual(defaults.sharpness_workers, 1)
        args = build_parser().parse_args(
            ["in.mov", "dataset", "--sharpness-window", "0.3"]
        )
        self.assertEqual(args.sharpness_window, 0.3)
        self.assertEqual(args.sharpness_keep_ratio, 0.8)
        args = build_parser().parse_args(
            ["in.mov", "dataset", "--sharpness-window", "0.3", "--sharpness-keep-ratio", "1"]
        )
        self.assertEqual(args.sharpness_keep_ratio, 1.0)

    def test_rejects_negative_sharpness_window_before_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            video.write_bytes(b"placeholder")
            dataset = root / "dataset"
            with self.assertRaises(ValueError):
                create_dataset(video, dataset, 3.0, 1000, 95, sharpness_window=-0.1)
            with self.assertRaises(ValueError):
                create_dataset(video, dataset, 3.0, 1000, 95, sharpness_workers=0)
            with self.assertRaises(ValueError):
                create_dataset(
                    video, dataset, 3.0, 1000, 95, sharpness_keep_ratio=1.5
                )
            self.assertFalse(dataset.exists())

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

    @unittest.skipUnless(
        importlib.util.find_spec("cv2") is not None,
        "OpenCV is installed by scripts/setup_gsplat.sh",
    )
    def test_sharpness_window_swaps_blurred_anchors_for_sharp_neighbors(self) -> None:
        import csv

        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.avi"
            fps = 10.0
            total = 60
            # Anchors for sample_fps=2 span the clip at ~5-frame spacing.
            anchors = select_frame_indices(total, fps, 2.0, 100)
            rng = np.random.default_rng(0)
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (128, 96),
            )
            self.assertTrue(writer.isOpened())
            try:
                for index in range(total):
                    frame = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
                    if index in anchors:
                        frame = cv2.GaussianBlur(frame, (0, 0), 6.0)
                    writer.write(frame)
            finally:
                writer.release()

            dataset = root / "dataset"
            create_dataset(video, dataset, 2.0, 100, 95, sharpness_window=0.2)

            metadata = json.loads(
                (dataset / "metadata.json").read_text(encoding="utf-8")
            )
            selected = metadata["selected_indices"]
            self.assertEqual(metadata["anchor_indices"], anchors)
            self.assertEqual(metadata["sharpness_window_seconds"], 0.2)
            self.assertEqual(metadata["sharpness_window_frames"], 2)
            self.assertEqual(metadata["sharpness_keep_ratio"], 0.8)
            self.assertEqual(len(selected), len(anchors))
            self.assertEqual(selected, sorted(set(selected)))
            # Every blurred anchor has a sharp neighbor within two frames, so
            # no anchor survives and every pick is sharper than its anchor.
            self.assertTrue(
                all(chosen != anchor for chosen, anchor in zip(selected, anchors))
            )
            self.assertTrue(
                all(abs(chosen - anchor) <= 2 for chosen, anchor in zip(selected, anchors))
            )
            self.assertTrue(
                all(
                    picked > anchored
                    for picked, anchored in zip(
                        metadata["selected_sharpness"],
                        metadata["anchor_sharpness"],
                    )
                )
            )
            self.assertEqual(
                sorted(path.name for path in (dataset / "images").glob("*.jpg")),
                [f"{index:06d}.jpg" for index in range(len(anchors))],
            )
            self.assertEqual(
                metadata["images_sha256"],
                _tree_sha256(dataset / "images"),
            )

            with (dataset / "sharpness_scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(anchors))
            self.assertLessEqual({"name", "raw", "ratio_local"}, set(rows[0]))
            self.assertEqual(rows[0]["name"], "000000.jpg")
            self.assertEqual(int(rows[0]["source_index"]), selected[0])
            with (dataset / "sharpness_all_frames.csv").open(newline="") as handle:
                all_rows = list(csv.DictReader(handle))
            self.assertEqual(len(all_rows), total)
            self.assertEqual(
                sum(int(row["is_selected"]) for row in all_rows),
                len(anchors),
            )

    def test_zero_window_metadata_omits_sharpness_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "clip.mov"
            video.write_bytes(b"clip")
            dataset = root / "dataset"
            with mock.patch(
                "scripts.extract_frames.extract_frames",
                side_effect=self.fake_sample,
            ):
                create_dataset(video, dataset, 3.0, 100000, 95)
            metadata = json.loads(
                (dataset / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["sharpness_window_seconds"], 0.0)
            self.assertEqual(metadata["sharpness_keep_ratio"], 0.0)
            self.assertNotIn("anchor_indices", metadata)
            self.assertNotIn("selected_sharpness", metadata)


if __name__ == "__main__":
    unittest.main()
