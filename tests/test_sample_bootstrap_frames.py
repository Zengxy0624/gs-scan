from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.sample_bootstrap_frames import (
    build_parser,
    consecutive_midpoints,
    sample_bootstrap_frames,
)


FRAME_COUNT = 40
PIXEL_STEP = 5
FRAME_SIZE = (64, 48)
SOURCE_FPS = 30.0


def _encode_video(path: Path) -> None:
    """Write a tiny video whose frame index is its flat gray level / step."""
    import numpy as np

    frames = [
        np.full(
            (FRAME_SIZE[1], FRAME_SIZE[0], 3),
            min(index * PIXEL_STEP, 255),
            dtype=np.uint8,
        )
        for index in range(FRAME_COUNT)
    ]
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            SOURCE_FPS,
            FRAME_SIZE,
        )
        if writer.isOpened():
            try:
                for frame in frames:
                    writer.write(frame)
            finally:
                writer.release()
            return
        writer.release()

    import imageio_ffmpeg

    generator = imageio_ffmpeg.write_frames(
        str(path),
        FRAME_SIZE,
        fps=SOURCE_FPS,
        pix_fmt_in="bgr24",
    )
    generator.send(None)
    for frame in frames:
        generator.send(frame.tobytes())
    generator.close()


class SampleBootstrapFramesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._video_home = tempfile.TemporaryDirectory()
        video = Path(cls._video_home.name) / "source.mp4"
        _encode_video(video)
        cls._video_bytes = video.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._video_home.cleanup()

    def make_base_dataset(
        self,
        root: Path,
        sources_spec: dict[str, list[int]],
        *,
        jpeg_quality: int = 92,
    ) -> Path:
        dataset = root / "merged"
        images = dataset / "images"
        images.mkdir(parents=True)
        captures = root / "captures"
        captures.mkdir(exist_ok=True)
        source_records = []
        for index, (stem, selected) in enumerate(sources_spec.items()):
            video = captures / f"{stem}.mp4"
            video.write_bytes(self._video_bytes)
            video_stat = video.stat()
            prefix = f"{index:02d}_{stem}_"
            for position in range(len(selected)):
                (images / f"{prefix}{position:06d}.jpg").write_bytes(b"base")
            source_records.append(
                {
                    "source_video": str(video.resolve()),
                    "source_size_bytes": video_stat.st_size,
                    "source_mtime_ns": video_stat.st_mtime_ns,
                    "source_sha256": hashlib.sha256(
                        self._video_bytes
                    ).hexdigest(),
                    "source_fps": SOURCE_FPS,
                    "total_source_frames": FRAME_COUNT,
                    "duration_seconds": (FRAME_COUNT - 1) / SOURCE_FPS,
                    "selected_indices": list(selected),
                    "selected_timestamps_seconds": [
                        index / SOURCE_FPS for index in selected
                    ],
                    "source_frame_width": FRAME_SIZE[0],
                    "source_frame_height": FRAME_SIZE[1],
                    "output_frame_width": FRAME_SIZE[0],
                    "output_frame_height": FRAME_SIZE[1],
                    "image_prefix": prefix,
                }
            )
        metadata = {
            "dataset_type": "video_set",
            "dataset_instance_id": str(uuid.uuid4()),
            "sources": source_records,
            "total_selected_frames": sum(
                len(selected) for selected in sources_spec.values()
            ),
            "output_frame_width": FRAME_SIZE[0],
            "output_frame_height": FRAME_SIZE[1],
            "images_sha256": "0" * 64,
            "sample_fps": 3.0,
            "max_frames": 100000,
            "jpeg_quality": jpeg_quality,
        }
        (dataset / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return dataset

    def write_metrics_csv(
        self,
        path: Path,
        rows: list[tuple[str, float]],
    ) -> Path:
        lines = ["index,name,source,psnr,ssim,lpips"]
        for index, (name, psnr) in enumerate(rows):
            source = name.rsplit("_", 1)[0] + "_"
            lines.append(f"{index},{name},{source},{psnr},0.5,0.5")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def assert_frame_content(self, image: Path, source_index: int) -> None:
        import cv2

        frame = cv2.imread(str(image))
        self.assertIsNotNone(frame, f"unreadable JPEG: {image}")
        expected = min(source_index * PIXEL_STEP, 255)
        self.assertLess(abs(float(frame.mean()) - expected), 12.0)

    def test_consecutive_midpoints_math(self) -> None:
        self.assertEqual(consecutive_midpoints([0, 10, 20]), [5, 15])
        self.assertEqual(consecutive_midpoints([0, 1, 2, 4, 20]), [3, 12])
        self.assertEqual(consecutive_midpoints([7, 8]), [])
        self.assertEqual(consecutive_midpoints([4]), [])

    def test_all_frames_for_source_extracts_midpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(
                root,
                {"vid_a": [0, 10, 20, 30], "vid_b": [0, 10, 20, 30]},
            )
            output = root / "candidates"

            sample_bootstrap_frames(
                base,
                output,
                all_frames_for_sources=["00_vid_a_"],
            )

            expected_names = [
                "00_vid_a_m000005.jpg",
                "00_vid_a_m000015.jpg",
                "00_vid_a_m000025.jpg",
            ]
            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                expected_names,
            )
            for name, index in zip(expected_names, (5, 15, 25)):
                self.assert_frame_content(output / "images" / name, index)
            self.assertEqual(
                (output / "candidate_names.txt").read_text(encoding="utf-8"),
                "".join(f"{name}\n" for name in expected_names),
            )

            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            base_metadata = json.loads(
                (base / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["dataset_type"], "bootstrap_candidates")
            uuid.UUID(metadata["dataset_instance_id"])
            self.assertEqual(metadata["base_dataset"], str(base.resolve()))
            self.assertEqual(
                metadata["base_dataset_instance_id"],
                base_metadata["dataset_instance_id"],
            )
            self.assertEqual(metadata["jpeg_quality"], 92)
            self.assertEqual(metadata["num_candidates"], 3)
            self.assertEqual(metadata["skipped_name_collisions"], 0)
            self.assertEqual(
                metadata["selection"],
                {
                    "all_frames_for_sources": ["00_vid_a_"],
                    "around_csv": None,
                    "worst": None,
                    "dense_ranges": [],
                    "skipped_unparseable_csv_rows": 0,
                },
            )
            self.assertEqual(
                metadata["candidates"],
                [
                    {
                        "name": name,
                        "image_prefix": "00_vid_a_",
                        "source_frame_index": source_index,
                        "timestamp_seconds": source_index / SOURCE_FPS,
                    }
                    for name, source_index in zip(expected_names, (5, 15, 25))
                ],
            )
            self.assertEqual(metadata["per_source_counts"], {"00_vid_a_": 3})

    def test_around_csv_worst_dedupes_and_handles_boundary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(
                root,
                {"vid_a": [0, 10, 20, 30], "vid_b": [0, 10, 20, 30]},
            )
            metrics = self.write_metrics_csv(
                root / "metrics.csv",
                [
                    ("00_vid_a_000000.jpg", 8.0),
                    ("00_vid_a_000001.jpg", 9.0),
                    ("00_vid_a_000003.jpg", 10.0),
                    ("01_vid_b_000001.jpg", 40.0),
                ],
            )
            output = root / "candidates"

            sample_bootstrap_frames(
                base,
                output,
                around_csv=metrics,
                worst=3,
            )

            # Row at position 0 has only a right midpoint (5), position 1
            # re-selects it (deduped) plus 15, and the last position adds
            # only its left midpoint (25). The high-psnr vid_b row is
            # excluded by --worst 3.
            expected_names = [
                "00_vid_a_m000005.jpg",
                "00_vid_a_m000015.jpg",
                "00_vid_a_m000025.jpg",
            ]
            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                expected_names,
            )
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["num_candidates"], 3)
            self.assertEqual(metadata["per_source_counts"], {"00_vid_a_": 3})
            self.assertEqual(
                metadata["selection"],
                {
                    "all_frames_for_sources": [],
                    "around_csv": str(metrics.resolve()),
                    "worst": 3,
                    "dense_ranges": [],
                    "skipped_unparseable_csv_rows": 0,
                },
            )

    def test_dense_range_selects_unused_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20, 30]})
            output = root / "candidates"

            sample_bootstrap_frames(
                base,
                output,
                dense_ranges=["00_vid_a_:8:12"],
            )

            # Index 10 is already a selected base frame and is excluded.
            expected = {
                "00_vid_a_m000008.jpg": 8,
                "00_vid_a_m000009.jpg": 9,
                "00_vid_a_m000011.jpg": 11,
                "00_vid_a_m000012.jpg": 12,
            }
            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                sorted(expected),
            )
            for name, source_index in expected.items():
                self.assert_frame_content(
                    output / "images" / name,
                    source_index,
                )

            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["num_candidates"], 4)
            self.assertEqual(metadata["per_source_counts"], {"00_vid_a_": 4})
            self.assertEqual(
                metadata["selection"],
                {
                    "all_frames_for_sources": [],
                    "around_csv": None,
                    "worst": None,
                    "dense_ranges": [
                        {
                            "image_prefix": "00_vid_a_",
                            "start": 8,
                            "end": 12,
                        }
                    ],
                    "skipped_unparseable_csv_rows": 0,
                },
            )

    def test_dense_range_dedupes_prior_round_name_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20, 30]})
            # Simulate a frame mined in an earlier bootstrap round that was
            # merged into the base dataset under the m-frame name.
            collision = base / "images" / "00_vid_a_m000009.jpg"
            collision.write_bytes(b"prior-round")
            output = root / "candidates"

            sample_bootstrap_frames(
                base,
                output,
                dense_ranges=["00_vid_a_:8:10"],
            )

            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                ["00_vid_a_m000008.jpg"],
            )
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["num_candidates"], 1)
            self.assertEqual(metadata["skipped_name_collisions"], 1)
            self.assertEqual(collision.read_bytes(), b"prior-round")

    def test_dense_range_overlapping_midpoints_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20, 30]})
            output = root / "candidates"

            sample_bootstrap_frames(
                base,
                output,
                all_frames_for_sources=["00_vid_a_"],
                dense_ranges=["00_vid_a_:4:6"],
            )

            # Midpoints are 5, 15, 25; the dense range adds 4, 5, 6, and
            # the shared index 5 is selected only once.
            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                [
                    "00_vid_a_m000004.jpg",
                    "00_vid_a_m000005.jpg",
                    "00_vid_a_m000006.jpg",
                    "00_vid_a_m000015.jpg",
                    "00_vid_a_m000025.jpg",
                ],
            )
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["num_candidates"], 5)
            self.assertEqual(metadata["skipped_name_collisions"], 0)

    def test_dense_range_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20]})
            cases = (
                ("99_missing_:0:5", "99_missing_"),
                ("00_vid_a_:6:5", "00_vid_a_:6:5"),
                (f"00_vid_a_:5:{FRAME_COUNT}", f"END < {FRAME_COUNT}"),
                ("00_vid_a_:-1:5", "00_vid_a_:-1:5"),
                ("00_vid_a_:5", "'00_vid_a_:5'"),
                ("00_vid_a_:a:b", "'00_vid_a_:a:b'"),
            )

            for index, (argument, pattern) in enumerate(cases):
                output = root / f"candidates-{index}"
                with (
                    self.subTest(argument=argument),
                    self.assertRaisesRegex(ValueError, pattern),
                ):
                    sample_bootstrap_frames(
                        base,
                        output,
                        dense_ranges=[argument],
                    )
                self.assertFalse(output.exists())

    def test_around_csv_skips_prior_round_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(
                root,
                {"vid_a": [0, 10, 20, 30], "vid_b": [0, 10, 20, 30]},
            )
            # The prior-round candidate row has the worst psnr but must not
            # count toward --worst 2: both base rows below it still qualify.
            metrics = self.write_metrics_csv(
                root / "metrics.csv",
                [
                    ("00_vid_a_m000005.jpg", 1.0),
                    ("00_vid_a_000001.jpg", 8.0),
                    ("00_vid_a_000002.jpg", 9.0),
                    ("01_vid_b_000001.jpg", 40.0),
                ],
            )
            output = root / "candidates"

            sample_bootstrap_frames(
                base,
                output,
                around_csv=metrics,
                worst=2,
            )

            # Position 1 contributes midpoints 5 and 15; position 2 (only
            # reached because the m-frame row was skipped, not counted)
            # contributes 15 and 25.
            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                [
                    "00_vid_a_m000005.jpg",
                    "00_vid_a_m000015.jpg",
                    "00_vid_a_m000025.jpg",
                ],
            )
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["num_candidates"], 3)
            self.assertEqual(metadata["per_source_counts"], {"00_vid_a_": 3})
            self.assertEqual(metadata["selection"]["worst"], 2)
            self.assertEqual(
                metadata["selection"]["skipped_unparseable_csv_rows"],
                1,
            )

    def test_no_room_midpoints_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 1, 2, 4]})
            output = root / "candidates"

            sample_bootstrap_frames(
                base,
                output,
                all_frames_for_sources=["00_vid_a_"],
            )

            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                ["00_vid_a_m000003.jpg"],
            )

    def test_name_collision_is_skipped_and_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20, 30]})
            collision = base / "images" / "00_vid_a_m000005.jpg"
            collision.write_bytes(b"already-sampled")
            output = root / "candidates"

            sample_bootstrap_frames(
                base,
                output,
                all_frames_for_sources=["00_vid_a_"],
            )

            names_path = output / "candidate_names.txt"
            self.assertEqual(
                names_path.read_text(encoding="utf-8"),
                "00_vid_a_m000015.jpg\n00_vid_a_m000025.jpg\n",
            )
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["num_candidates"], 2)
            self.assertEqual(metadata["skipped_name_collisions"], 1)
            self.assertEqual(collision.read_bytes(), b"already-sampled")

    def test_decode_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Midpoint 55 is beyond the 40-frame synthetic video, so the
            # decode loop must stop early after staging some frames.
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20, 90]})
            output = root / "candidates"

            with self.assertRaisesRegex(RuntimeError, "stopped early"):
                sample_bootstrap_frames(
                    base,
                    output,
                    all_frames_for_sources=["00_vid_a_"],
                )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".candidates.*")), [])

    def test_rejects_existing_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20]})
            output = root / "candidates"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                sample_bootstrap_frames(
                    base,
                    output,
                    all_frames_for_sources=["00_vid_a_"],
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_rejects_unknown_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20]})
            output = root / "candidates"

            with self.assertRaisesRegex(ValueError, "99_missing_"):
                sample_bootstrap_frames(
                    base,
                    output,
                    all_frames_for_sources=["99_missing_"],
                )

            self.assertFalse(output.exists())

    def test_rejects_incomplete_selection_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20]})
            metrics = self.write_metrics_csv(
                root / "metrics.csv",
                [("00_vid_a_000001.jpg", 8.0)],
            )
            cases = (
                {},
                {"around_csv": metrics},
                {"worst": 2},
            )

            for index, keywords in enumerate(cases):
                output = root / f"candidates-{index}"
                with (
                    self.subTest(keywords=keywords),
                    self.assertRaises(ValueError),
                ):
                    sample_bootstrap_frames(base, output, **keywords)
                self.assertFalse(output.exists())

    def test_rejects_non_video_set_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "single"
            dataset.mkdir()
            (dataset / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_type": "single_video",
                        "dataset_instance_id": str(uuid.uuid4()),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "video_set"):
                sample_bootstrap_frames(
                    dataset,
                    root / "candidates",
                    all_frames_for_sources=["00_vid_a_"],
                )

    def test_rejects_changed_source_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.make_base_dataset(root, {"vid_a": [0, 10, 20]})
            video = root / "captures" / "vid_a.mp4"
            video.write_bytes(self._video_bytes + b"tampered")
            output = root / "candidates"

            with self.assertRaisesRegex(ValueError, "size changed"):
                sample_bootstrap_frames(
                    base,
                    output,
                    all_frames_for_sources=["00_vid_a_"],
                )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".candidates.*")), [])

    def test_cli_parser_collects_repeated_prefixes(self) -> None:
        args = build_parser().parse_args(
            [
                "base",
                "candidates",
                "--all-frames-for-source",
                "00_vid_a_",
                "--all-frames-for-source",
                "01_vid_b_",
                "--around-csv",
                "metrics.csv",
                "--worst",
                "24",
            ]
        )

        self.assertEqual(args.base_dataset, Path("base"))
        self.assertEqual(args.output_dir, Path("candidates"))
        self.assertEqual(
            args.all_frames_for_source,
            ["00_vid_a_", "01_vid_b_"],
        )
        self.assertEqual(args.around_csv, Path("metrics.csv"))
        self.assertEqual(args.worst, 24)
        self.assertIsNone(args.jpeg_quality)
        self.assertEqual(args.dense_range, [])

    def test_cli_parser_collects_repeated_dense_ranges(self) -> None:
        args = build_parser().parse_args(
            [
                "base",
                "candidates",
                "--dense-range",
                "00_a001_08071507_c004_:6270:6430",
                "--dense-range",
                "01_vid_b_:0:5",
            ]
        )

        self.assertEqual(
            args.dense_range,
            ["00_a001_08071507_c004_:6270:6430", "01_vid_b_:0:5"],
        )
        self.assertEqual(args.all_frames_for_source, [])


if __name__ == "__main__":
    unittest.main()
