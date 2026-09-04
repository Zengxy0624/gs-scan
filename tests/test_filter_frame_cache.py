from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.extract_frames import _tree_sha256
from scripts.filter_frame_cache import (
    build_parser,
    filter_frame_cache,
    main,
    removal_positions,
)
from scripts.merge_frame_datasets import merge_frame_datasets
from scripts.validate_dataset_cache import validate_cache


def make_cache(
    root: Path,
    name: str,
    ratios: list[float],
    *,
    raws: list[float] | None = None,
    window: float = 0.3,
) -> tuple[Path, Path]:
    """A sharpness-window frame cache with one image per ratio value."""
    count = len(ratios)
    raws = raws or [30.0] * count
    capture = root / "captures" / f"{name}.mov"
    capture.parent.mkdir(exist_ok=True)
    capture.write_bytes(f"video-{name}".encode("utf-8"))
    stat = capture.stat()
    dataset = root / "frames" / name
    images = dataset / "images"
    images.mkdir(parents=True)
    for index in range(count):
        (images / f"{index:06d}.jpg").write_bytes(f"{name}-{index}".encode("utf-8"))
    selected = [index * 20 + (3 if index % 2 else 0) for index in range(count)]
    metadata = {
        "dataset_type": "single_video",
        "dataset_instance_id": str(uuid.uuid4()),
        "source_video": str(capture.resolve()),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
        "images_sha256": _tree_sha256(images),
        "sample_fps": 3.0,
        "max_frames": 100000,
        "jpeg_quality": 95,
        "source_fps": 60.0,
        "total_source_frames": count * 20 + 20,
        "duration_seconds": (count * 20 + 19) / 60,
        "selected_indices": selected,
        "selected_timestamps_seconds": [index / 60 for index in selected],
        "source_frame_width": 64,
        "source_frame_height": 48,
        "output_frame_width": 64,
        "output_frame_height": 48,
        "sharpness_window_seconds": window,
        "sharpness_window_frames": 18,
        "sharpness_keep_ratio": 0.8,
        "anchor_indices": [index * 20 for index in range(count)],
        "anchor_sharpness": [raw * 0.9 for raw in raws],
        "selected_sharpness": raws,
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (dataset / "sharpness_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "raw", "ratio_local", "source_index", "anchor_index",
             "shift_frames", "timestamp_seconds", "anchor_raw"]
        )
        for index in range(count):
            writer.writerow(
                [f"{index:06d}.jpg", raws[index], ratios[index], selected[index],
                 index * 20, selected[index] - index * 20, selected[index] / 60,
                 raws[index] * 0.9]
            )
    with (dataset / "sharpness_all_frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_index", "timestamp_seconds", "raw", "is_anchor", "is_selected"])
        for frame in range(count * 20 + 20):
            writer.writerow(
                [frame, frame / 60, 25.0, int(frame % 20 == 0), int(frame in selected)]
            )
    return dataset, capture


class RemovalRuleTests(unittest.TestCase):
    def test_ratio_and_floor_rules(self) -> None:
        rows = [
            {"name": "a", "raw": "30", "ratio_local": "1.0"},
            {"name": "b", "raw": "30", "ratio_local": "0.69"},
            {"name": "c", "raw": "5", "ratio_local": "1.1"},
        ]
        self.assertEqual(removal_positions(rows, 0.7, 0.0), [1])
        self.assertEqual(removal_positions(rows, 0.7, 10.0), [1, 2])
        self.assertEqual(removal_positions(rows, 0.0, 0.0), [])
        rows[0]["ratio_local"] = "nan"
        with self.assertRaises(ValueError):
            removal_positions(rows, 0.7, 0.0)


class FilterFrameCacheTests(unittest.TestCase):
    def test_filters_renumbers_hardlinks_and_rewrites_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ratios = [1.0, 0.5, 1.1, 0.65, 0.9, 1.2]
            source, capture = make_cache(root, "clip", ratios)
            original_metadata = (source / "metadata.json").read_text()
            output = root / "frames" / "clip_sharp070"

            summary = filter_frame_cache(source, output, ratio_threshold=0.7)

            self.assertEqual(summary["removed_frames"], 2)
            self.assertEqual(summary["kept_frames"], 4)
            names = sorted(path.name for path in (output / "images").iterdir())
            self.assertEqual(names, [f"{i:06d}.jpg" for i in range(4)])
            # 000002.jpg of the source (ratio 1.1) becomes 000001.jpg, hard-linked.
            self.assertEqual(
                (output / "images" / "000001.jpg").stat().st_ino,
                (source / "images" / "000002.jpg").stat().st_ino,
            )
            metadata = json.loads((output / "metadata.json").read_text())
            source_meta = json.loads(original_metadata)
            kept = [0, 2, 4, 5]
            for field in (
                "selected_indices", "selected_timestamps_seconds", "anchor_indices",
                "anchor_sharpness", "selected_sharpness",
            ):
                self.assertEqual(metadata[field], [source_meta[field][i] for i in kept])
            self.assertEqual(metadata["images_sha256"], _tree_sha256(output / "images"))
            self.assertNotEqual(metadata["dataset_instance_id"], source_meta["dataset_instance_id"])
            self.assertEqual(metadata["sharpness_drop_ratio"], 0.7)
            self.assertEqual(metadata["sharpness_drop_raw_floor"], 0.0)
            self.assertEqual(metadata["removed_frame_count"], 2)
            self.assertEqual(
                metadata["removed_source_indices"],
                sorted(source_meta["selected_indices"][i] for i in (1, 3)),
            )
            self.assertEqual(metadata["filtered_from"], str(source.resolve()))
            # recipe and identity are carried over unchanged
            for field in ("source_sha256", "sample_fps", "sharpness_window_seconds", "sharpness_keep_ratio"):
                self.assertEqual(metadata[field], source_meta[field])

            with (output / "sharpness_scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["name"] for row in rows], names)
            self.assertEqual(
                [int(row["source_index"]) for row in rows],
                [source_meta["selected_indices"][i] for i in kept],
            )
            with (output / "sharpness_all_frames.csv").open(newline="") as handle:
                all_rows = list(csv.DictReader(handle))
            self.assertEqual(sum(int(row["is_selected"]) for row in all_rows), 4)

            # source untouched; filtered copy validates and merges with its peers
            self.assertEqual((source / "metadata.json").read_text(), original_metadata)
            self.assertEqual(len(list((source / "images").iterdir())), 6)
            validate_cache(
                output, capture, sample_fps=3.0, max_frames=100000,
                sharpness_window=0.3, sharpness_keep_ratio=0.8, sharpness_drop_ratio=0.7,
            )
            other, other_capture = make_cache(root, "other", [1.0, 1.0, 0.2])
            other_filtered = root / "frames" / "other_sharp070"
            filter_frame_cache(other, other_filtered, ratio_threshold=0.7)
            merged = root / "frames" / "merged"
            merge_frame_datasets([output, other_filtered], merged)
            validate_cache(
                merged, [capture, other_capture], sample_fps=3.0, max_frames=100000,
                sharpness_drop_ratio=0.7,
            )
            with self.assertRaisesRegex(ValueError, "one extraction recipe"):
                merge_frame_datasets([output, other], root / "frames" / "mixed")

    def test_refuses_unscored_filtered_or_inconsistent_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = make_cache(root, "clip", [1.0, 1.0, 1.0])
            output = root / "frames" / "out"

            filter_frame_cache(source, output, ratio_threshold=0.7)  # nothing removed is fine
            with self.assertRaisesRegex(ValueError, "already a filtered copy"):
                filter_frame_cache(output, root / "frames" / "twice", 0.7)
            with self.assertRaises(FileExistsError):
                filter_frame_cache(source, output, 0.7)

            unscored, _ = make_cache(root, "plain", [1.0, 1.0], window=0.0)
            with self.assertRaisesRegex(ValueError, "without --sharpness-window"):
                filter_frame_cache(unscored, root / "frames" / "plain_out", 0.7)

            tampered, _ = make_cache(root, "tampered", [1.0, 1.0, 1.0])
            (tampered / "images" / "000001.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "no longer match"):
                filter_frame_cache(tampered, root / "frames" / "tampered_out", 0.7)
            self.assertEqual(list((root / "frames").glob(".tampered_out.*")), [])

            with self.assertRaisesRegex(ValueError, "fewer than two"):
                filter_frame_cache(source, root / "frames" / "empty", ratio_threshold=5.0)

    def test_cli_defaults_and_error_exit(self) -> None:
        args = build_parser().parse_args(["src", "dst"])
        self.assertEqual(args.ratio_threshold, 0.7)
        self.assertEqual(args.raw_floor, 0.0)
        self.assertEqual(args.link_mode, "hardlink")
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                main([str(Path(temporary) / "missing"), str(Path(temporary) / "out")]), 1
            )


if __name__ == "__main__":
    unittest.main()
