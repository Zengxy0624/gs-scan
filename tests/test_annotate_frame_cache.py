from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
import uuid
from fractions import Fraction
from pathlib import Path

from scripts.annotate_frame_cache import (
    annotate_frame_cache,
    build_parser,
    load_annotation_row,
    main,
    parse_aperture,
    parse_capture_metadata,
    parse_iso6709,
    parse_lens_mm,
    parse_shutter,
)
from scripts.extract_frames import _tree_sha256
from scripts.validate_dataset_cache import validate_cache

HAS_DEPS = (
    importlib.util.find_spec("piexif") is not None
    and importlib.util.find_spec("cv2") is not None
)

ANNOTATION = """clip,file,annotation,lens,rec_time,timecode,duration_s,iso,shutter,aperture,white_balance_k,tint,color,gps,notes
1,JC_01_1245.mov,Front: medium camera height,13mm,2026-08-27T12:45:44-0500,12:45:44:00,34.25,86,1/106,f2.2,3490,2,Rec.709 yuv420p,+40.11075-088.22833/,"settling"
12,JC_12_1331.mov,Object: transparent tube,24mm,2026-08-27T13:31:00-0500,13:31:00:00,102.05,100,1/120,f1.8,3490,2,Rec.709 yuv420p,+40.11067-088.22830/,
"""


class ParserTests(unittest.TestCase):
    def test_field_parsers(self) -> None:
        self.assertEqual(parse_lens_mm("13mm"), 13.0)
        self.assertEqual(parse_lens_mm(" 24 mm"), 24.0)
        self.assertIsNone(parse_lens_mm(""))
        self.assertEqual(parse_shutter("1/106"), Fraction(1, 106))
        self.assertEqual(parse_shutter("0.5"), Fraction(1, 2))
        self.assertEqual(parse_aperture("f2.2"), Fraction(11, 5))
        self.assertEqual(parse_aperture("f/1.8"), Fraction(9, 5))
        self.assertEqual(parse_iso6709("+40.11075-088.22833/"), (40.11075, -88.22833))
        self.assertIsNone(parse_iso6709(""))
        for bad in (parse_lens_mm, parse_shutter, parse_aperture, parse_iso6709):
            with self.assertRaises(ValueError):
                bad("nonsense")

    def test_capture_metadata_and_row_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotation.csv"
            path.write_text(ANNOTATION, encoding="utf-8")
            row = load_annotation_row(path, "JC_12_1331.mov")
            self.assertEqual(row["annotation"], "Object: transparent tube")
            capture = parse_capture_metadata(row, "Apple", "iPhone 17 Pro", None)
            self.assertEqual(capture["lens_mm"], 24.0)
            self.assertEqual(capture["iso"], 100)
            self.assertEqual(capture["shutter_rational"], (1, 120))
            self.assertAlmostEqual(capture["aperture_f"], 1.8)
            self.assertEqual(capture["white_balance_k"], 3490)
            self.assertEqual(capture["rec_time_iso"], "2026-08-27T13:31:00-05:00")
            with self.assertRaisesRegex(ValueError, "0 rows"):
                load_annotation_row(path, "missing.mov")


def make_jpeg_cache(root: Path, name: str, count: int, video_name: str) -> tuple[Path, Path]:
    import cv2
    import numpy as np

    capture = root / "captures" / video_name
    capture.parent.mkdir(exist_ok=True)
    capture.write_bytes(f"video-{name}".encode("utf-8"))
    stat = capture.stat()
    dataset = root / "frames" / name
    images = dataset / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for index in range(count):
        ok, data = cv2.imencode(".jpg", rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
        assert ok
        (images / f"{index:06d}.jpg").write_bytes(data.tobytes())
    selected = [index * 20 for index in range(count)]
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
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (dataset / "sharpness_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "raw", "ratio_local"])
        for index in range(count):
            writer.writerow([f"{index:06d}.jpg", 30.0 + index, 1.0])
    return dataset, capture


@unittest.skipUnless(HAS_DEPS, "piexif and OpenCV are installed by scripts/setup_gsplat.sh")
class AnnotateFrameCacheTests(unittest.TestCase):
    def test_embeds_exif_and_writes_per_image_metadata(self) -> None:
        import cv2
        import piexif

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotation.csv"
            annotation.write_text(ANNOTATION, encoding="utf-8")
            cache, capture = make_jpeg_cache(root, "JC_01_1245", 3, "JC_01_1245.mov")
            before = json.loads((cache / "metadata.json").read_text())
            pixels_before = cv2.imread(str(cache / "images" / "000001.jpg"))

            summary = annotate_frame_cache(
                cache, annotation, make="Apple", model="iPhone 17 Pro", software="Blackmagic Cam"
            )

            self.assertEqual(summary["images"], 3)
            exif = piexif.load(str(cache / "images" / "000001.jpg"))
            self.assertEqual(exif["0th"][piexif.ImageIFD.Make], b"Apple")
            self.assertEqual(exif["0th"][piexif.ImageIFD.Model], b"iPhone 17 Pro")
            self.assertEqual(
                exif["0th"][piexif.ImageIFD.ImageDescription], b"Front: medium camera height"
            )
            self.assertEqual(exif["Exif"][piexif.ExifIFD.ISOSpeedRatings], 86)
            self.assertEqual(exif["Exif"][piexif.ExifIFD.ExposureTime], (1, 106))
            self.assertEqual(exif["Exif"][piexif.ExifIFD.FNumber], (11, 5))
            self.assertEqual(exif["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm], 13)
            self.assertEqual(exif["Exif"][piexif.ExifIFD.WhiteBalance], 1)
            # frame 1 sits at source index 20 = 0.333 s after the recording start
            self.assertEqual(
                exif["Exif"][piexif.ExifIFD.DateTimeOriginal], b"2026:08:27 12:45:44"
            )
            self.assertEqual(exif["Exif"][piexif.ExifIFD.SubSecTimeOriginal], b"333")
            comment = json.loads(exif["Exif"][piexif.ExifIFD.UserComment][8:])
            self.assertEqual(comment["source_frame_index"], 20)
            self.assertEqual(comment["white_balance_k"], 3490)
            self.assertEqual(exif["GPS"][piexif.GPSIFD.GPSLatitudeRef], b"N")
            self.assertEqual(exif["GPS"][piexif.GPSIFD.GPSLongitudeRef], b"W")
            self.assertEqual(exif["GPS"][piexif.GPSIFD.GPSLatitude][0], (40, 1))

            # pixels untouched, hashes and metadata refreshed, CSV written
            pixels_after = cv2.imread(str(cache / "images" / "000001.jpg"))
            self.assertTrue((pixels_before == pixels_after).all())
            after = json.loads((cache / "metadata.json").read_text())
            self.assertNotEqual(after["images_sha256"], before["images_sha256"])
            self.assertEqual(after["images_sha256"], _tree_sha256(cache / "images"))
            self.assertNotEqual(after["dataset_instance_id"], before["dataset_instance_id"])
            self.assertTrue(after["exif_embedded"])
            self.assertEqual(after["capture_metadata"]["iso"], 86)
            self.assertEqual(after["capture_metadata"]["annotation_row"]["lens"], "13mm")
            self.assertEqual(after["selected_indices"], before["selected_indices"])
            with (cache / "image_metadata.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["name"] for row in rows], ["000000.jpg", "000001.jpg", "000002.jpg"])
            self.assertEqual(rows[1]["iso"], "86")
            self.assertEqual(rows[1]["capture_datetime"], "2026-08-27T12:45:44.333-05:00")
            self.assertEqual(rows[2]["sharpness_raw"], "32.0")
            self.assertEqual(set(rows[0]), {
                "name", "source_video", "clip", "annotation", "lens_mm", "iso", "shutter",
                "aperture", "white_balance_k", "tint", "color", "capture_datetime",
                "source_frame_index", "timestamp_seconds", "sharpness_raw",
                "sharpness_ratio_local",
            })
            self.assertEqual(sorted(p.name for p in (cache / "images").iterdir()), [
                "000000.jpg", "000001.jpg", "000002.jpg",
            ])
            validate_cache(cache, capture, sample_fps=3.0, max_frames=100000)

            # re-running is idempotent for the image content
            digest = after["images_sha256"]
            annotate_frame_cache(cache, annotation, make="Apple", model="iPhone 17 Pro", software="Blackmagic Cam")
            self.assertEqual(json.loads((cache / "metadata.json").read_text())["images_sha256"], digest)

    def test_camera_identity_falls_back_to_csv_columns(self) -> None:
        import piexif

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotation.csv"
            annotation.write_text(
                "file,lens,iso,make,model,software\nJC_01_1245.mov,13mm,86,Sony,ZV-E10,Firmware 2\n",
                encoding="utf-8",
            )
            cache, _ = make_jpeg_cache(root, "JC_01_1245", 2, "JC_01_1245.mov")
            annotate_frame_cache(cache, annotation, model="Override")
            exif = piexif.load(str(cache / "images" / "000000.jpg"))
            self.assertEqual(exif["0th"][piexif.ImageIFD.Make], b"Sony")
            self.assertEqual(exif["0th"][piexif.ImageIFD.Model], b"Override")
            self.assertEqual(exif["0th"][piexif.ImageIFD.Software], b"Firmware 2")
            self.assertEqual(exif["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm], 13)

    def test_rejects_unknown_video_and_inconsistent_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotation.csv"
            annotation.write_text(ANNOTATION, encoding="utf-8")
            cache, _ = make_jpeg_cache(root, "other", 2, "JC_99_0000.mov")
            with self.assertRaisesRegex(ValueError, "0 rows"):
                annotate_frame_cache(cache, annotation)
            cache2, _ = make_jpeg_cache(root, "JC_01_1245", 2, "JC_01_1245.mov")
            (cache2 / "images" / "000000.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "do not match"):
                annotate_frame_cache(cache2, annotation)
            self.assertEqual(main([str(cache2), "--annotation", str(annotation)]), 1)

    def test_cli_options(self) -> None:
        args = build_parser().parse_args(
            ["cache", "--annotation", "a.csv", "--make", "Apple", "--force"]
        )
        self.assertEqual(args.make, "Apple")
        self.assertTrue(args.force)
        self.assertIsNone(args.model)


if __name__ == "__main__":
    unittest.main()
