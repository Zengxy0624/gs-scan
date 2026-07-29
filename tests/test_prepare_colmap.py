from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_colmap import _validate_input, reconstruct


class InputValidationTests(unittest.TestCase):
    def test_requires_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                _validate_input(dataset, overlap=15)

    def test_requires_two_images_and_positive_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            images = dataset / "images"
            images.mkdir()
            (images / "000000.jpg").write_bytes(b"image")

            with self.assertRaises(ValueError):
                _validate_input(dataset, overlap=15)

            (images / "000001.jpg").write_bytes(b"image")
            with self.assertRaises(ValueError):
                _validate_input(dataset, overlap=0)

    def test_refuses_existing_colmap_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            images = dataset / "images"
            images.mkdir()
            (images / "000000.jpg").write_bytes(b"image")
            (images / "000001.jpg").write_bytes(b"image")

            (dataset / "database.db").write_bytes(b"database")
            with self.assertRaises(FileExistsError):
                _validate_input(dataset, overlap=15)

            (dataset / "database.db").unlink()
            sparse = dataset / "sparse"
            sparse.mkdir()
            (sparse / "partial.bin").write_bytes(b"partial")
            with self.assertRaises(FileExistsError):
                _validate_input(dataset, overlap=15)


class FakeReconstruction:
    def __init__(self, registered: int, points: int) -> None:
        self.registered = registered
        self.points = points
        self.written_to: Path | None = None

    def num_reg_images(self) -> int:
        return self.registered

    def num_points3D(self) -> int:
        return self.points

    def summary(self) -> str:
        return f"{self.registered} registered, {self.points} points"

    def write(self, path: Path) -> None:
        self.written_to = Path(path)


class OrchestrationTests(unittest.TestCase):
    def test_runs_expected_pycolmap_pipeline_and_selects_largest_model(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        smaller = FakeReconstruction(registered=2, points=10)
        larger = FakeReconstruction(registered=5, points=50)

        class ImageReaderOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class SequentialPairingOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        def record(name: str, result: object = None):
            def function(**kwargs: object) -> object:
                calls.append((name, kwargs))
                return result

            return function

        fake_pycolmap = types.SimpleNamespace(
            __version__="test",
            CameraMode=types.SimpleNamespace(SINGLE="single"),
            ImageReaderOptions=ImageReaderOptions,
            SequentialPairingOptions=SequentialPairingOptions,
            extract_features=record("extract_features"),
            match_sequential=record("match_sequential"),
            incremental_mapping=record(
                "incremental_mapping",
                {0: smaller, 1: larger},
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary).resolve()
            images = dataset / "images"
            images.mkdir()
            (images / "000000.jpg").write_bytes(b"image")
            (images / "000001.jpg").write_bytes(b"image")

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                model_path = reconstruct(
                    dataset,
                    overlap=9,
                    camera_model="SIMPLE_RADIAL",
                )

            self.assertEqual(model_path, dataset / "sparse" / "0")
            self.assertEqual(larger.written_to, model_path)
            self.assertIsNone(smaller.written_to)
            self.assertEqual(
                [name for name, _ in calls],
                [
                    "extract_features",
                    "match_sequential",
                    "incremental_mapping",
                ],
            )

            extract_call = calls[0][1]
            self.assertEqual(extract_call["database_path"], dataset / "database.db")
            self.assertEqual(extract_call["image_path"], images)
            self.assertEqual(extract_call["camera_mode"], "single")
            self.assertEqual(
                extract_call["reader_options"].kwargs,
                {"camera_model": "SIMPLE_RADIAL"},
            )

            match_call = calls[1][1]
            self.assertEqual(
                match_call["pairing_options"].kwargs,
                {"overlap": 9},
            )

            mapping_call = calls[2][1]
            self.assertEqual(mapping_call["database_path"], dataset / "database.db")
            self.assertEqual(mapping_call["image_path"], images)
            self.assertEqual(mapping_call["output_path"], dataset / "sparse")


if __name__ == "__main__":
    unittest.main()
