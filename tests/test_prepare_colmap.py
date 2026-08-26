from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_colmap import (
    _minimum_registered_images,
    _validate_input,
    build_parser,
    reconstruct,
)


class InputValidationTests(unittest.TestCase):
    def test_defaults_to_opencv_camera_model(self) -> None:
        args = build_parser().parse_args(["dataset"])
        self.assertEqual(args.camera_model, "OPENCV")
        self.assertEqual(args.matching, "sequential")
        self.assertIsNone(args.min_source_registration_ratio)
        self.assertEqual(args.vocab_tree_num_images, 100)
        self.assertIsNone(args.vocab_tree)
        self.assertFalse(args.guided_matching)
        self.assertFalse(args.sift_affine_shape)
        self.assertFalse(args.sift_domain_size_pooling)
        self.assertIsNone(args.feature_max_image_size)
        self.assertIsNone(args.feature_max_num_features)
        self.assertIsNone(args.init_min_tri_angle)

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

    def test_reuse_matches_requires_and_accepts_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            images = dataset / "images"
            images.mkdir()
            (images / "000000.jpg").write_bytes(b"image")
            (images / "000001.jpg").write_bytes(b"image")

            with self.assertRaises(FileNotFoundError):
                _validate_input(dataset, overlap=15, reuse_matches=True)

            (dataset / "database.db").write_bytes(b"database")
            _validate_input(dataset, overlap=15, reuse_matches=True)

            # Non-empty sparse output is refused even when reusing matches.
            sparse = dataset / "sparse"
            sparse.mkdir()
            (sparse / "partial.bin").write_bytes(b"partial")
            with self.assertRaises(FileExistsError):
                _validate_input(dataset, overlap=15, reuse_matches=True)

    def test_resume_model_requires_reuse_and_complete_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            snapshot = dataset / "snapshots" / "123"
            snapshot.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "--reuse-matches"):
                reconstruct(
                    dataset,
                    overlap=15,
                    camera_model="OPENCV",
                    resume_model=snapshot,
                )

            with self.assertRaisesRegex(FileNotFoundError, "not a complete"):
                reconstruct(
                    dataset,
                    overlap=15,
                    camera_model="OPENCV",
                    reuse_matches=True,
                    resume_model=snapshot,
                )

    def test_rejects_invalid_mapper_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            cases = (
                {"ba_global_frames_ratio": 1.0},
                {"ba_global_points_ratio": 0.9},
                {"ba_global_max_refinements": 0},
                {"ba_global_max_num_iterations": 0},
                {"snapshot_frames_freq": 0},
            )
            for overrides in cases:
                with self.subTest(overrides=overrides):
                    with self.assertRaises(ValueError):
                        reconstruct(
                            dataset,
                            overlap=15,
                            camera_model="OPENCV",
                            **overrides,
                        )

    def test_registration_threshold_uses_ten_image_floor_when_available(self) -> None:
        self.assertEqual(_minimum_registered_images(8, 0.5), 8)
        self.assertEqual(_minimum_registered_images(12, 0.5), 10)
        self.assertEqual(_minimum_registered_images(20, 0.75), 15)
        self.assertEqual(_minimum_registered_images(21, 0.5), 11)

        for invalid_ratio in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(invalid_ratio=invalid_ratio):
                with self.assertRaises(ValueError):
                    _minimum_registered_images(20, invalid_ratio)


class FakeReconstruction:
    def __init__(
        self,
        registered: int,
        points: int,
        registered_names: list[str] | None = None,
        unregistered_names: list[str] | None = None,
    ) -> None:
        self.registered = registered
        self.points = points
        self.written_to: Path | None = None
        registered_names = registered_names or []
        all_names = [*registered_names, *(unregistered_names or [])]
        self.images = {
            index: types.SimpleNamespace(name=name)
            for index, name in enumerate(all_names)
        }
        self._registered_image_ids = list(range(len(registered_names)))

    def num_reg_images(self) -> int:
        return self.registered

    def reg_image_ids(self) -> list[int]:
        return self._registered_image_ids

    def num_points3D(self) -> int:
        return self.points

    def summary(self) -> str:
        return f"{self.registered} registered, {self.points} points"

    def write(self, path: Path) -> None:
        self.written_to = Path(path)
        self.written_to.mkdir(parents=True, exist_ok=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (self.written_to / name).write_bytes(name.encode("ascii"))


class OrchestrationTests(unittest.TestCase):
    def test_runs_expected_pycolmap_pipeline_and_selects_largest_model(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        smaller = FakeReconstruction(registered=2, points=10)
        larger = FakeReconstruction(registered=10, points=50)

        class ImageReaderOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class SequentialPairingOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FeatureExtractionOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FeatureMatchingOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class IncrementalPipelineOptions:
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
            Device=types.SimpleNamespace(cpu="cpu"),
            ImageReaderOptions=ImageReaderOptions,
            SequentialPairingOptions=SequentialPairingOptions,
            FeatureExtractionOptions=FeatureExtractionOptions,
            FeatureMatchingOptions=FeatureMatchingOptions,
            IncrementalPipelineOptions=IncrementalPipelineOptions,
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
            for index in range(12):
                (images / f"{index:06d}.jpg").write_bytes(b"image")

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                model_path = reconstruct(
                    dataset,
                    overlap=9,
                    camera_model="OPENCV_FISHEYE",
                    num_threads=6,
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
                {"camera_model": "OPENCV_FISHEYE"},
            )
            self.assertEqual(extract_call["camera_model"], "OPENCV_FISHEYE")
            self.assertEqual(extract_call["device"], "cpu")
            self.assertEqual(
                extract_call["extraction_options"].kwargs,
                {"num_threads": 6},
            )

            match_call = calls[1][1]
            self.assertEqual(
                match_call["pairing_options"].kwargs,
                {"overlap": 9, "quadratic_overlap": True},
            )
            self.assertEqual(
                match_call["matching_options"].kwargs,
                {"num_threads": 6, "use_gpu": False},
            )
            self.assertEqual(match_call["device"], "cpu")

            mapping_call = calls[2][1]
            self.assertEqual(mapping_call["database_path"], dataset / "database.db")
            self.assertEqual(mapping_call["image_path"], images)
            self.assertEqual(mapping_call["output_path"], dataset / "sparse")
            self.assertEqual(
                mapping_call["options"].kwargs,
                {"num_threads": 6},
            )

            with (dataset / "reconstruction_stats.json").open(
                encoding="utf-8"
            ) as stats_file:
                stats = json.load(stats_file)
            self.assertEqual(
                stats,
                {
                    "camera_mode": "SINGLE",
                    "camera_model": "OPENCV_FISHEYE",
                    "feature_extraction": {
                        "max_image_size": 3200,
                        "max_num_features": 8192,
                        "estimate_affine_shape": False,
                        "domain_size_pooling": False,
                    },
                    "feature_matching": {"guided_matching": False},
                    "matching_strategy": "sequential",
                    "matching_overlap": 9,
                    "matching_quadratic_overlap": True,
                    "minimum_registered_images": 10,
                    "minimum_registration_ratio": 0.5,
                    "mapper": {
                        "init_min_tri_angle": 16.0,
                        "ba_global_frames_ratio": 1.1,
                        "ba_global_points_ratio": 1.1,
                        "ba_global_max_refinements": 5,
                        "ba_global_max_num_iterations": 50,
                    },
                    "ba_use_gpu": False,
                    "reused_matches": False,
                    "resumed_from_model": None,
                    "snapshot_frames_freq": 0,
                    "num_input_images": 12,
                    "num_points3D": 50,
                    "num_registered_images": 10,
                    "registration_ratio": 10 / 12,
                    "selected_model_id": 1,
                    "selected_model_files": {
                        name: {
                            "size_bytes": len(name),
                            "sha256": hashlib.sha256(
                                name.encode("ascii")
                            ).hexdigest(),
                        }
                        for name in (
                            "cameras.bin",
                            "images.bin",
                            "points3D.bin",
                        )
                    },
                    "selected_model_path": "sparse/0",
                    "status": "passed",
                },
            )

    def test_exhaustive_matching_uses_cpu_and_skips_sequential_pairing(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        reconstruction = FakeReconstruction(registered=10, points=50)

        class Options:
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
            Device=types.SimpleNamespace(cpu="cpu"),
            ImageReaderOptions=Options,
            SequentialPairingOptions=Options,
            FeatureExtractionOptions=Options,
            FeatureMatchingOptions=Options,
            IncrementalPipelineOptions=Options,
            extract_features=record("extract_features"),
            match_sequential=record("match_sequential"),
            match_exhaustive=record("match_exhaustive"),
            incremental_mapping=record(
                "incremental_mapping",
                {0: reconstruction},
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary).resolve()
            images = dataset / "images"
            images.mkdir()
            for index in range(12):
                (images / f"{index:06d}.jpg").write_bytes(b"image")

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                reconstruct(
                    dataset,
                    overlap=9,
                    camera_model="OPENCV",
                    num_threads=4,
                    matching_strategy="exhaustive",
                )

            self.assertEqual(
                [name for name, _ in calls],
                ["extract_features", "match_exhaustive", "incremental_mapping"],
            )
            match_call = calls[1][1]
            self.assertEqual(
                match_call["matching_options"].kwargs,
                {"num_threads": 4, "use_gpu": False},
            )
            self.assertEqual(match_call["device"], "cpu")
            stats = json.loads(
                (dataset / "reconstruction_stats.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stats["matching_strategy"], "exhaustive")
            self.assertFalse(stats["matching_quadratic_overlap"])

    def test_vocab_tree_matching_applies_and_records_quality_options(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        reconstruction = FakeReconstruction(registered=10, points=50)

        class BasicOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FeatureExtractionOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.num_threads = int(kwargs["num_threads"])
                self.max_image_size = 3200
                self.sift = types.SimpleNamespace(
                    max_num_features=8192,
                    estimate_affine_shape=False,
                    domain_size_pooling=False,
                )

        class FeatureMatchingOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.num_threads = int(kwargs["num_threads"])
                self.use_gpu = bool(kwargs["use_gpu"])
                self.guided_matching = False

        class VocabTreePairingOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.num_images = int(kwargs["num_images"])
                self.num_threads = int(kwargs["num_threads"])
                self.num_nearest_neighbors = 5
                self.num_checks = 64
                self.num_images_after_verification = 0
                self.max_num_features = -1
                self.vocab_tree_path = str(
                    kwargs.get("vocab_tree_path", "default-vocab-tree-spec")
                )

        class IncrementalPipelineOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.num_threads = int(kwargs["num_threads"])
                self.mapper = types.SimpleNamespace(init_min_tri_angle=16.0)

        def record(name: str, result: object = None):
            def function(**kwargs: object) -> object:
                calls.append((name, kwargs))
                return result

            return function

        fake_pycolmap = types.SimpleNamespace(
            __version__="test",
            CameraMode=types.SimpleNamespace(SINGLE="single"),
            Device=types.SimpleNamespace(cpu="cpu"),
            ImageReaderOptions=BasicOptions,
            FeatureExtractionOptions=FeatureExtractionOptions,
            FeatureMatchingOptions=FeatureMatchingOptions,
            VocabTreePairingOptions=VocabTreePairingOptions,
            IncrementalPipelineOptions=IncrementalPipelineOptions,
            extract_features=record("extract_features"),
            match_vocabtree=record("match_vocabtree"),
            incremental_mapping=record(
                "incremental_mapping",
                {0: reconstruction},
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary).resolve()
            images = dataset / "images"
            images.mkdir()
            for index in range(12):
                (images / f"{index:06d}.jpg").write_bytes(b"image")
            vocab_tree = dataset / "vocab-tree.bin"
            vocab_tree.write_bytes(b"local vocabulary tree")

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                reconstruct(
                    dataset,
                    overlap=9,
                    camera_model="OPENCV",
                    num_threads=5,
                    matching_strategy="vocab_tree",
                    vocab_tree_num_images=200,
                    vocab_tree=vocab_tree,
                    guided_matching=True,
                    sift_affine_shape=True,
                    sift_domain_size_pooling=True,
                    feature_max_image_size=4096,
                    feature_max_num_features=12000,
                    init_min_tri_angle=8.0,
                )

            self.assertEqual(
                [name for name, _ in calls],
                ["extract_features", "match_vocabtree", "incremental_mapping"],
            )
            extraction_options = calls[0][1]["extraction_options"]
            self.assertEqual(extraction_options.kwargs, {"num_threads": 5})
            self.assertEqual(extraction_options.max_image_size, 4096)
            self.assertEqual(extraction_options.sift.max_num_features, 12000)
            self.assertTrue(extraction_options.sift.estimate_affine_shape)
            self.assertTrue(extraction_options.sift.domain_size_pooling)

            match_call = calls[1][1]
            pairing_options = match_call["pairing_options"]
            self.assertEqual(
                pairing_options.kwargs,
                {
                    "num_images": 200,
                    "num_threads": 5,
                    "vocab_tree_path": str(vocab_tree),
                },
            )
            self.assertEqual(pairing_options.max_num_features, 12000)
            self.assertTrue(match_call["matching_options"].guided_matching)
            self.assertEqual(match_call["device"], "cpu")
            mapping_options = calls[2][1]["options"]
            self.assertEqual(mapping_options.mapper.init_min_tri_angle, 8.0)

            stats = json.loads(
                (dataset / "reconstruction_stats.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stats["camera_mode"], "SINGLE")
            self.assertEqual(
                stats["feature_extraction"],
                {
                    "max_image_size": 4096,
                    "max_num_features": 12000,
                    "estimate_affine_shape": True,
                    "domain_size_pooling": True,
                },
            )
            self.assertEqual(
                stats["feature_matching"],
                {"guided_matching": True},
            )
            self.assertEqual(
                stats["mapper"],
                {
                    "init_min_tri_angle": 8.0,
                    "ba_global_frames_ratio": 1.1,
                    "ba_global_points_ratio": 1.1,
                    "ba_global_max_refinements": 5,
                    "ba_global_max_num_iterations": 50,
                },
            )
            self.assertEqual(
                stats["vocab_tree"],
                {
                    "num_images": 200,
                    "num_nearest_neighbors": 5,
                    "num_checks": 64,
                    "num_images_after_verification": 0,
                    "max_num_features": 12000,
                    "path": str(vocab_tree),
                    "sha256": hashlib.sha256(
                        vocab_tree.read_bytes()
                    ).hexdigest(),
                },
            )
            self.assertEqual(stats["matching_strategy"], "vocab_tree")
            self.assertFalse(stats["matching_quadratic_overlap"])

    def test_reuse_matches_skips_extraction_and_matching_and_tunes_mapper(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        reconstruction = FakeReconstruction(registered=10, points=50)

        class Options:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class IncrementalPipelineOptions:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.mapper = types.SimpleNamespace(init_min_tri_angle=16.0)
                self.snapshot_path = ""
                self.snapshot_frames_freq = 0

        def record(name: str, result: object = None):
            def function(**kwargs: object) -> object:
                calls.append((name, kwargs))
                return result

            return function

        fake_pycolmap = types.SimpleNamespace(
            __version__="test",
            CameraMode=types.SimpleNamespace(SINGLE="single"),
            Device=types.SimpleNamespace(cpu="cpu"),
            ImageReaderOptions=Options,
            FeatureExtractionOptions=Options,
            FeatureMatchingOptions=Options,
            SequentialPairingOptions=Options,
            IncrementalPipelineOptions=IncrementalPipelineOptions,
            extract_features=record("extract_features"),
            match_sequential=record("match_sequential"),
            incremental_mapping=record(
                "incremental_mapping",
                {0: reconstruction},
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary).resolve()
            images = dataset / "images"
            images.mkdir()
            for index in range(12):
                (images / f"{index:06d}.jpg").write_bytes(b"image")
            (dataset / "database.db").write_bytes(b"existing matches")
            snapshot = dataset / "snapshots" / "1786700000000"
            snapshot.mkdir(parents=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (snapshot / name).write_bytes(b"model")

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                reconstruct(
                    dataset,
                    overlap=9,
                    camera_model="OPENCV",
                    num_threads=6,
                    reuse_matches=True,
                    ba_global_frames_ratio=1.3,
                    ba_global_points_ratio=1.4,
                    ba_global_max_refinements=2,
                    ba_global_max_num_iterations=40,
                    snapshot_frames_freq=250,
                    resume_model=snapshot,
                )

            self.assertEqual(
                [name for name, _ in calls],
                ["incremental_mapping"],
            )
            self.assertEqual(calls[0][1]["input_path"], str(snapshot))
            mapping_options = calls[0][1]["options"]
            self.assertEqual(mapping_options.ba_global_frames_ratio, 1.3)
            self.assertEqual(mapping_options.ba_global_points_ratio, 1.4)
            self.assertEqual(mapping_options.ba_global_max_refinements, 2)
            self.assertEqual(mapping_options.ba_global_max_num_iterations, 40)
            self.assertEqual(mapping_options.snapshot_frames_freq, 250)
            self.assertEqual(
                mapping_options.snapshot_path,
                str(dataset / "snapshots"),
            )
            self.assertTrue((dataset / "snapshots").is_dir())

            stats = json.loads(
                (dataset / "reconstruction_stats.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(stats["reused_matches"])
            self.assertEqual(stats["resumed_from_model"], str(snapshot))
            self.assertEqual(stats["snapshot_frames_freq"], 250)
            self.assertEqual(
                stats["mapper"],
                {
                    "init_min_tri_angle": 16.0,
                    "ba_global_frames_ratio": 1.3,
                    "ba_global_points_ratio": 1.4,
                    "ba_global_max_refinements": 2,
                    "ba_global_max_num_iterations": 40,
                },
            )
            self.assertEqual(stats["status"], "passed")

    def test_video_set_source_registration_gate_records_pass_and_fail(
        self,
    ) -> None:
        class Options:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        for upward_registered, expected_status in ((8, "passed"), (7, "failed")):
            with (
                self.subTest(
                    upward_registered=upward_registered,
                    expected_status=expected_status,
                ),
                tempfile.TemporaryDirectory() as temporary,
            ):
                dataset = Path(temporary).resolve()
                images = dataset / "images"
                images.mkdir()
                prefixes = ("00_inward_orbit_", "01_upward_")
                for prefix in prefixes:
                    for index in range(10):
                        (images / f"{prefix}{index:06d}.jpg").write_bytes(b"image")
                (dataset / "metadata.json").write_text(
                    json.dumps(
                        {
                            "dataset_type": "video_set",
                            "sources": [
                                {
                                    "image_prefix": prefix,
                                    "selected_indices": list(range(10)),
                                }
                                for prefix in prefixes
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                registered_names = [
                    f"{prefixes[0]}{index:06d}.jpg" for index in range(8)
                ]
                registered_names.extend(
                    f"{prefixes[1]}{index:06d}.jpg"
                    for index in range(upward_registered)
                )
                unregistered_names = [
                    f"{prefixes[0]}{index:06d}.jpg" for index in range(8, 10)
                ]
                unregistered_names.extend(
                    f"{prefixes[1]}{index:06d}.jpg"
                    for index in range(upward_registered, 10)
                )
                reconstruction = FakeReconstruction(
                    registered=len(registered_names),
                    points=50,
                    registered_names=registered_names,
                    unregistered_names=unregistered_names,
                )
                fake_pycolmap = types.SimpleNamespace(
                    __version__="test",
                    CameraMode=types.SimpleNamespace(SINGLE="single"),
                    Device=types.SimpleNamespace(cpu="cpu"),
                    ImageReaderOptions=Options,
                    SequentialPairingOptions=Options,
                    FeatureExtractionOptions=Options,
                    FeatureMatchingOptions=Options,
                    IncrementalPipelineOptions=Options,
                    extract_features=lambda **kwargs: None,
                    match_sequential=lambda **kwargs: None,
                    match_exhaustive=lambda **kwargs: None,
                    incremental_mapping=lambda **kwargs: {0: reconstruction},
                )

                with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                    if expected_status == "passed":
                        reconstruct(
                            dataset,
                            overlap=9,
                            camera_model="OPENCV",
                            matching_strategy="exhaustive",
                            min_source_registration_ratio=0.8,
                        )
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"per-source.*01_upward_: 7/10.*need at least 80.0%",
                        ):
                            reconstruct(
                                dataset,
                                overlap=9,
                                camera_model="OPENCV",
                                matching_strategy="exhaustive",
                                min_source_registration_ratio=0.8,
                            )

                stats = json.loads(
                    (dataset / "reconstruction_stats.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(stats["status"], expected_status)
                self.assertEqual(stats["minimum_source_registration_ratio"], 0.8)
                self.assertEqual(
                    stats["source_registration"],
                    [
                        {
                            "source_index": 0,
                            "image_prefix": "00_inward_orbit_",
                            "num_input_images": 10,
                            "num_registered_images": 8,
                            "registration_ratio": 0.8,
                            "minimum_registration_ratio": 0.8,
                            "status": "passed",
                        },
                        {
                            "source_index": 1,
                            "image_prefix": "01_upward_",
                            "num_input_images": 10,
                            "num_registered_images": upward_registered,
                            "registration_ratio": upward_registered / 10,
                            "minimum_registration_ratio": 0.8,
                            "status": expected_status,
                        },
                    ],
                )
                if expected_status == "passed":
                    self.assertIsNotNone(reconstruction.written_to)
                else:
                    self.assertIsNone(reconstruction.written_to)

    def test_fails_quality_gate_and_records_diagnostics(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        weak = FakeReconstruction(registered=9, points=40)

        class Options:
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
            Device=types.SimpleNamespace(cpu="cpu"),
            ImageReaderOptions=Options,
            SequentialPairingOptions=Options,
            FeatureExtractionOptions=Options,
            FeatureMatchingOptions=Options,
            IncrementalPipelineOptions=Options,
            extract_features=record("extract_features"),
            match_sequential=record("match_sequential"),
            incremental_mapping=record("incremental_mapping", {7: weak}),
        )

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary).resolve()
            images = dataset / "images"
            images.mkdir()
            for index in range(20):
                (images / f"{index:06d}.jpg").write_bytes(b"image")

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"9/20 images registered.*need at least 10",
                ):
                    reconstruct(
                        dataset,
                        overlap=9,
                        camera_model="SIMPLE_RADIAL",
                        num_threads=2,
                    )

            self.assertIsNone(weak.written_to)
            with (dataset / "reconstruction_stats.json").open(
                encoding="utf-8"
            ) as stats_file:
                stats = json.load(stats_file)
            self.assertEqual(stats["status"], "failed")
            self.assertEqual(stats["num_registered_images"], 9)
            self.assertEqual(stats["minimum_registered_images"], 10)

    def test_does_not_publish_passed_stats_when_model_write_fails(self) -> None:
        class FailingWriteReconstruction(FakeReconstruction):
            def write(self, path: Path) -> None:
                raise OSError("disk full")

        class PartialWriteReconstruction(FakeReconstruction):
            def write(self, path: Path) -> None:
                path.mkdir(parents=True, exist_ok=True)
                (path / "cameras.bin").write_bytes(b"partial")

        class Options:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        cases = (
            (
                FailingWriteReconstruction(registered=10, points=50),
                OSError,
                "disk full",
            ),
            (
                PartialWriteReconstruction(registered=10, points=50),
                RuntimeError,
                "not written completely",
            ),
        )
        for reconstruction, error_type, message in cases:
            with (
                self.subTest(error=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                dataset = Path(temporary).resolve()
                images = dataset / "images"
                images.mkdir()
                for index in range(12):
                    (images / f"{index:06d}.jpg").write_bytes(b"image")

                fake_pycolmap = types.SimpleNamespace(
                    __version__="test",
                    CameraMode=types.SimpleNamespace(SINGLE="single"),
                    Device=types.SimpleNamespace(cpu="cpu"),
                    ImageReaderOptions=Options,
                    SequentialPairingOptions=Options,
                    FeatureExtractionOptions=Options,
                    FeatureMatchingOptions=Options,
                    IncrementalPipelineOptions=Options,
                    extract_features=lambda **kwargs: None,
                    match_sequential=lambda **kwargs: None,
                    incremental_mapping=lambda **kwargs: {0: reconstruction},
                )
                with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                    with self.assertRaisesRegex(error_type, message):
                        reconstruct(
                            dataset,
                            overlap=9,
                            camera_model="SIMPLE_RADIAL",
                            num_threads=2,
                        )

                self.assertFalse(
                    (dataset / "reconstruction_stats.json").exists()
                )

    def test_rejects_invalid_thread_count_before_importing_pycolmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            with self.assertRaises(ValueError):
                reconstruct(
                    dataset,
                    overlap=9,
                    camera_model="SIMPLE_RADIAL",
                    num_threads=0,
                )

    def test_rejects_invalid_registration_ratio_before_importing_pycolmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            with self.assertRaises(ValueError):
                reconstruct(
                    dataset,
                    overlap=9,
                    camera_model="SIMPLE_RADIAL",
                    min_registration_ratio=0.0,
                )

    def test_rejects_invalid_quality_options_before_importing_pycolmap(self) -> None:
        cases = (
            ("vocab-tree-num-images", {"vocab_tree_num_images": 0}),
            ("feature-max-image-size", {"feature_max_image_size": 0}),
            ("feature-max-num-features", {"feature_max_num_features": -1}),
            ("init-min-tri-angle", {"init_min_tri_angle": float("nan")}),
            ("init-min-tri-angle", {"init_min_tri_angle": 181.0}),
        )
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            for message, options in cases:
                with self.subTest(message=message, options=options):
                    with (
                        patch.dict(sys.modules, {"pycolmap": None}),
                        self.assertRaisesRegex(ValueError, message),
                    ):
                        reconstruct(
                            dataset,
                            overlap=9,
                            camera_model="SIMPLE_RADIAL",
                            **options,
                        )

            with (
                patch.dict(sys.modules, {"pycolmap": None}),
                self.assertRaisesRegex(FileNotFoundError, "Vocabulary-tree"),
            ):
                reconstruct(
                    dataset,
                    overlap=9,
                    camera_model="SIMPLE_RADIAL",
                    vocab_tree=dataset / "missing-tree.bin",
                )


if __name__ == "__main__":
    unittest.main()
