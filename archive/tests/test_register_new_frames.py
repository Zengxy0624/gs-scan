from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from archive.scripts.register_new_frames import (
    MATCHING_OMP_THREADS,
    OPENBLAS_SHIM_BASENAME,
    build_parser,
    register_new_frames,
)


SHIM_ENVIRONMENT = {
    "LD_PRELOAD": f"/fake/{OPENBLAS_SHIM_BASENAME}",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": str(MATCHING_OMP_THREADS),
}


BASE_IMAGE_NAMES = tuple(f"00_vid_{index:06d}.jpg" for index in range(4))
CANDIDATE_NAMES = ("00_vid_m000100.jpg", "00_vid_m000200.jpg")


def _base_stats() -> dict[str, object]:
    return {
        "status": "passed",
        "camera_model": "OPENCV",
        "feature_extraction": {
            "max_image_size": 3840,
            "max_num_features": 8192,
            "estimate_affine_shape": True,
            "domain_size_pooling": True,
        },
        "mapper": {
            "init_min_tri_angle": 8.0,
            "ba_global_frames_ratio": 1.4,
            "ba_global_points_ratio": 1.4,
            "ba_global_max_refinements": 1,
            "ba_global_max_num_iterations": 30,
        },
        "num_registered_images": 4,
        "num_points3D": 100,
        "selected_model_files": {
            "cameras.bin": {"size_bytes": 5, "sha256": "0" * 64},
        },
    }


def _make_base(root: Path, stats: dict[str, object] | None = None) -> Path:
    base = root / "base"
    images = base / "images"
    images.mkdir(parents=True)
    for name in BASE_IMAGE_NAMES:
        (images / name).write_bytes(b"base image " + name.encode("ascii"))
    (base / "database.db").write_bytes(b"base colmap database")
    model = base / "sparse" / "0"
    model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(b"base " + name.encode("ascii"))
    (base / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_type": "video_set",
                "dataset_instance_id": "base-instance",
                "sources": [{"image_prefix": "00_vid_"}],
            }
        ),
        encoding="utf-8",
    )
    (base / "reconstruction_stats.json").write_text(
        json.dumps(stats if stats is not None else _base_stats()),
        encoding="utf-8",
    )
    return base


def _make_candidates(
    root: Path,
    names: tuple[str, ...] = CANDIDATE_NAMES,
    listed_names: tuple[str, ...] | None = None,
) -> Path:
    candidates = root / "candidates"
    images = candidates / "images"
    images.mkdir(parents=True)
    for name in names:
        (images / name).write_bytes(b"candidate " + name.encode("ascii"))
    (candidates / "candidate_names.txt").write_text(
        "\n".join(listed_names if listed_names is not None else names) + "\n",
        encoding="utf-8",
    )
    (candidates / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_type": "bootstrap_candidates",
                "dataset_instance_id": "cand-instance",
            }
        ),
        encoding="utf-8",
    )
    return candidates


def _make_vocab_tree(root: Path) -> Path:
    vocab_tree = root / "vocab-tree.bin"
    vocab_tree.write_bytes(b"vocabulary tree")
    return vocab_tree


def _tree_snapshot(root: Path) -> list[tuple[str, int, int]]:
    return sorted(
        (
            str(path.relative_to(root)),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in root.rglob("*")
        if path.is_file()
    )


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


def _fake_pycolmap(
    calls: list[tuple[str, dict[str, object]]],
    mapping_result: dict[int, FakeReconstruction],
    base_camera_ids: tuple[int, ...] = (3,),
    db_num_cameras: int = 1,
) -> types.SimpleNamespace:
    class ImageReaderOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.existing_camera_id = -1

    class FeatureExtractionOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.max_image_size = 3200
            self.sift = types.SimpleNamespace(
                max_num_features=8192,
                estimate_affine_shape=False,
                domain_size_pooling=False,
            )

    class FeatureMatchingOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.guided_matching = False

    class VocabTreePairingOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.match_list_path = ""
            self.max_num_features = -1

    class IncrementalPipelineOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class Reconstruction:
        def __init__(self, path: str) -> None:
            calls.append(("load_base_model", {"path": path}))
            self.cameras = {
                camera_id: object() for camera_id in base_camera_ids
            }

    class Database:
        @staticmethod
        def open(path: str) -> "Database":
            calls.append(("database_open", {"path": path}))
            return Database()

        def num_cameras(self) -> int:
            return db_num_cameras

        def close(self) -> None:
            calls.append(("database_close", {}))

    def record(name: str, result: object = None):
        def function(**kwargs: object) -> object:
            calls.append((name, kwargs))
            return result

        return function

    return types.SimpleNamespace(
        __version__="test",
        CameraMode=types.SimpleNamespace(SINGLE="single"),
        Device=types.SimpleNamespace(cpu="cpu"),
        ImageReaderOptions=ImageReaderOptions,
        FeatureExtractionOptions=FeatureExtractionOptions,
        FeatureMatchingOptions=FeatureMatchingOptions,
        VocabTreePairingOptions=VocabTreePairingOptions,
        IncrementalPipelineOptions=IncrementalPipelineOptions,
        Reconstruction=Reconstruction,
        Database=Database,
        extract_features=record("extract_features"),
        match_vocabtree=record("match_vocabtree"),
        incremental_mapping=record("incremental_mapping", mapping_result),
    )


class ParserTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = build_parser().parse_args(
            [
                "--base",
                "base",
                "--candidates",
                "cand",
                "--output",
                "out",
                "--vocab-tree",
                "tree.bin",
            ]
        )
        self.assertEqual(args.vocab_tree_num_images, 50)
        self.assertFalse(args.guided_matching)
        self.assertEqual(args.num_threads, -1)
        self.assertFalse(args.ba_use_gpu)
        self.assertEqual(args.ba_gpu_index, "0")
        self.assertEqual(args.mapping_blas_threads, 48)
        self.assertEqual(args.min_candidate_registration_ratio, 0.8)
        self.assertIsNone(args.ba_global_frames_ratio)
        self.assertIsNone(args.ba_global_points_ratio)
        self.assertIsNone(args.ba_global_max_refinements)
        self.assertIsNone(args.ba_global_max_num_iterations)


class ValidationTests(unittest.TestCase):
    def _call(self, root: Path, base: Path, candidates: Path, **kwargs) -> Path:
        with patch.dict(sys.modules, {"pycolmap": None}):
            return register_new_frames(
                base,
                candidates,
                kwargs.pop("output", root / "out"),
                kwargs.pop("vocab_tree", _make_vocab_tree(root)),
                **kwargs,
            )

    def test_rejects_missing_base_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            (base / "reconstruction_stats.json").unlink()
            candidates = _make_candidates(root)
            with self.assertRaisesRegex(FileNotFoundError, "stats"):
                self._call(root, base, candidates)

    def test_rejects_failed_base_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stats = _base_stats()
            stats["status"] = "failed"
            base = _make_base(root, stats=stats)
            candidates = _make_candidates(root)
            with self.assertRaisesRegex(ValueError, "quality gates"):
                self._call(root, base, candidates)

    def test_rejects_incomplete_base_model_and_empty_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            (base / "sparse" / "0" / "points3D.bin").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "points3D.bin"):
                self._call(root, base, candidates)

            for name in ("points3D.bin",):
                (base / "sparse" / "0" / name).write_bytes(b"model")
            (base / "database.db").write_bytes(b"")
            with self.assertRaisesRegex(FileNotFoundError, "database"):
                self._call(root, base, candidates)

    def test_rejects_candidate_name_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(
                root,
                names=(BASE_IMAGE_NAMES[0], CANDIDATE_NAMES[0]),
            )
            with self.assertRaisesRegex(ValueError, "collides"):
                self._call(root, base, candidates)

    def test_rejects_missing_candidate_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(
                root,
                names=CANDIDATE_NAMES,
                listed_names=(*CANDIDATE_NAMES, "00_vid_m000300.jpg"),
            )
            with self.assertRaisesRegex(FileNotFoundError, "m000300"):
                self._call(root, base, candidates)

    def test_rejects_wrong_candidate_dataset_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            (candidates / "metadata.json").write_text(
                json.dumps({"dataset_type": "single_video"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "bootstrap_candidates"):
                self._call(root, base, candidates)

    def test_rejects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            output = root / "out"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                self._call(root, base, candidates, output=output)

    def test_rejects_missing_vocab_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            with self.assertRaisesRegex(FileNotFoundError, "Vocabulary-tree"):
                self._call(
                    root,
                    base,
                    candidates,
                    vocab_tree=root / "missing-tree.bin",
                )

    def test_rejects_invalid_options_before_importing_pycolmap(self) -> None:
        cases = (
            ("--num-threads", {"num_threads": 0}),
            ("--vocab-tree-num-images", {"vocab_tree_num_images": 0}),
            ("--mapping-blas-threads", {"mapping_blas_threads": -1}),
            (
                "--min-candidate-registration-ratio",
                {"min_candidate_registration_ratio": 0.0},
            ),
            ("--ba-global-frames-ratio", {"ba_global_frames_ratio": 1.0}),
            ("--ba-global-points-ratio", {"ba_global_points_ratio": 0.9}),
            ("--ba-global-max-refinements", {"ba_global_max_refinements": 0}),
            (
                "--ba-global-max-num-iterations",
                {"ba_global_max_num_iterations": 0},
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            for message, options in cases:
                with self.subTest(options=options):
                    with self.assertRaisesRegex(ValueError, message):
                        self._call(root, base, candidates, **options)


class ThreadEnvironmentTests(unittest.TestCase):
    def _assert_rejected_before_staging(
        self,
        environment: dict[str, str],
        message_pattern: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            vocab_tree = _make_vocab_tree(root)
            output = root / "augmented"

            preserved = {
                key: value
                for key, value in os.environ.items()
                if key
                not in ("LD_PRELOAD", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            }
            with patch.dict(
                os.environ, {**preserved, **environment}, clear=True
            ):
                with self.assertRaisesRegex(RuntimeError, message_pattern):
                    register_new_frames(base, candidates, output, vocab_tree)

            self.assertFalse(output.exists())
            self.assertFalse(
                any(path.name.startswith(".augmented") for path in root.iterdir())
            )

    def test_missing_shim_fails_before_any_staging(self) -> None:
        environment = dict(SHIM_ENVIRONMENT)
        del environment["LD_PRELOAD"]
        self._assert_rejected_before_staging(environment, "LD_PRELOAD")

    def test_missing_openblas_pin_fails_before_any_staging(self) -> None:
        environment = dict(SHIM_ENVIRONMENT)
        del environment["OPENBLAS_NUM_THREADS"]
        self._assert_rejected_before_staging(
            environment, "OPENBLAS_NUM_THREADS=1"
        )

    def test_multithreaded_openblas_fails_before_any_staging(self) -> None:
        environment = {**SHIM_ENVIRONMENT, "OPENBLAS_NUM_THREADS": "64"}
        self._assert_rejected_before_staging(
            environment, "OPENBLAS_NUM_THREADS=1"
        )

    def test_missing_omp_cap_fails_before_any_staging(self) -> None:
        environment = dict(SHIM_ENVIRONMENT)
        del environment["OMP_NUM_THREADS"]
        self._assert_rejected_before_staging(environment, "OMP_NUM_THREADS")

    def test_oversized_omp_cap_fails_before_any_staging(self) -> None:
        environment = {**SHIM_ENVIRONMENT, "OMP_NUM_THREADS": "64"}
        self._assert_rejected_before_staging(environment, "OMP_NUM_THREADS")


class OrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        environment = patch.dict(os.environ, SHIM_ENVIRONMENT)
        environment.start()
        self.addCleanup(environment.stop)

    def test_success_adopts_base_options_and_registers_candidates(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        reconstruction = FakeReconstruction(
            registered=6,
            points=500,
            registered_names=[*BASE_IMAGE_NAMES, *CANDIDATE_NAMES],
        )
        fake_pycolmap = _fake_pycolmap(calls, {0: reconstruction})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            vocab_tree = _make_vocab_tree(root)
            output = root / "augmented"
            base_before = _tree_snapshot(base)
            candidates_before = _tree_snapshot(candidates)

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                model_path = register_new_frames(
                    base,
                    candidates,
                    output,
                    vocab_tree,
                    vocab_tree_num_images=40,
                    guided_matching=True,
                    num_threads=4,
                    ba_global_max_refinements=3,
                )

            self.assertEqual(model_path, output / "sparse" / "0")
            self.assertEqual(reconstruction.written_to, model_path)
            self.assertEqual(
                [name for name, _ in calls],
                [
                    "load_base_model",
                    "extract_features",
                    "database_open",
                    "database_close",
                    "match_vocabtree",
                    "incremental_mapping",
                ],
            )

            self.assertEqual(
                calls[0][1]["path"], str(output / "resume_model")
            )

            extract_call = calls[1][1]
            self.assertEqual(
                extract_call["database_path"], output / "database.db"
            )
            self.assertEqual(extract_call["image_path"], output / "images")
            self.assertEqual(extract_call["camera_mode"], "single")
            self.assertEqual(extract_call["camera_model"], "OPENCV")
            self.assertEqual(extract_call["device"], "cpu")
            reader_options = extract_call["reader_options"]
            self.assertEqual(reader_options.kwargs, {"camera_model": "OPENCV"})
            self.assertEqual(reader_options.existing_camera_id, 3)
            extraction_options = extract_call["extraction_options"]
            self.assertEqual(extraction_options.kwargs, {"num_threads": 4})
            self.assertEqual(extraction_options.max_image_size, 3840)
            self.assertEqual(extraction_options.sift.max_num_features, 8192)
            self.assertTrue(extraction_options.sift.estimate_affine_shape)
            self.assertTrue(extraction_options.sift.domain_size_pooling)

            self.assertEqual(
                calls[2][1]["path"], str(output / "database.db")
            )

            match_call = calls[4][1]
            pairing_options = match_call["pairing_options"]
            self.assertEqual(
                pairing_options.kwargs,
                {
                    "num_images": 40,
                    "num_threads": 4,
                    "vocab_tree_path": str(vocab_tree),
                },
            )
            match_list_path = output / "bootstrap_match_list.txt"
            self.assertEqual(
                pairing_options.match_list_path, str(match_list_path)
            )
            self.assertEqual(pairing_options.max_num_features, 8192)
            self.assertEqual(
                match_list_path.read_text(encoding="utf-8").split(),
                list(CANDIDATE_NAMES),
            )
            self.assertTrue(match_call["matching_options"].guided_matching)
            self.assertEqual(
                match_call["matching_options"].kwargs,
                {"num_threads": 4, "use_gpu": False},
            )
            self.assertEqual(match_call["device"], "cpu")

            mapping_call = calls[5][1]
            self.assertEqual(
                mapping_call["database_path"], output / "database.db"
            )
            self.assertEqual(mapping_call["image_path"], output / "images")
            self.assertEqual(mapping_call["output_path"], output / "sparse")
            self.assertEqual(
                mapping_call["input_path"], str(output / "resume_model")
            )
            mapping_options = mapping_call["options"]
            self.assertEqual(mapping_options.kwargs, {"num_threads": 4})
            self.assertEqual(mapping_options.ba_global_frames_ratio, 1.4)
            self.assertEqual(mapping_options.ba_global_points_ratio, 1.4)
            self.assertEqual(mapping_options.ba_global_max_refinements, 3)
            self.assertEqual(
                mapping_options.ba_global_max_num_iterations, 30
            )

            # The augmented dataset holds every image plus the copied inputs.
            staged_names = sorted(
                path.name for path in (output / "images").iterdir()
            )
            self.assertEqual(
                staged_names,
                sorted([*BASE_IMAGE_NAMES, *CANDIDATE_NAMES]),
            )
            self.assertEqual(
                (output / "database.db").read_bytes(),
                b"base colmap database",
            )
            self.assertEqual(
                (output / "candidate_names.txt").read_text(
                    encoding="utf-8"
                ).split(),
                list(CANDIDATE_NAMES),
            )
            self.assertFalse((output / "resume_model").exists())
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                self.assertGreater(
                    (model_path / name).stat().st_size, 0
                )

            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["dataset_type"], "video_set")
            self.assertEqual(
                metadata["bootstrap"],
                {
                    "candidates_dataset": str(candidates),
                    "candidates_instance_id": "cand-instance",
                    "num_candidates": 2,
                    "names_file": "candidate_names.txt",
                },
            )

            stats = json.loads(
                (output / "reconstruction_stats.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stats["status"], "passed")
            self.assertEqual(
                stats["base"],
                {
                    "dataset": str(base),
                    "num_registered_images": 4,
                    "num_points3D": 100,
                    "selected_model_files": {
                        "cameras.bin": {
                            "size_bytes": 5,
                            "sha256": "0" * 64,
                        },
                    },
                },
            )
            self.assertEqual(
                stats["feature_extraction"],
                {
                    "max_image_size": 3840,
                    "max_num_features": 8192,
                    "estimate_affine_shape": True,
                    "domain_size_pooling": True,
                },
            )
            self.assertEqual(
                stats["feature_matching"], {"guided_matching": True}
            )
            self.assertEqual(
                stats["mapper"],
                {
                    "ba_global_frames_ratio": 1.4,
                    "ba_global_points_ratio": 1.4,
                    "ba_global_max_refinements": 3,
                    "ba_global_max_num_iterations": 30,
                },
            )
            self.assertEqual(
                stats["bootstrap"],
                {
                    "num_candidates": 2,
                    "num_candidate_registered": 2,
                    "candidate_registration_ratio": 1.0,
                    "min_candidate_registration_ratio": 0.8,
                    "match_list": "bootstrap_match_list.txt",
                    "vocab_tree": {
                        "path": str(vocab_tree),
                        "sha256": hashlib.sha256(
                            vocab_tree.read_bytes()
                        ).hexdigest(),
                    },
                },
            )
            self.assertEqual(stats["num_input_images"], 6)
            self.assertEqual(stats["num_registered_images"], 6)
            self.assertEqual(stats["num_points3D"], 500)
            self.assertEqual(stats["registration_ratio"], 1.0)
            self.assertEqual(stats["matching_strategy"], "vocab_tree")
            self.assertIn("selected_model_files", stats)

            # The base and candidate datasets are left byte-identical.
            self.assertEqual(_tree_snapshot(base), base_before)
            self.assertEqual(_tree_snapshot(candidates), candidates_before)

    def test_hard_fails_when_extraction_adds_a_second_camera(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        reconstruction = FakeReconstruction(registered=6, points=500)
        fake_pycolmap = _fake_pycolmap(
            calls, {0: reconstruction}, db_num_cameras=2
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            vocab_tree = _make_vocab_tree(root)

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                with self.assertRaisesRegex(RuntimeError, "2 cameras"):
                    register_new_frames(
                        base, candidates, root / "out", vocab_tree
                    )

            self.assertNotIn(
                "match_vocabtree", [name for name, _ in calls]
            )

    def test_hard_fails_when_base_model_has_multiple_cameras(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        fake_pycolmap = _fake_pycolmap(calls, {}, base_camera_ids=(3, 5))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = _make_base(root)
            candidates = _make_candidates(root)
            vocab_tree = _make_vocab_tree(root)

            with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                with self.assertRaisesRegex(
                    RuntimeError, "exactly one camera"
                ):
                    register_new_frames(
                        base, candidates, root / "out", vocab_tree
                    )

            self.assertNotIn(
                "extract_features", [name for name, _ in calls]
            )

    def test_gate_failures_write_failed_stats_and_raise(self) -> None:
        cases = (
            (
                "candidate gate",
                FakeReconstruction(
                    registered=5,
                    points=500,
                    registered_names=[*BASE_IMAGE_NAMES, CANDIDATE_NAMES[0]],
                    unregistered_names=[CANDIDATE_NAMES[1]],
                ),
                r"candidate images registered \(50\.0%\)",
                1,
            ),
            (
                "base gate",
                FakeReconstruction(
                    registered=3,
                    points=500,
                    registered_names=[BASE_IMAGE_NAMES[0], *CANDIDATE_NAMES],
                    unregistered_names=list(BASE_IMAGE_NAMES[1:]),
                ),
                r"registers 3 images but the base model registered 4",
                2,
            ),
        )
        for label, reconstruction, message, expected_registered in cases:
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
                calls: list[tuple[str, dict[str, object]]] = []
                fake_pycolmap = _fake_pycolmap(calls, {0: reconstruction})
                root = Path(temporary).resolve()
                base = _make_base(root)
                candidates = _make_candidates(root)
                vocab_tree = _make_vocab_tree(root)
                output = root / "out"

                with patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                    with self.assertRaisesRegex(RuntimeError, message):
                        register_new_frames(
                            base, candidates, output, vocab_tree
                        )

                self.assertIsNone(reconstruction.written_to)
                self.assertTrue((output / "resume_model").is_dir())
                stats = json.loads(
                    (output / "reconstruction_stats.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(stats["status"], "failed")
                self.assertEqual(
                    stats["bootstrap"]["num_candidate_registered"],
                    expected_registered,
                )
                self.assertNotIn("selected_model_files", stats)


if __name__ == "__main__":
    unittest.main()
