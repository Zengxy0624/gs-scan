from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.export_rerun_viz import (
    SOURCE_COLOR_TABLE,
    SUBSAMPLE_SEED,
    build_parser,
    camera_k_matrix,
    main,
    source_prefix,
)


OPENCV_PARAMS = (2614.0, 2615.0, 1920.0, 1080.0, 0.05, -0.08, -0.001, 0.0002)


class FakeRotation:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = np.asarray(matrix, dtype=np.float64)

    def matrix(self) -> np.ndarray:
        return self._matrix


class FakeRigid3d:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.rotation = FakeRotation(rotation)
        self.translation = np.asarray(translation, dtype=np.float64)


class FakeImage:
    def __init__(
        self,
        name: str,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        camera_id: int = 1,
        has_pose: bool = True,
    ) -> None:
        self.name = name
        self.camera_id = camera_id
        self.has_pose = has_pose
        self._pose = FakeRigid3d(
            rotation if rotation is not None else np.eye(3),
            translation if translation is not None else np.zeros(3),
        )

    def cam_from_world(self) -> FakeRigid3d:
        return self._pose


class FakeCamera:
    def __init__(
        self,
        model_name: str = "OPENCV",
        params: tuple[float, ...] = OPENCV_PARAMS,
        width: int = 3840,
        height: int = 2160,
    ) -> None:
        self.model = types.SimpleNamespace(name=model_name)
        self.params = list(params)
        self.width = width
        self.height = height


class FakePoint3D:
    def __init__(self, xyz, color) -> None:
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.color = np.asarray(color, dtype=np.uint8)


class FakeReconstruction:
    def __init__(self, images=None, cameras=None, points3D=None) -> None:
        self.images = images or {}
        self.cameras = cameras or {1: FakeCamera()}
        self.points3D = points3D or {}


class FakeRecording:
    def __init__(self) -> None:
        self.flushed = 0

    def flush(self) -> None:
        self.flushed += 1


def _archetype(kind):
    class Archetype:
        def __init__(self, *args, **kwargs) -> None:
            self.kind = kind
            self.args = args
            self.kwargs = kwargs

    Archetype.__name__ = kind
    return Archetype


def _make_fake_rerun() -> types.ModuleType:
    module = types.ModuleType("rerun")
    module.calls = []
    module.recording = FakeRecording()
    module.Points3D = _archetype("Points3D")
    module.Transform3D = _archetype("Transform3D")
    module.Pinhole = _archetype("Pinhole")

    def init(application_id, **kwargs):
        module.calls.append(("init", application_id, kwargs))

    def save(path, **kwargs):
        Path(path).write_bytes(b"fake rrd payload")
        module.calls.append(("save", path))

    def log(entity_path, entity, **kwargs):
        module.calls.append(("log", entity_path, entity, kwargs))

    module.init = init
    module.save = save
    module.log = log
    module.get_global_data_recording = lambda: module.recording

    def disconnect(recording=None):
        module.calls.append(("disconnect", recording))

    module.disconnect = disconnect
    return module


def _make_fake_pycolmap(reconstruction: FakeReconstruction) -> types.ModuleType:
    module = types.ModuleType("pycolmap")
    module.loaded_paths = []

    def load(path):
        module.loaded_paths.append(path)
        return reconstruction

    module.Reconstruction = load
    return module


def _default_points(count: int = 5) -> dict[int, FakePoint3D]:
    return {
        index: FakePoint3D([index, 0.0, 1.0], [index, 20, 30])
        for index in range(count)
    }


class ExportRerunVizTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.output = self.root / "viz.rrd"

    def _make_dataset(self, model_files=("cameras.bin", "images.bin", "points3D.bin")):
        dataset = self.root / "dataset"
        model = dataset / "sparse" / "0"
        model.mkdir(parents=True)
        for name in model_files:
            (model / name).write_bytes(b"binary " + name.encode("ascii"))
        return dataset

    def _run(self, reconstruction: FakeReconstruction, argv=None):
        dataset = self.root / "dataset"
        if not dataset.exists():
            self._make_dataset()
        fake_rerun = _make_fake_rerun()
        fake_pycolmap = _make_fake_pycolmap(reconstruction)
        modules = {"rerun": fake_rerun, "pycolmap": fake_pycolmap}
        with patch.dict(sys.modules, modules):
            exit_code = main(
                argv
                if argv is not None
                else [str(dataset), "--output", str(self.output)]
            )
        return exit_code, fake_rerun, fake_pycolmap

    def _logged(self, fake_rerun, entity_path):
        return [
            call for call in fake_rerun.calls
            if call[0] == "log" and call[1] == entity_path
        ]

    # --- CLI validation -------------------------------------------------

    def test_missing_dataset_directory_fails(self) -> None:
        exit_code, fake_rerun, _ = self._run(
            FakeReconstruction(),
            argv=[str(self.root / "absent"), "--output", str(self.output)],
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(fake_rerun.calls, [])
        self.assertFalse(self.output.exists())

    def test_missing_model_file_fails(self) -> None:
        self._make_dataset(model_files=("cameras.bin", "images.bin"))
        exit_code, fake_rerun, _ = self._run(FakeReconstruction())
        self.assertEqual(exit_code, 1)
        self.assertEqual(fake_rerun.calls, [])

    def test_existing_output_fails(self) -> None:
        self._make_dataset()
        self.output.write_bytes(b"already there")
        exit_code, fake_rerun, _ = self._run(FakeReconstruction())
        self.assertEqual(exit_code, 1)
        self.assertEqual(fake_rerun.calls, [])
        self.assertEqual(self.output.read_bytes(), b"already there")

    def test_bad_max_points_fails(self) -> None:
        dataset = self._make_dataset()
        exit_code, fake_rerun, _ = self._run(
            FakeReconstruction(),
            argv=[
                str(dataset),
                "--output",
                str(self.output),
                "--max-points",
                "0",
            ],
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(fake_rerun.calls, [])

    def test_bad_every_nth_camera_fails(self) -> None:
        dataset = self._make_dataset()
        exit_code, fake_rerun, _ = self._run(
            FakeReconstruction(),
            argv=[
                str(dataset),
                "--output",
                str(self.output),
                "--every-nth-camera",
                "-1",
            ],
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(fake_rerun.calls, [])

    def test_output_flag_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([str(self.root / "dataset")])

    # --- Points ----------------------------------------------------------

    def test_points_logged_static_with_colors_and_radius(self) -> None:
        reconstruction = FakeReconstruction(points3D=_default_points(4))
        exit_code, fake_rerun, fake_pycolmap = self._run(reconstruction)
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            fake_pycolmap.loaded_paths,
            [str(self.root / "dataset" / "sparse" / "0")],
        )
        (call,) = self._logged(fake_rerun, "points")
        entity = call[2]
        self.assertEqual(entity.kind, "Points3D")
        np.testing.assert_array_equal(
            entity.args[0], [[i, 0.0, 1.0] for i in range(4)]
        )
        np.testing.assert_array_equal(
            entity.kwargs["colors"], [[i, 20, 30] for i in range(4)]
        )
        self.assertEqual(entity.kwargs["radii"], 0.002)
        self.assertEqual(call[3], {"static": True})

    def test_subsampling_caps_points_and_is_deterministic(self) -> None:
        count, cap = 50, 10
        argv_extra = ["--max-points", str(cap)]
        dataset = self._make_dataset()
        argv = [str(dataset), "--output", str(self.output), *argv_extra]
        exit_code, fake_rerun, _ = self._run(
            FakeReconstruction(points3D=_default_points(count)), argv=argv
        )
        self.assertEqual(exit_code, 0)
        (call,) = self._logged(fake_rerun, "points")
        positions = np.asarray(call[2].args[0])
        colors = np.asarray(call[2].kwargs["colors"])
        self.assertEqual(positions.shape, (cap, 3))
        expected = np.random.default_rng(SUBSAMPLE_SEED).choice(
            count, size=cap, replace=False
        )
        np.testing.assert_array_equal(positions[:, 0], expected)
        np.testing.assert_array_equal(colors[:, 0], expected)

        # A second run reproduces the identical subsample.
        self.output.unlink()
        exit_code, fake_rerun_again, _ = self._run(
            FakeReconstruction(points3D=_default_points(count)), argv=argv
        )
        self.assertEqual(exit_code, 0)
        (again,) = self._logged(fake_rerun_again, "points")
        np.testing.assert_array_equal(np.asarray(again[2].args[0]), positions)

    # --- Camera grouping, colors, thinning -------------------------------

    def test_source_prefix_grouping(self) -> None:
        self.assertEqual(source_prefix("00_vid_000123.jpg"), "00_vid_")
        self.assertEqual(source_prefix("17_a001_08071705_c028_000408.jpg"), "17_a001_08071705_c028_")

    def test_camera_centers_colored_by_source(self) -> None:
        images = {}
        image_id = 1
        for source, count in (("00_a_", 3), ("01_b_", 2)):
            for index in range(count):
                name = f"{source}{index:06d}.jpg"
                images[image_id] = FakeImage(name, translation=[image_id, 0, 0])
                image_id += 1
        reconstruction = FakeReconstruction(images=images)
        exit_code, fake_rerun, _ = self._run(reconstruction)
        self.assertEqual(exit_code, 0)
        (call,) = self._logged(fake_rerun, "camera_centers")
        entity = call[2]
        self.assertEqual(entity.kind, "Points3D")
        self.assertEqual(call[3], {"static": True})
        colors = np.asarray(entity.kwargs["colors"])
        self.assertEqual(colors.shape, (5, 3))
        np.testing.assert_array_equal(
            colors,
            [SOURCE_COLOR_TABLE[0]] * 3 + [SOURCE_COLOR_TABLE[1]] * 2,
        )
        # Identity rotation: center = -t.
        positions = np.asarray(entity.args[0])
        np.testing.assert_allclose(positions[:, 0], [-1, -2, -3, -4, -5])

    def test_every_nth_camera_selection(self) -> None:
        images = {
            index + 1: FakeImage(f"00_a_{index:06d}.jpg")
            for index in range(5)
        }
        dataset = self._make_dataset()
        argv = [
            str(dataset),
            "--output",
            str(self.output),
            "--every-nth-camera",
            "2",
        ]
        exit_code, fake_rerun, _ = self._run(
            FakeReconstruction(images=images), argv=argv
        )
        self.assertEqual(exit_code, 0)
        frusta_entities = sorted(
            {
                call[1]
                for call in fake_rerun.calls
                if call[0] == "log" and call[1].startswith("cameras/")
            }
        )
        self.assertEqual(
            frusta_entities,
            [
                "cameras/00_a_/00_a_000000.jpg",
                "cameras/00_a_/00_a_000002.jpg",
                "cameras/00_a_/00_a_000004.jpg",
            ],
        )
        # All five centers are still present.
        (centers,) = self._logged(fake_rerun, "camera_centers")
        self.assertEqual(np.asarray(centers[2].args[0]).shape, (5, 3))

    def test_unposed_images_are_skipped(self) -> None:
        images = {
            1: FakeImage("00_a_000000.jpg"),
            2: FakeImage("00_a_000001.jpg", has_pose=False),
        }
        exit_code, fake_rerun, _ = self._run(FakeReconstruction(images=images))
        self.assertEqual(exit_code, 0)
        (centers,) = self._logged(fake_rerun, "camera_centers")
        self.assertEqual(np.asarray(centers[2].args[0]).shape, (1, 3))

    # --- Transform math and intrinsics -----------------------------------

    def test_transform_math_on_toy_pose(self) -> None:
        # 90 degree rotation about z, world-to-camera.
        rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([1.0, 2.0, 3.0])
        images = {1: FakeImage("00_a_000000.jpg", rotation, translation)}
        exit_code, fake_rerun, _ = self._run(FakeReconstruction(images=images))
        self.assertEqual(exit_code, 0)
        entity_path = "cameras/00_a_/00_a_000000.jpg"
        calls = self._logged(fake_rerun, entity_path)
        self.assertEqual(len(calls), 2)
        transform = calls[0][2]
        self.assertEqual(transform.kind, "Transform3D")
        self.assertEqual(calls[0][3], {"static": True})
        expected_center = -rotation.T @ translation
        np.testing.assert_allclose(transform.kwargs["mat3x3"], rotation.T)
        np.testing.assert_allclose(transform.kwargs["translation"], expected_center)
        (centers,) = self._logged(fake_rerun, "camera_centers")
        np.testing.assert_allclose(
            np.asarray(centers[2].args[0])[0], expected_center
        )

    def test_pinhole_k_from_opencv_params(self) -> None:
        images = {1: FakeImage("00_a_000000.jpg")}
        exit_code, fake_rerun, _ = self._run(FakeReconstruction(images=images))
        self.assertEqual(exit_code, 0)
        calls = self._logged(fake_rerun, "cameras/00_a_/00_a_000000.jpg")
        pinhole = calls[1][2]
        self.assertEqual(pinhole.kind, "Pinhole")
        self.assertEqual(calls[1][3], {"static": True})
        fx, fy, cx, cy = OPENCV_PARAMS[:4]
        self.assertEqual(
            pinhole.kwargs["image_from_camera"],
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(pinhole.kwargs["resolution"], [3840, 2160])

    def test_camera_k_matrix_rejects_unsupported_model(self) -> None:
        with self.assertRaises(ValueError):
            camera_k_matrix("SIMPLE_RADIAL", (2600.0, 1920.0, 1080.0, 0.01))
        with self.assertRaises(ValueError):
            camera_k_matrix("PINHOLE", (2600.0, 2600.0))
        self.assertEqual(
            camera_k_matrix("PINHOLE", (10.0, 20.0, 30.0, 40.0)),
            [[10.0, 0.0, 30.0], [0.0, 20.0, 40.0], [0.0, 0.0, 1.0]],
        )

    # --- Recording lifecycle ---------------------------------------------

    def test_init_and_save_precede_logging(self) -> None:
        reconstruction = FakeReconstruction(
            images={1: FakeImage("00_a_000000.jpg")},
            points3D=_default_points(2),
        )
        exit_code, fake_rerun, _ = self._run(reconstruction)
        self.assertEqual(exit_code, 0)
        kinds = [call[0] for call in fake_rerun.calls]
        self.assertEqual(kinds[:2], ["init", "save"])
        self.assertIn("log", kinds[2:])
        init_call = fake_rerun.calls[0]
        self.assertEqual(init_call[1], "3dgs-health-sfm")
        self.assertEqual(init_call[2], {"spawn": False})
        self.assertEqual(fake_rerun.calls[1][1], str(self.output))
        self.assertGreaterEqual(fake_rerun.recording.flushed, 1)
        self.assertEqual(fake_rerun.calls[-1][0], "disconnect")
        self.assertTrue(self.output.is_file())

    def test_custom_application_id(self) -> None:
        dataset = self._make_dataset()
        argv = [
            str(dataset),
            "--output",
            str(self.output),
            "--application-id",
            "my-viz",
        ]
        exit_code, fake_rerun, _ = self._run(
            FakeReconstruction(points3D=_default_points(1)), argv=argv
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_rerun.calls[0][1], "my-viz")


if __name__ == "__main__":
    unittest.main()
