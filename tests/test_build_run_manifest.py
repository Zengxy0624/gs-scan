from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_run_manifest import build_manifest


def create_sparse_model(dataset: Path) -> None:
    model = dataset / "sparse" / "0"
    model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(name.encode("ascii"))


class RunManifestTests(unittest.TestCase):
    def test_manifest_changes_with_source_dataset_code_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.mov"
            source.write_bytes(b"source")
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "metadata.json").write_text("{}", encoding="utf-8")
            (dataset / "reconstruction_stats.json").write_text(
                '{"status":"passed"}',
                encoding="utf-8",
            )
            create_sparse_model(dataset)
            pipeline = root / "pipeline.py"
            pipeline.write_text("version = 1\n", encoding="utf-8")

            baseline = build_manifest(
                "scene",
                dataset,
                source,
                [pipeline],
                6.0,
                100000,
                15,
                0.5,
                30000,
            )
            self.assertEqual(baseline["scene"], "scene")
            self.assertEqual(baseline["source"]["size_bytes"], 6)
            self.assertEqual(
                baseline["parameters"]["max_steps"],
                30000,
            )

            source.write_bytes(b"changed source")
            self.assertNotEqual(
                baseline,
                build_manifest(
                    "scene",
                    dataset,
                    source,
                    [pipeline],
                    6.0,
                    100000,
                    15,
                    0.5,
                    30000,
                ),
            )
            source.write_bytes(b"source")
            pipeline.write_text("version = 2\n", encoding="utf-8")
            changed_code = build_manifest(
                "scene",
                dataset,
                source,
                [pipeline],
                6.0,
                100000,
                15,
                0.5,
                30000,
            )
            self.assertNotEqual(
                baseline["pipeline_files"],
                changed_code["pipeline_files"],
            )
            sparse_before = changed_code["dataset_artifacts"]
            (dataset / "sparse" / "0" / "points3D.bin").write_bytes(
                b"changed sparse model"
            )
            changed_sparse = build_manifest(
                "scene",
                dataset,
                source,
                [pipeline],
                6.0,
                100000,
                15,
                0.5,
                30000,
            )
            self.assertNotEqual(
                sparse_before,
                changed_sparse["dataset_artifacts"],
            )
            changed_steps = build_manifest(
                "scene",
                dataset,
                source,
                [pipeline],
                6.0,
                100000,
                15,
                0.5,
                7000,
            )
            self.assertNotEqual(
                changed_code["parameters"],
                changed_steps["parameters"],
            )

if __name__ == "__main__":
    unittest.main()
