from __future__ import annotations

import contextlib
import csv
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from archive.scripts.score_candidate_frames import (
    CSV_FIELDS,
    affine_fit,
    apply_affine,
    apply_parser_transform,
    main,
    psnr,
    rank_rows,
    score_candidates,
    source_prefix,
    write_scores_csv,
)


class AffineFitTests(unittest.TestCase):
    def test_recovers_known_gain_and_bias(self) -> None:
        rng = np.random.default_rng(7)
        rendered = rng.uniform(0.2, 0.7, size=(12, 16, 3))
        gain_true = np.array([1.15, 0.9, 1.05])
        bias_true = np.array([-0.05, 0.08, 0.02])
        reference = rendered * gain_true + bias_true

        gain, bias = affine_fit(rendered, reference)

        np.testing.assert_allclose(gain, gain_true, atol=1e-10)
        np.testing.assert_allclose(bias, bias_true, atol=1e-10)
        np.testing.assert_allclose(
            apply_affine(rendered, gain, bias), reference, atol=1e-10
        )

    def test_improves_psnr_on_exposure_shifted_pair(self) -> None:
        rng = np.random.default_rng(11)
        rendered = rng.uniform(0.1, 0.9, size=(20, 20, 3))
        reference = np.clip(rendered * 0.8 + 0.12, 0.0, 1.0)

        raw = psnr(rendered, reference)
        gain, bias = affine_fit(rendered, reference)
        corrected = psnr(apply_affine(rendered, gain, bias), reference)

        self.assertGreater(corrected, raw)

    def test_shape_mismatch_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            affine_fit(np.zeros((2, 2, 3)), np.zeros((3, 2, 3)))


class PsnrTests(unittest.TestCase):
    def test_known_value(self) -> None:
        rendered = np.zeros((4, 4, 3))
        reference = np.full((4, 4, 3), 0.5)
        self.assertAlmostEqual(
            psnr(rendered, reference), -10.0 * math.log10(0.25), places=12
        )

    def test_identical_arrays_are_infinite(self) -> None:
        image = np.full((3, 3, 3), 0.25)
        self.assertEqual(psnr(image, image), math.inf)

    def test_shape_mismatch_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            psnr(np.zeros((2, 2, 3)), np.zeros((2, 3, 3)))


def _row(name: str, psnr_affine: float | None) -> dict[str, object]:
    registered = psnr_affine is not None
    return {
        "name": name,
        "source": source_prefix(name),
        "registered": registered,
        "psnr_raw": (psnr_affine - 1.0) if registered else None,
        "psnr_affine": psnr_affine,
        "affine_gain_r": 1.0 if registered else None,
        "affine_gain_g": 1.0 if registered else None,
        "affine_gain_b": 1.0 if registered else None,
    }


class RankingTests(unittest.TestCase):
    def test_ranks_ascending_with_unregistered_last(self) -> None:
        rows = [
            _row("00_a_m000010.jpg", 24.0),
            _row("00_a_m000020.jpg", None),
            _row("01_b_m000030.jpg", 18.5),
            _row("01_b_m000040.jpg", 31.0),
        ]

        ranked = rank_rows(rows)

        self.assertEqual(
            [row["name"] for row in ranked],
            [
                "01_b_m000030.jpg",
                "00_a_m000010.jpg",
                "01_b_m000040.jpg",
                "00_a_m000020.jpg",
            ],
        )
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3, None])
        self.assertEqual(
            [row["registered"] for row in ranked], [True, True, True, False]
        )

    def test_source_prefix_keeps_bootstrap_marker_token(self) -> None:
        self.assertEqual(
            source_prefix("05_a001_08071524_c009_m001142.jpg"),
            "05_a001_08071524_c009_",
        )


class ScoreCandidatesTests(unittest.TestCase):
    def test_scores_via_injected_render_callback(self) -> None:
        rng = np.random.default_rng(3)
        renders = {
            "00_a_m000001.jpg": rng.uniform(0.2, 0.8, size=(8, 8, 3)),
            "00_a_m000002.jpg": rng.uniform(0.2, 0.8, size=(8, 8, 3)),
        }

        def render_frame(name: str) -> np.ndarray | None:
            return renders.get(name)

        def load_frame(name: str) -> np.ndarray:
            return np.clip(renders[name] * 0.9 + 0.05, 0.0, 1.0)

        rows = score_candidates(
            ["00_a_m000001.jpg", "00_a_m000003.jpg", "00_a_m000002.jpg"],
            render_frame,
            load_frame,
        )

        self.assertEqual(len(rows), 3)
        by_name = {row["name"]: row for row in rows}
        self.assertFalse(by_name["00_a_m000003.jpg"]["registered"])
        self.assertIsNone(by_name["00_a_m000003.jpg"]["psnr_affine"])
        for name in renders:
            row = by_name[name]
            self.assertTrue(row["registered"])
            self.assertGreater(row["psnr_affine"], row["psnr_raw"])
            self.assertAlmostEqual(row["affine_gain_r"], 0.9, places=6)


class CsvTests(unittest.TestCase):
    def test_written_csv_shape_and_empty_unregistered_fields(self) -> None:
        ranked = rank_rows(
            [
                _row("00_a_m000010.jpg", 24.0),
                _row("00_a_m000020.jpg", None),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "scores.csv"
            write_scores_csv(ranked, output)
            with output.open(newline="", encoding="utf-8") as handle:
                records = list(csv.reader(handle))

        self.assertEqual(records[0], list(CSV_FIELDS))
        self.assertEqual(len(records), 3)
        self.assertEqual(
            records[1],
            [
                "1",
                "00_a_m000010.jpg",
                "00_a_",
                "true",
                "23.0000",
                "24.0000",
                "1.000000",
                "1.000000",
                "1.000000",
            ],
        )
        self.assertEqual(
            records[2],
            ["", "00_a_m000020.jpg", "00_a_", "false", "", "", "", "", ""],
        )


class ParserTransformTests(unittest.TestCase):
    def test_matches_hand_computed_similarity(self) -> None:
        # gsplat's datasets.colmap.Parser stores the applied world
        # normalization on the ``transform`` attribute (a 4x4 similarity) and
        # applies it via datasets.normalize.transform_cameras: left-multiply,
        # then divide the rotation block by the norm of its first row.
        scale = 2.0
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        transform = np.eye(4)
        transform[:3, :3] = scale * rotation
        transform[:3, 3] = [1.0, -2.0, 3.0]
        stub_parser = SimpleNamespace(transform=transform)

        camtoworld = np.eye(4)
        camtoworld[:3, 3] = [4.0, 5.0, 6.0]

        result = apply_parser_transform(stub_parser, camtoworld)

        expected = np.eye(4)
        expected[:3, :3] = rotation  # rescaled back to a rotation
        expected[:3, 3] = scale * rotation @ [4.0, 5.0, 6.0] + [1.0, -2.0, 3.0]
        np.testing.assert_allclose(result, expected, atol=1e-12)
        np.testing.assert_allclose(expected[:3, 3], [-9.0, 6.0, 15.0])

    def test_rejects_bad_shapes(self) -> None:
        stub_parser = SimpleNamespace(transform=np.eye(3))
        with self.assertRaises(ValueError):
            apply_parser_transform(stub_parser, np.eye(4))
        stub_parser = SimpleNamespace(transform=np.eye(4))
        with self.assertRaises(ValueError):
            apply_parser_transform(stub_parser, np.eye(3))


class CliValidationTests(unittest.TestCase):
    def run_main(self, argv: list[str]) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(argv)
        return code, stderr.getvalue()

    def test_missing_augmented_dataset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, stderr = self.run_main(
                [
                    "--dataset",
                    str(root / "missing"),
                    "--base-dataset",
                    str(root),
                    "--ply",
                    str(root / "model.ply"),
                    "--output",
                    str(root / "scores.csv"),
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("Augmented dataset not found", stderr)

    def test_bad_downscale_fails_before_touching_disk(self) -> None:
        code, stderr = self.run_main(
            [
                "--dataset",
                "unused",
                "--base-dataset",
                "unused",
                "--ply",
                "unused.ply",
                "--output",
                "unused.csv",
                "--downscale",
                "0",
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("--downscale", stderr)

    def test_missing_names_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "augmented"
            (dataset / "sparse" / "0").mkdir(parents=True)
            (dataset / "images").mkdir()
            base = root / "base"
            (base / "sparse" / "0").mkdir(parents=True)
            ply = root / "model.ply"
            ply.write_bytes(b"ply")
            code, stderr = self.run_main(
                [
                    "--dataset",
                    str(dataset),
                    "--base-dataset",
                    str(base),
                    "--ply",
                    str(ply),
                    "--output",
                    str(root / "scores.csv"),
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("Candidate name list not found", stderr)


if __name__ == "__main__":
    unittest.main()
