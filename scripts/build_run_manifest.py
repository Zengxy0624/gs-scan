#!/usr/bin/env python3
"""Build and compare semantic provenance manifests for gsplat runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 2
CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _source_identity(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source capture not found: {resolved}")
    stat = resolved.stat()
    digest = _sha256(resolved)
    recorded_digest = metadata.get("source_sha256")
    if (
        isinstance(recorded_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", recorded_digest)
        and recorded_digest != digest
    ):
        raise ValueError("Source capture content no longer matches frame metadata")
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def _artifact_identity(path: Path, base: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Manifest input not found: {resolved}")
    try:
        label = str(resolved.relative_to(base))
    except ValueError:
        label = str(resolved)
    return {
        "path": label,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _image_tree_identity(image_dir: Path) -> dict[str, Any]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training images not found: {image_dir}")
    digest = hashlib.sha256()
    count = 0
    size_bytes = 0
    for image in sorted(path for path in image_dir.rglob("*") if path.is_file()):
        label = image.relative_to(image_dir).as_posix()
        image_size = image.stat().st_size
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(image).encode("ascii"))
        digest.update(b"\0")
        count += 1
        size_bytes += image_size
    if count == 0:
        raise ValueError(f"Training image tree is empty: {image_dir}")
    return {
        "path": "images",
        "file_count": count,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _clean_git_revision(repository: Path) -> str:
    repository = repository.expanduser().resolve()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError(
            f"gsplat submodule has local changes; refusing reusable training: "
            f"{repository}"
        )
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Could not resolve gsplat Git revision: {repository}")
    return revision


def build_manifest(
    scene: str,
    dataset: Path,
    source: Path,
    sample_fps: float,
    max_frames: int,
    overlap: int,
    min_registration_ratio: float,
    max_steps: int,
    camera_model: str,
    gsplat_revision: str,
    colmap_threads: int = -1,
    gpu_id: str = "unspecified",
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Training dataset not found: {dataset}")
    if not scene:
        raise ValueError("scene must not be empty")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive and finite")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    if overlap < 1:
        raise ValueError("overlap must be positive")
    if not 0 < min_registration_ratio <= 1:
        raise ValueError("min_registration_ratio must be in (0, 1]")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if camera_model not in CAMERA_MODELS:
        raise ValueError(f"Unsupported camera_model: {camera_model}")
    if colmap_threads == 0 or colmap_threads < -1:
        raise ValueError("colmap_threads must be -1 or positive")
    if not re.fullmatch(r"[0-9]+|unspecified", gpu_id):
        raise ValueError("gpu_id must be non-negative or 'unspecified'")
    if not re.fullmatch(r"[0-9a-f]{40}", gsplat_revision):
        raise ValueError("gsplat_revision must be a full 40-character Git commit")

    metadata_path = dataset / "metadata.json"
    stats_path = dataset / "reconstruction_stats.json"
    metadata = _read_json_object(metadata_path)
    stats = _read_json_object(stats_path)
    dataset_instance_id = str(metadata.get("dataset_instance_id"))
    try:
        uuid.UUID(dataset_instance_id)
    except ValueError:
        raise ValueError("Dataset has no valid instance identifier") from None
    source_snapshot = _source_identity(source, metadata)
    if not math.isclose(
        float(metadata.get("sample_fps", -1)),
        sample_fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Dataset sample_fps does not match manifest request")
    if metadata.get("max_frames") != max_frames:
        raise ValueError("Dataset max_frames does not match manifest request")
    expected_reconstruction = {
        "status": "passed",
        "camera_model": camera_model,
        "matching_overlap": overlap,
        "matching_quadratic_overlap": True,
        "minimum_registration_ratio": min_registration_ratio,
        "selected_model_path": "sparse/0",
    }
    for key, expected_value in expected_reconstruction.items():
        if stats.get(key) != expected_value:
            raise ValueError(
                f"COLMAP {key} does not match manifest request: "
                f"expected {expected_value!r}, found {stats.get(key)!r}"
            )

    image_snapshot = _image_tree_identity(dataset / "images")
    if metadata.get("images_sha256") != image_snapshot["sha256"]:
        raise ValueError("Extracted-image content no longer matches frame metadata")

    # These recipe versions are semantic cache keys. Bump the relevant value
    # whenever result-affecting extraction or reconstruction behavior changes.
    identity = {
        "dataset": {
            "source_sha256": source_snapshot["sha256"],
            "extraction": {
                "recipe_version": 1,
                "sample_fps": sample_fps,
                "max_frames": max_frames,
                "jpeg_quality": metadata.get("jpeg_quality", 95),
                "images_sha256": image_snapshot["sha256"],
            },
            "reconstruction": {
                "recipe_version": 2,
                "camera_mode": "SINGLE",
                "camera_model": camera_model,
                "matching": "sequential",
                "overlap": overlap,
                "quadratic_overlap": True,
            },
        },
        "trainer": {
            "implementation": {
                "repository": "nerfstudio-project/gsplat",
                "commit": gsplat_revision,
                "entrypoint": "examples/simple_trainer.py",
            },
            "config": {
                "preset": "default",
                "data_type": "colmap",
                "data_factor": 1,
                "camera_model": "pinhole",
                "max_steps": max_steps,
                "eval_steps": [max_steps],
                "save_steps": [max_steps],
                "ply_steps": [max_steps],
                "save_ply": True,
                "disable_viewer": True,
            },
        },
    }

    dataset_files = [metadata_path, stats_path]
    dataset_files.extend(
        dataset / "sparse" / "0" / name
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    )
    artifact_snapshots = [
        _artifact_identity(path, dataset) for path in dataset_files
    ]
    recorded_model_files = stats.get("selected_model_files")
    if not isinstance(recorded_model_files, dict):
        raise ValueError("COLMAP model-integrity record is missing")
    for artifact in artifact_snapshots[-3:]:
        recorded = recorded_model_files.get(Path(artifact["path"]).name)
        if not isinstance(recorded, dict) or any(
            recorded.get(key) != artifact[key] for key in ("size_bytes", "sha256")
        ):
            raise ValueError(
                f"COLMAP model file no longer matches reconstruction stats: "
                f"{artifact['path']}"
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "identity_sha256": _canonical_digest(identity),
        "provenance": {
            "scene": scene,
            "source": source_snapshot,
            "quality_gate": {
                "min_registration_ratio": min_registration_ratio,
                "recorded_status": stats.get("status"),
                "registered_images": stats.get("num_registered_images"),
                "input_images": stats.get("num_input_images"),
            },
            "resources": {
                "colmap_threads": colmap_threads,
                "training_gpu": gpu_id,
            },
            "dataset_snapshot": {
                "instance_id": dataset_instance_id,
                "images": image_snapshot,
                "artifacts": artifact_snapshots,
                "reconstruction_stats": stats,
            },
            "environment": {
                "python": sys.version.split()[0],
                "pycolmap": _package_version("pycolmap"),
                "torch": _package_version("torch"),
                "gsplat": _package_version("gsplat"),
            },
        },
    }
    return manifest


def manifest_identity_matches(
    expected: Path,
    actual: Path,
    strict_snapshot: bool = False,
) -> bool:
    expected_manifest = _read_json_object(expected.expanduser().resolve())
    actual_manifest = _read_json_object(actual.expanduser().resolve())
    for manifest in (expected_manifest, actual_manifest):
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return False
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            return False
        if manifest.get("identity_sha256") != _canonical_digest(identity):
            return False
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            return False
        snapshot = provenance.get("dataset_snapshot")
        if not isinstance(snapshot, dict):
            return False
        if not isinstance(snapshot.get("images"), dict):
            return False
        if not isinstance(snapshot.get("artifacts"), list):
            return False
        if not isinstance(snapshot.get("reconstruction_stats"), dict):
            return False
        try:
            uuid.UUID(str(snapshot.get("instance_id")))
        except ValueError:
            return False

    if (
        expected_manifest["identity_sha256"]
        != actual_manifest["identity_sha256"]
    ):
        return False

    expected_snapshot = expected_manifest["provenance"]["dataset_snapshot"]
    actual_snapshot = actual_manifest["provenance"]["dataset_snapshot"]
    if strict_snapshot and (
        expected_snapshot["instance_id"] != actual_snapshot["instance_id"]
    ):
        return False
    if expected_snapshot["instance_id"] == actual_snapshot["instance_id"]:
        return _canonical_digest(expected_snapshot) == _canonical_digest(
            actual_snapshot
        )
    return True


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build one manifest.")
    build.add_argument("--scene", required=True)
    build.add_argument("--dataset", required=True, type=Path)
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--gsplat-root", required=True, type=Path)
    build.add_argument("--sample-fps", required=True, type=float)
    build.add_argument("--max-frames", required=True, type=int)
    build.add_argument("--overlap", required=True, type=int)
    build.add_argument("--min-registration-ratio", required=True, type=float)
    build.add_argument("--max-steps", required=True, type=int)
    build.add_argument("--camera-model", required=True)
    build.add_argument("--colmap-threads", required=True, type=int)
    build.add_argument("--gpu-id", required=True)
    build.add_argument("--output", required=True, type=Path)

    compare = commands.add_parser(
        "compare",
        help="Compare only the semantic identities of two manifests.",
    )
    compare.add_argument("expected", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("--strict-snapshot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compare":
            return (
                0
                if manifest_identity_matches(
                    args.expected,
                    args.actual,
                    args.strict_snapshot,
                )
                else 1
            )

        revision = _clean_git_revision(args.gsplat_root)
        manifest = build_manifest(
            args.scene,
            args.dataset,
            args.source,
            args.sample_fps,
            args.max_frames,
            args.overlap,
            args.min_registration_ratio,
            args.max_steps,
            args.camera_model,
            revision,
            args.colmap_threads,
            args.gpu_id,
        )
        _atomic_write_json(args.output.expanduser().resolve(), manifest)
    except (
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
