#!/usr/bin/env python3
"""Build the provenance manifest that binds a gsplat result to its inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source capture not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
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


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_manifest(
    scene: str,
    dataset: Path,
    source: Path,
    pipeline_files: Sequence[Path],
    sample_fps: float,
    max_frames: int,
    overlap: int,
    min_registration_ratio: float,
    max_steps: int,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Training dataset not found: {dataset}")
    dataset_files = [
        dataset / "metadata.json",
        dataset / "reconstruction_stats.json",
    ]
    dataset_files.extend(
        dataset / "sparse" / "0" / name
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    )
    project_root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "scene": scene,
        "source": _source_identity(source),
        "parameters": {
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "overlap": overlap,
            "min_registration_ratio": min_registration_ratio,
            "max_steps": max_steps,
        },
        "dataset_artifacts": [
            _artifact_identity(path, dataset) for path in dataset_files
        ],
        "pipeline_files": [
            _artifact_identity(path, project_root) for path in pipeline_files
        ],
        "environment": {
            "pycolmap": _package_version("pycolmap"),
            "torch": _package_version("torch"),
            "gsplat": _package_version("gsplat"),
        },
    }


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
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--pipeline-file",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--sample-fps", required=True, type=float)
    parser.add_argument("--max-frames", required=True, type=int)
    parser.add_argument("--overlap", required=True, type=int)
    parser.add_argument("--min-registration-ratio", required=True, type=float)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_manifest(
            args.scene,
            args.dataset,
            args.source,
            args.pipeline_file,
            args.sample_fps,
            args.max_frames,
            args.overlap,
            args.min_registration_ratio,
            args.max_steps,
        )
        _atomic_write_json(args.output.expanduser().resolve(), manifest)
    except (OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
