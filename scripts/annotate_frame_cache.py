#!/usr/bin/env python3
"""Embed per-video capture metadata from an annotation CSV into a frame cache.

Every image of a single-video frame cache receives EXIF tags built from the
annotation row of its source video (ISO, shutter, aperture, 35 mm-equivalent
focal length, white balance, per-frame capture time, GPS, description) plus
optional camera make/model/software, so each JPEG carries its own metadata
after caches are concatenated. The row is also stored under
``capture_metadata`` in ``metadata.json``, a per-image ``image_metadata.csv``
is written beside it, and ``images_sha256`` is recomputed. JPEG pixel data is
untouched: the EXIF segment is inserted losslessly.

The annotation CSV needs at least a ``file`` column (source video file name);
recognised optional columns are ``clip``, ``annotation``, ``lens`` (e.g.
``13mm``), ``rec_time`` (ISO 8601 with offset), ``timecode``, ``iso``,
``shutter`` (``1/120``), ``aperture`` (``f2.2``), ``white_balance_k``,
``tint``, ``color``, ``gps`` (ISO 6709, ``+40.11075-088.22833/``), ``notes``.

Example:
  python scripts/annotate_frame_cache.py work/jump_center/JC_01_1245 \\
      --annotation captures/jump_center/annotation.csv \\
      --make Apple --model "iPhone 17 Pro" --software "Blackmagic Cam 3.4"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

IMAGE_METADATA_COLUMNS = (
    "name",
    "source_video",
    "clip",
    "annotation",
    "lens_mm",
    "iso",
    "shutter",
    "aperture",
    "white_balance_k",
    "tint",
    "color",
    "capture_datetime",
    "source_frame_index",
    "timestamp_seconds",
    "sharpness_raw",
    "sharpness_ratio_local",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_lens_mm(text: str) -> float | None:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*mm\s*", text or "", re.I)
    if match is None:
        if (text or "").strip():
            raise ValueError(f"Unrecognised lens value: {text!r}")
        return None
    return float(match.group(1))


def parse_shutter(text: str) -> Fraction | None:
    text = (text or "").strip().rstrip("s")
    if not text:
        return None
    match = re.fullmatch(r"1\s*/\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return 1 / Fraction(match.group(1))
    try:
        return Fraction(text).limit_denominator(100000)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Unrecognised shutter value: {text!r}") from None


def parse_aperture(text: str) -> Fraction | None:
    match = re.fullmatch(r"\s*f?/?\s*([0-9]+(?:\.[0-9]+)?)\s*", text or "", re.I)
    if match is None:
        if (text or "").strip():
            raise ValueError(f"Unrecognised aperture value: {text!r}")
        return None
    return Fraction(match.group(1)).limit_denominator(100)


def parse_iso6709(text: str) -> tuple[float, float] | None:
    match = re.fullmatch(
        r"\s*([+-][0-9]+(?:\.[0-9]+)?)([+-][0-9]+(?:\.[0-9]+)?)(?:[+-][0-9.]+)?/?\s*",
        text or "",
    )
    if match is None:
        if (text or "").strip():
            raise ValueError(f"Unrecognised ISO 6709 GPS value: {text!r}")
        return None
    return float(match.group(1)), float(match.group(2))


def parse_int(text: str, label: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        raise ValueError(f"Unrecognised {label} value: {text!r}") from None


def parse_rec_time(text: str) -> datetime | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"Unrecognised rec_time value: {text!r}") from None


def _rational(value: Fraction) -> tuple[int, int]:
    return int(value.numerator), int(value.denominator)


def _dms(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    value = abs(value)
    degrees = int(value)
    minutes_full = (value - degrees) * 60
    minutes = int(minutes_full)
    seconds = round((minutes_full - minutes) * 60 * 10000)
    return (degrees, 1), (minutes, 1), (seconds, 10000)


def _exif_datetime(moment: datetime) -> tuple[str, str, str | None]:
    """EXIF DateTime string, sub-second string, and UTC offset (if known)."""
    stamp = moment.strftime("%Y:%m:%d %H:%M:%S")
    subsec = f"{moment.microsecond // 1000:03d}"
    offset = None
    if moment.utcoffset() is not None:
        total = int(moment.utcoffset().total_seconds())
        sign = "+" if total >= 0 else "-"
        total = abs(total)
        offset = f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"
    return stamp, subsec, offset


def load_annotation_row(annotation_csv: Path, video_name: str) -> dict[str, str]:
    if not annotation_csv.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {annotation_csv}")
    with annotation_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "file" not in rows[0]:
        raise ValueError(f"Annotation CSV needs a 'file' column: {annotation_csv}")
    matches = [row for row in rows if Path(row["file"]).name == video_name]
    if len(matches) != 1:
        raise ValueError(
            f"Annotation CSV has {len(matches)} rows for {video_name}: {annotation_csv}"
        )
    return {key: (value or "").strip() for key, value in matches[0].items() if key}


def parse_capture_metadata(
    row: dict[str, str],
    make: str | None,
    model: str | None,
    software: str | None,
) -> dict[str, Any]:
    shutter = parse_shutter(row.get("shutter", ""))
    aperture = parse_aperture(row.get("aperture", ""))
    gps = parse_iso6709(row.get("gps", ""))
    rec_time = parse_rec_time(row.get("rec_time", ""))
    return {
        "annotation_row": dict(row),
        "make": make,
        "model": model,
        "software": software,
        "lens_mm": parse_lens_mm(row.get("lens", "")),
        "iso": parse_int(row.get("iso", ""), "iso"),
        "shutter_seconds": float(shutter) if shutter is not None else None,
        "shutter_rational": _rational(shutter) if shutter is not None else None,
        "aperture_f": float(aperture) if aperture is not None else None,
        "white_balance_k": parse_int(row.get("white_balance_k", ""), "white_balance_k"),
        "tint": parse_int(row.get("tint", ""), "tint"),
        "gps_latitude": gps[0] if gps else None,
        "gps_longitude": gps[1] if gps else None,
        "rec_time_iso": rec_time.isoformat() if rec_time else None,
    }


def build_exif(
    capture: dict[str, Any],
    frame_time: datetime | None,
    comment: dict[str, Any],
) -> bytes:
    """Serialize EXIF bytes for one frame from parsed capture metadata."""
    import piexif

    zeroth: dict[int, Any] = {}
    exif: dict[int, Any] = {}
    gps: dict[int, Any] = {}
    row = capture["annotation_row"]
    if capture.get("make"):
        zeroth[piexif.ImageIFD.Make] = capture["make"]
    if capture.get("model"):
        zeroth[piexif.ImageIFD.Model] = capture["model"]
    if capture.get("software"):
        zeroth[piexif.ImageIFD.Software] = capture["software"]
    if row.get("annotation"):
        zeroth[piexif.ImageIFD.ImageDescription] = row["annotation"]
    if frame_time is not None:
        stamp, subsec, offset = _exif_datetime(frame_time)
        zeroth[piexif.ImageIFD.DateTime] = stamp
        exif[piexif.ExifIFD.DateTimeOriginal] = stamp
        exif[piexif.ExifIFD.DateTimeDigitized] = stamp
        exif[piexif.ExifIFD.SubSecTimeOriginal] = subsec
        exif[piexif.ExifIFD.SubSecTimeDigitized] = subsec
        if offset is not None and hasattr(piexif.ExifIFD, "OffsetTimeOriginal"):
            exif[piexif.ExifIFD.OffsetTimeOriginal] = offset
    if capture.get("shutter_rational") is not None:
        exif[piexif.ExifIFD.ExposureTime] = tuple(capture["shutter_rational"])
    if capture.get("aperture_f") is not None:
        exif[piexif.ExifIFD.FNumber] = _rational(
            Fraction(capture["aperture_f"]).limit_denominator(100)
        )
    if capture.get("iso") is not None:
        exif[piexif.ExifIFD.ISOSpeedRatings] = int(capture["iso"])
    if capture.get("lens_mm") is not None:
        # The annotation lens is the 35 mm-equivalent focal length; COLMAP reads
        # this tag first when choosing its focal-length prior.
        exif[piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(capture["lens_mm"]))
        lens_label = f"{capture['lens_mm']:g}mm"
        exif[piexif.ExifIFD.LensModel] = (
            f"{capture['model']} {lens_label}" if capture.get("model") else lens_label
        )
    if capture.get("white_balance_k") is not None:
        exif[piexif.ExifIFD.WhiteBalance] = 1  # manual / fixed
    exif[piexif.ExifIFD.UserComment] = b"ASCII\x00\x00\x00" + json.dumps(
        comment, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    if capture.get("gps_latitude") is not None:
        lat, lon = capture["gps_latitude"], capture["gps_longitude"]
        gps[piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
        gps[piexif.GPSIFD.GPSLatitudeRef] = "N" if lat >= 0 else "S"
        gps[piexif.GPSIFD.GPSLatitude] = _dms(lat)
        gps[piexif.GPSIFD.GPSLongitudeRef] = "E" if lon >= 0 else "W"
        gps[piexif.GPSIFD.GPSLongitude] = _dms(lon)
    return piexif.dump({"0th": zeroth, "Exif": exif, "GPS": gps})


def _load_sharpness(cache: Path) -> dict[str, dict[str, str]]:
    path = cache / "sharpness_scores.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def annotate_frame_cache(
    cache: Path,
    annotation_csv: Path,
    make: str | None = None,
    model: str | None = None,
    software: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Embed EXIF into every cache image and record the capture metadata."""
    import piexif

    cache = cache.expanduser().resolve()
    annotation_csv = annotation_csv.expanduser().resolve()
    metadata_path = cache / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Frame metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_type") != "single_video":
        raise ValueError(f"Expected a single_video frame cache: {cache}")
    selected = metadata.get("selected_indices")
    timestamps = metadata.get("selected_timestamps_seconds")
    if (
        not isinstance(selected, list)
        or not isinstance(timestamps, list)
        or len(selected) != len(timestamps)
        or len(selected) < 1
    ):
        raise ValueError(f"Cache has no valid selected frame list: {cache}")
    images_dir = cache / "images"
    names = [f"{index:06d}.jpg" for index in range(len(selected))]
    for name in names:
        if not (images_dir / name).is_file():
            raise FileNotFoundError(f"Cache image missing: {images_dir / name}")
    if not force and _tree_sha256(images_dir) != metadata.get("images_sha256"):
        raise ValueError(
            f"Cache images do not match their metadata (interrupted run?): {cache}; "
            "pass --force to re-annotate"
        )

    video_name = Path(str(metadata.get("source_video", ""))).name
    row = load_annotation_row(annotation_csv, video_name)
    # CLI values win; otherwise optional make/model/software columns of the CSV.
    capture = parse_capture_metadata(
        row,
        make or row.get("make") or None,
        model or row.get("model") or None,
        software or row.get("software") or None,
    )
    rec_time = parse_rec_time(row.get("rec_time", ""))
    sharpness = _load_sharpness(cache)

    csv_rows: list[dict[str, Any]] = []
    for name, source_index, timestamp in zip(names, selected, timestamps, strict=True):
        frame_time = (
            rec_time + timedelta(seconds=float(timestamp)) if rec_time else None
        )
        comment = {
            "source_video": video_name,
            "clip": row.get("clip"),
            "source_frame_index": int(source_index),
            "timestamp_seconds": round(float(timestamp), 4),
            "timecode": row.get("timecode"),
            "white_balance_k": capture["white_balance_k"],
            "tint": capture["tint"],
            "color": row.get("color"),
            "notes": row.get("notes"),
        }
        exif_bytes = build_exif(capture, frame_time, comment)
        image_path = images_dir / name
        staged = images_dir / f".{name}.exif.tmp"
        try:
            piexif.insert(exif_bytes, str(image_path), str(staged))
            os.replace(staged, image_path)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        score = sharpness.get(name, {})
        csv_rows.append(
            {
                "name": name,
                "source_video": video_name,
                "clip": row.get("clip", ""),
                "annotation": row.get("annotation", ""),
                "lens_mm": capture["lens_mm"] if capture["lens_mm"] is not None else "",
                "iso": capture["iso"] if capture["iso"] is not None else "",
                "shutter": row.get("shutter", ""),
                "aperture": row.get("aperture", ""),
                "white_balance_k": capture["white_balance_k"]
                if capture["white_balance_k"] is not None
                else "",
                "tint": capture["tint"] if capture["tint"] is not None else "",
                "color": row.get("color", ""),
                "capture_datetime": frame_time.isoformat(timespec="milliseconds")
                if frame_time
                else "",
                "source_frame_index": int(source_index),
                "timestamp_seconds": f"{float(timestamp):.4f}",
                "sharpness_raw": score.get("raw", ""),
                "sharpness_ratio_local": score.get("ratio_local", ""),
            }
        )

    staged_csv = Path(
        tempfile.mkstemp(prefix=".image_metadata.", suffix=".csv", dir=str(cache))[1]
    )
    with staged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(IMAGE_METADATA_COLUMNS))
        writer.writeheader()
        writer.writerows(csv_rows)
    os.replace(staged_csv, cache / "image_metadata.csv")

    metadata.update(
        {
            "dataset_instance_id": str(uuid.uuid4()),
            "images_sha256": _tree_sha256(images_dir),
            "exif_embedded": True,
            "capture_metadata": capture,
            "annotation_csv": str(annotation_csv),
            "annotation_csv_sha256": _sha256(annotation_csv),
        }
    )
    staged_meta = Path(
        tempfile.mkstemp(prefix=".metadata.", suffix=".json", dir=str(cache))[1]
    )
    staged_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(staged_meta, metadata_path)

    summary = {
        "cache": str(cache),
        "images": len(names),
        "video": video_name,
        "lens_mm": capture["lens_mm"],
        "iso": capture["iso"],
        "shutter": row.get("shutter"),
        "white_balance_k": capture["white_balance_k"],
    }
    print(
        f"Annotated {len(names)} images of {cache.name} from {video_name}: "
        f"lens {capture['lens_mm']} mm, ISO {capture['iso']}, shutter "
        f"{row.get('shutter')}, WB {capture['white_balance_k']} K"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache", type=Path, help="Single-video frame cache to annotate.")
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Annotation CSV with one row per source video ('file' column).",
    )
    parser.add_argument("--make", help="Camera make for EXIF (default: CSV 'make' column).")
    parser.add_argument("--model", help="Camera model for EXIF (default: CSV 'model' column).")
    parser.add_argument("--software", help="Recording software for EXIF (default: CSV 'software' column).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate even if the images no longer match metadata.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        annotate_frame_cache(
            args.cache,
            args.annotation,
            make=args.make,
            model=args.model,
            software=args.software,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
