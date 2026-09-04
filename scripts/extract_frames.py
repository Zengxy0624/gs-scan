"""Uniformly sample a video into the images/ layout expected by COLMAP.

With ``--sharpness-window SECONDS`` every uniformly spaced anchor frame is
checked against its neighbors within +/-SECONDS (variance of the Laplacian of
the grayscale frame). An anchor that is blurry relative to the sharpest frame
in its window (below ``--sharpness-keep-ratio`` times that maximum) is swapped
for that frame, so motion blur is handled at extraction time instead of being
filtered out later, while anchors that are already sharp keep their uniform
timing. Selections stay strictly increasing and unique, so the dataset still
holds exactly one image per anchor. The video is decoded twice in that mode: once
to score every frame, once to write the winners. Two CSVs are written beside
``metadata.json``: ``sharpness_scores.csv`` (selected frames; ``name``,
``raw``, ``ratio_local`` columns compatible with filter_sharp_dataset.py) and
``sharpness_all_frames.csv`` (every source frame).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_SHARPNESS_WORKERS = 8
DEFAULT_SHARPNESS_KEEP_RATIO = 0.8
SHARPNESS_RATIO_NEIGHBORS = 10


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


@dataclass(frozen=True)
class VideoSample:
    source_fps: float
    total_source_frames: int
    duration_seconds: float
    selected_indices: list[int]
    selected_timestamps_seconds: list[float]
    source_frame_width: int
    source_frame_height: int
    output_frame_width: int
    output_frame_height: int
    sharpness_window_seconds: float = 0.0
    sharpness_window_frames: int = 0
    sharpness_keep_ratio: float = 0.0
    anchor_indices: list[int] | None = None
    anchor_sharpness: list[float] | None = None
    selected_sharpness: list[float] | None = None


def _sample_metadata(sample: VideoSample) -> dict[str, object]:
    """Serialize a sample, omitting fields unused by this extraction mode."""
    return {key: value for key, value in asdict(sample).items() if value is not None}


def select_frame_indices(
    total_frames: int,
    source_fps: float,
    sample_fps: float,
    max_frames: int,
) -> list[int]:
    """Return uniform indices spanning the video, constrained by rate and count."""
    if total_frames < 1:
        raise ValueError("total_frames must be positive")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")

    duration_seconds = (total_frames - 1) / source_fps
    rate_limited_count = max(2, math.floor(duration_seconds * sample_fps) + 1)
    count = min(total_frames, max_frames, rate_limited_count)
    if count == 1:
        return [0]

    last = total_frames - 1
    return [round(position * last / (count - 1)) for position in range(count)]


def select_sharp_indices(
    anchors: Sequence[int],
    sharpness: Sequence[float],
    window_frames: int,
    keep_ratio: float = DEFAULT_SHARPNESS_KEEP_RATIO,
) -> list[int]:
    """Swap blurry anchors for the sharpest frame within ``window_frames``.

    An anchor is kept when it is still selectable and its sharpness is at
    least ``keep_ratio`` times the best sharpness in its window (``1`` always
    takes the sharpest frame, ``0`` never replaces an anchor). Anchors are processed left to right. Each pick
    must lie strictly after the previous pick and must leave one free frame
    for every later anchor, so the result is strictly increasing, unique, and
    exactly as long as ``anchors``. Ties prefer the frame closest to the
    anchor, then the earlier frame.
    """
    total = len(sharpness)
    if window_frames < 0:
        raise ValueError("window_frames must be non-negative")
    if not 0 <= keep_ratio <= 1:
        raise ValueError("keep_ratio must be between 0 and 1")
    if not anchors:
        raise ValueError("at least one anchor is required")
    if list(anchors) != sorted(set(anchors)):
        raise ValueError("anchors must be strictly increasing")
    if anchors[0] < 0 or anchors[-1] >= total:
        raise ValueError("anchors must index into the sharpness sequence")

    chosen: list[int] = []
    previous = -1
    count = len(anchors)
    for position, anchor in enumerate(anchors):
        remaining_after = count - 1 - position
        low = max(anchor - window_frames, previous + 1)
        high = min(anchor + window_frames, total - 1 - remaining_after)
        if low > high:  # unreachable for valid input; guard against regressions
            raise RuntimeError(
                f"empty selection window for anchor {anchor}: [{low}, {high}]"
            )
        best = max(
            range(low, high + 1),
            key=lambda index: (sharpness[index], -abs(index - anchor), -index),
        )
        if low <= anchor <= high and sharpness[anchor] >= keep_ratio * sharpness[best]:
            best = anchor
        chosen.append(best)
        previous = best
    return chosen


def frame_sharpness(frame: object) -> float:
    """Variance of the Laplacian of the grayscale frame (higher = sharper)."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _open_capture(video_path: Path) -> tuple[object, float, int, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(source_fps) or source_fps <= 0 or total_frames < 2:
        capture.release()
        raise RuntimeError(
            "The decoder did not report valid FPS metadata and at least two "
            "frames; transcode the video to H.264 MP4 and retry."
        )
    return capture, source_fps, total_frames, source_width, source_height


def measure_frame_sharpness(
    video_path: Path,
    expected_total_frames: int,
    workers: int = DEFAULT_SHARPNESS_WORKERS,
) -> list[float]:
    """Decode the whole video once and score every frame's sharpness."""
    if workers < 1:
        raise ValueError("workers must be positive")
    capture, _, _, _, _ = _open_capture(video_path)
    scores: list[float] = []
    pending: deque[Future[float]] = deque()
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                pending.append(pool.submit(frame_sharpness, frame))
                # Bound the decoded frames held in memory awaiting a worker.
                while len(pending) >= 4 * workers:
                    scores.append(pending.popleft().result())
                if len(scores) and len(scores) % 5000 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"  scored {len(scores)} frames "
                        f"({len(scores) / elapsed:.0f} frames/s)",
                        flush=True,
                    )
            while pending:
                scores.append(pending.popleft().result())
    finally:
        capture.release()

    if len(scores) != expected_total_frames:
        raise RuntimeError(
            "Video decoding stopped early during sharpness scoring: "
            f"expected {expected_total_frames} frames, decoded {len(scores)}."
        )
    return scores


def local_sharpness_ratios(
    raw: Sequence[float],
    neighbors: int = SHARPNESS_RATIO_NEIGHBORS,
) -> list[float]:
    """raw / rolling median over +/-neighbors positions (1.0 when undefined)."""
    ratios: list[float] = []
    for index, value in enumerate(raw):
        window = raw[max(0, index - neighbors) : index + neighbors + 1]
        median = statistics.median(window)
        ratios.append(value / median if median > 0 else 1.0)
    return ratios


def write_sharpness_reports(
    directory: Path,
    prefix: str,
    sample: VideoSample,
    all_frame_sharpness: Sequence[float],
) -> tuple[Path, Path]:
    """Write the selected-frame score CSV and the all-frames sharpness CSV."""
    if (
        sample.anchor_indices is None
        or sample.anchor_sharpness is None
        or sample.selected_sharpness is None
    ):
        raise ValueError("sample has no sharpness selection to report")
    scores_path = directory / f"{prefix}sharpness_scores.csv"
    ratios = local_sharpness_ratios(sample.selected_sharpness)
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "raw",
                "ratio_local",
                "source_index",
                "anchor_index",
                "shift_frames",
                "timestamp_seconds",
                "anchor_raw",
            ]
        )
        for position, selected in enumerate(sample.selected_indices):
            anchor = sample.anchor_indices[position]
            writer.writerow(
                [
                    f"{prefix}{position:06d}.jpg",
                    f"{sample.selected_sharpness[position]:.4f}",
                    f"{ratios[position]:.4f}",
                    selected,
                    anchor,
                    selected - anchor,
                    f"{sample.selected_timestamps_seconds[position]:.4f}",
                    f"{sample.anchor_sharpness[position]:.4f}",
                ]
            )

    all_path = directory / f"{prefix}sharpness_all_frames.csv"
    anchors = set(sample.anchor_indices)
    selected_set = set(sample.selected_indices)
    with all_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source_index", "timestamp_seconds", "raw", "is_anchor", "is_selected"]
        )
        for index, value in enumerate(all_frame_sharpness):
            writer.writerow(
                [
                    index,
                    f"{index / sample.source_fps:.4f}",
                    f"{value:.4f}",
                    int(index in anchors),
                    int(index in selected_set),
                ]
            )
    return scores_path, all_path


def extract_frames(
    video_path: Path,
    images_dir: Path,
    sample_fps: float,
    max_frames: int,
    jpeg_quality: int,
    sharpness_window: float = 0.0,
    sharpness_workers: int = DEFAULT_SHARPNESS_WORKERS,
    sharpness_keep_ratio: float = DEFAULT_SHARPNESS_KEEP_RATIO,
    report_dir: Path | None = None,
    report_prefix: str = "",
) -> VideoSample:
    """Decode the video and save all selected frames.

    With ``sharpness_window > 0`` the video is decoded twice: a scoring pass
    over every frame, then a writing pass for the chosen frames. Reports are
    written to ``report_dir`` (default: the parent of ``images_dir``).
    """
    import cv2

    capture, source_fps, total_frames, source_width, source_height = _open_capture(
        video_path
    )
    capture.release()

    anchors = select_frame_indices(total_frames, source_fps, sample_fps, max_frames)
    window_frames = 0
    all_sharpness: list[float] | None = None
    if sharpness_window > 0:
        window_frames = int(round(sharpness_window * source_fps))
        print(
            f"Scoring sharpness of {total_frames} frames "
            f"(window +/-{window_frames} frames = {sharpness_window} s)",
            flush=True,
        )
        all_sharpness = measure_frame_sharpness(
            video_path,
            total_frames,
            sharpness_workers,
        )
        selected = select_sharp_indices(
            anchors,
            all_sharpness,
            window_frames,
            sharpness_keep_ratio,
        )
        moved = sum(1 for a, s in zip(anchors, selected, strict=True) if a != s)
        print(
            f"Selected {len(selected)} frames; {moved} anchors below "
            f"{sharpness_keep_ratio:g}x of their window maximum moved to a "
            "sharper neighbor",
            flush=True,
        )
    else:
        selected = anchors

    capture, _, _, _, _ = _open_capture(video_path)
    images_dir.mkdir(parents=True, exist_ok=False)
    selected_cursor = 0
    source_index = 0
    output_width = 0
    output_height = 0

    try:
        while selected_cursor < len(selected):
            ok, frame = capture.read()
            if not ok:
                break
            if source_index == selected[selected_cursor]:
                output_height, output_width = frame.shape[:2]
                target = images_dir / f"{selected_cursor:06d}.jpg"
                saved = cv2.imwrite(
                    str(target),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not saved:
                    raise RuntimeError(f"Failed to write extracted frame: {target}")
                selected_cursor += 1
            source_index += 1
    finally:
        capture.release()

    if selected_cursor != len(selected):
        raise RuntimeError(
            "Video decoding stopped early: "
            f"expected {len(selected)} sampled frames, wrote {selected_cursor}."
        )

    sample = VideoSample(
        source_fps=source_fps,
        total_source_frames=total_frames,
        duration_seconds=(total_frames - 1) / source_fps,
        selected_indices=list(selected),
        selected_timestamps_seconds=[index / source_fps for index in selected],
        source_frame_width=source_width,
        source_frame_height=source_height,
        output_frame_width=output_width,
        output_frame_height=output_height,
        sharpness_window_seconds=float(sharpness_window),
        sharpness_window_frames=window_frames,
        sharpness_keep_ratio=(
            float(sharpness_keep_ratio) if all_sharpness is not None else 0.0
        ),
        anchor_indices=list(anchors) if all_sharpness is not None else None,
        anchor_sharpness=(
            [all_sharpness[index] for index in anchors]
            if all_sharpness is not None
            else None
        ),
        selected_sharpness=(
            [all_sharpness[index] for index in selected]
            if all_sharpness is not None
            else None
        ),
    )
    if all_sharpness is not None:
        write_sharpness_reports(
            report_dir if report_dir is not None else images_dir.parent,
            report_prefix,
            sample,
            all_sharpness,
        )
    return sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uniformly sample a video into a fresh gsplat dataset."
    )
    parser.add_argument("video", type=Path, help="Source video path.")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Fresh output directory; images/ and metadata.json are created here.",
    )
    parser.add_argument(
        "--additional-video",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional source video. Repeat to build one ordered multi-video "
            "dataset; --max-frames applies separately to every source."
        ),
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=3.0,
        help="Maximum temporal sampling rate (default: 3).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100000,
        help=(
            "Hard cap of at least 2 while spanning the full video "
            "(default: 100000)."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="OpenCV JPEG quality from 1 to 100 (default: 95).",
    )
    parser.add_argument(
        "--sharpness-window",
        type=float,
        default=0.0,
        help=(
            "Replace each uniformly sampled anchor frame by the sharpest frame "
            "within this many seconds of it (variance of Laplacian). 0 keeps "
            "the exact uniform anchors (default: 0)."
        ),
    )
    parser.add_argument(
        "--sharpness-keep-ratio",
        type=float,
        default=DEFAULT_SHARPNESS_KEEP_RATIO,
        help=(
            "Keep an anchor whose sharpness is at least this fraction of the "
            "sharpest frame in its window; lower anchors are replaced. 1 always "
            "takes the sharpest frame, 0 never replaces (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--sharpness-workers",
        type=int,
        default=DEFAULT_SHARPNESS_WORKERS,
        help=(
            "Threads scoring frame sharpness while the video decodes "
            "(default: %(default)s)."
        ),
    )
    return parser


def _validate_creation_request(
    videos: Sequence[Path],
    dataset: Path,
    sample_fps: float,
    max_frames: int,
    jpeg_quality: int,
    sharpness_window: float = 0.0,
    sharpness_workers: int = DEFAULT_SHARPNESS_WORKERS,
    sharpness_keep_ratio: float = DEFAULT_SHARPNESS_KEEP_RATIO,
) -> None:
    if not videos:
        raise ValueError("at least one source video is required")
    for video in videos:
        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
    if dataset.exists():
        raise FileExistsError(
            f"Dataset already exists: {dataset}. Use a fresh versioned directory."
        )
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")
    if max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if not math.isfinite(sharpness_window) or sharpness_window < 0:
        raise ValueError("--sharpness-window must be zero or positive")
    if sharpness_workers < 1:
        raise ValueError("--sharpness-workers must be positive")
    if not (math.isfinite(sharpness_keep_ratio) and 0 <= sharpness_keep_ratio <= 1):
        raise ValueError("--sharpness-keep-ratio must be between 0 and 1")


def _source_prefix(index: int, video: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", video.stem.lower()).strip("_")
    return f"{index:02d}_{stem or 'video'}_"


def create_dataset(
    video: Path,
    dataset: Path,
    sample_fps: float,
    max_frames: int,
    jpeg_quality: int,
    sharpness_window: float = 0.0,
    sharpness_workers: int = DEFAULT_SHARPNESS_WORKERS,
    sharpness_keep_ratio: float = DEFAULT_SHARPNESS_KEEP_RATIO,
) -> Path:
    video = video.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
    _validate_creation_request(
        [video],
        dataset,
        sample_fps,
        max_frames,
        jpeg_quality,
        sharpness_window,
        sharpness_workers,
        sharpness_keep_ratio,
    )

    dataset.parent.mkdir(parents=True, exist_ok=True)
    source_stat = video.stat()
    print(f"Hashing source capture: {video}", flush=True)
    source_sha256 = _sha256(video)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{dataset.name}.", dir=str(dataset.parent))
    )
    try:
        sample = extract_frames(
            video,
            temporary / "images",
            sample_fps,
            max_frames,
            jpeg_quality,
            sharpness_window=sharpness_window,
            sharpness_workers=sharpness_workers,
            sharpness_keep_ratio=sharpness_keep_ratio,
            report_dir=temporary,
        )
        images_sha256 = _tree_sha256(temporary / "images")
        metadata = {
            "dataset_type": "single_video",
            "dataset_instance_id": str(uuid.uuid4()),
            "source_video": str(video),
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_sha256": source_sha256,
            "images_sha256": images_sha256,
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "jpeg_quality": jpeg_quality,
            **_sample_metadata(sample),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(dataset)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Wrote {len(sample.selected_indices)} frames to {dataset / 'images'}")
    print(f"Dataset metadata: {dataset / 'metadata.json'}")
    return dataset


def create_video_set_dataset(
    videos: Sequence[Path],
    dataset: Path,
    sample_fps: float,
    max_frames: int,
    jpeg_quality: int,
    sharpness_window: float = 0.0,
    sharpness_workers: int = DEFAULT_SHARPNESS_WORKERS,
    sharpness_keep_ratio: float = DEFAULT_SHARPNESS_KEEP_RATIO,
) -> Path:
    """Sample an ordered set of videos into one atomic COLMAP dataset."""
    resolved_videos = [video.expanduser().resolve() for video in videos]
    dataset = dataset.expanduser().resolve()
    if len(resolved_videos) < 2:
        raise ValueError("a video-set dataset requires at least two source videos")
    _validate_creation_request(
        resolved_videos,
        dataset,
        sample_fps,
        max_frames,
        jpeg_quality,
        sharpness_window,
        sharpness_workers,
        sharpness_keep_ratio,
    )

    source_identities = []
    for video in resolved_videos:
        source_stat = video.stat()
        print(f"Hashing source capture: {video}", flush=True)
        source_identities.append(
            {
                "source_video": str(video),
                "source_size_bytes": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_sha256": _sha256(video),
            }
        )

    dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{dataset.name}.", dir=str(dataset.parent))
    )
    try:
        images_dir = temporary / "images"
        images_dir.mkdir()
        source_records = []
        expected_dimensions: tuple[int, int] | None = None
        total_selected_frames = 0

        for index, (video, identity) in enumerate(
            zip(resolved_videos, source_identities, strict=True)
        ):
            image_prefix = _source_prefix(index, video)
            staging_dir = temporary / f".source-{index:02d}"
            sample = extract_frames(
                video,
                staging_dir,
                sample_fps,
                max_frames,
                jpeg_quality,
                sharpness_window=sharpness_window,
                sharpness_workers=sharpness_workers,
                sharpness_keep_ratio=sharpness_keep_ratio,
                report_dir=temporary,
                report_prefix=image_prefix,
            )
            dimensions = (
                sample.output_frame_width,
                sample.output_frame_height,
            )
            if expected_dimensions is None:
                expected_dimensions = dimensions
            elif dimensions != expected_dimensions:
                raise ValueError(
                    "All videos in one dataset must decode to the same output "
                    "dimensions for shared camera calibration: "
                    f"expected {expected_dimensions[0]}x{expected_dimensions[1]}, "
                    f"got {dimensions[0]}x{dimensions[1]} for {video}"
                )

            for frame_index in range(len(sample.selected_indices)):
                source_image = staging_dir / f"{frame_index:06d}.jpg"
                target_image = images_dir / f"{image_prefix}{frame_index:06d}.jpg"
                source_image.replace(target_image)
            staging_dir.rmdir()

            selected_count = len(sample.selected_indices)
            total_selected_frames += selected_count
            source_records.append(
                {
                    **identity,
                    "image_prefix": image_prefix,
                    **_sample_metadata(sample),
                }
            )

        if expected_dimensions is None:
            raise RuntimeError("No videos were extracted")
        images_sha256 = _tree_sha256(images_dir)
        metadata = {
            "dataset_type": "video_set",
            "dataset_instance_id": str(uuid.uuid4()),
            "sources": source_records,
            "total_selected_frames": total_selected_frames,
            "output_frame_width": expected_dimensions[0],
            "output_frame_height": expected_dimensions[1],
            "images_sha256": images_sha256,
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "jpeg_quality": jpeg_quality,
            "sharpness_window_seconds": float(sharpness_window),
            "sharpness_keep_ratio": (
                float(sharpness_keep_ratio) if sharpness_window > 0 else 0.0
            ),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(dataset)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Wrote {total_selected_frames} frames to {dataset / 'images'}")
    print(f"Dataset metadata: {dataset / 'metadata.json'}")
    return dataset


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.additional_video:
            create_video_set_dataset(
                [args.video, *args.additional_video],
                args.dataset,
                args.sample_fps,
                args.max_frames,
                args.jpeg_quality,
                sharpness_window=args.sharpness_window,
                sharpness_workers=args.sharpness_workers,
                sharpness_keep_ratio=args.sharpness_keep_ratio,
            )
        else:
            create_dataset(
                args.video,
                args.dataset,
                args.sample_fps,
                args.max_frames,
                args.jpeg_quality,
                sharpness_window=args.sharpness_window,
                sharpness_workers=args.sharpness_workers,
                sharpness_keep_ratio=args.sharpness_keep_ratio,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
