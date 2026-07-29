"""Uniformly sample a video into the images/ layout expected by COLMAP."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


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


def extract_frames(
    video_path: Path,
    images_dir: Path,
    sample_fps: float,
    max_frames: int,
    jpeg_quality: int,
) -> VideoSample:
    """Decode the video once and save all uniformly selected frames."""
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

    selected = select_frame_indices(total_frames, source_fps, sample_fps, max_frames)
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

    return VideoSample(
        source_fps=source_fps,
        total_source_frames=total_frames,
        duration_seconds=(total_frames - 1) / source_fps,
        selected_indices=selected,
        selected_timestamps_seconds=[index / source_fps for index in selected],
        source_frame_width=source_width,
        source_frame_height=source_height,
        output_frame_width=output_width,
        output_frame_height=output_height,
    )


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
        "--sample-fps",
        type=float,
        default=3.0,
        help="Maximum temporal sampling rate (default: 3).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Hard cap of at least 2 while spanning the full video (default: 1000).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="OpenCV JPEG quality from 1 to 100 (default: 95).",
    )
    return parser


def create_dataset(
    video: Path,
    dataset: Path,
    sample_fps: float,
    max_frames: int,
    jpeg_quality: int,
) -> Path:
    video = video.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
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

    dataset.parent.mkdir(parents=True, exist_ok=True)
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
        )
        metadata = {
            "source_video": str(video),
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "jpeg_quality": jpeg_quality,
            **asdict(sample),
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        create_dataset(
            args.video,
            args.dataset,
            args.sample_fps,
            args.max_frames,
            args.jpeg_quality,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
