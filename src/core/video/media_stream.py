from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Tuple

import cv2 as cv
import numpy as np


def read_video_rgb(video_path: Path) -> Tuple[np.ndarray, float]:
    """Load RGB frames deterministically so strict downstream steps retain true timing."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 0.0
    frames = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    stack = np.stack(frames)
    return stack, float(fps or 0.0)


def probe_video_rotation(video_path: Path, decoded_shape: tuple[int, ...] | None = None) -> int:
    """Return rotation in degrees (0/90/180/270) if metadata is available.

    Checks both stream_tags=rotate (older format) and display matrix side data
    (newer format used by Teams/webcam recordings).

    If decoded_shape (H, W, ...) is provided, compares against ffprobe coded
    dimensions to detect when OpenCV already applied the rotation during decode.
    In that case returns 0 (no additional rotation needed).
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0

    # First try stream_tags=rotate (older format)
    rotation = 0

    # First try stream_tags=rotate (older format)
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate",
        "-of", "default=nk=1:nw=1", str(video_path),
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
        r = int(output.decode("utf-8").strip()) % 360
        if r in {90, 180, 270}:
            rotation = r
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass

    # Try display matrix side data (newer format, e.g., Teams recordings)
    if rotation == 0:
        command = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream_side_data=rotation",
            "-of", "default=nk=1:nw=1", str(video_path),
        ]
        try:
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
            r = int(float(output.decode("utf-8").strip())) % 360
            if r in {90, 180, 270}:
                rotation = r
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass

    if rotation == 0:
        return 0

    # Check if OpenCV already applied the rotation during decode.
    # Compare decoded frame dimensions against ffprobe coded dimensions.
    # If they differ (width/height swapped), OpenCV handled it.
    if decoded_shape is not None and rotation in {90, 270}:
        decoded_h, decoded_w = decoded_shape[0], decoded_shape[1]
        command = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=coded_width,coded_height",
            "-of", "csv=p=0", str(video_path),
        ]
        try:
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
            parts = [x for x in output.decode().strip().split(",") if x]
            coded_w, coded_h = int(parts[0]), int(parts[1])
            # If decoded dims are swapped vs coded dims, rotation already applied
            if decoded_h == coded_w and decoded_w == coded_h:
                print(f"[media] Rotation {rotation}° already applied by decoder, skipping")
                return 0
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass

    return rotation


def detect_frame_rotation(frame_rgb: np.ndarray) -> int:
    """Attempt to infer rotation using OCR orientation detection."""
    try:
        import pytesseract
    except ImportError:
        return 0
    if not shutil.which("tesseract"):
        print("[media] tesseract not found; skipping OCR-based rotation detection")
        return 0
    try:
        from PIL import Image
    except ImportError:
        return 0

    height, width = frame_rgb.shape[:2]
    if max(height, width) > 720:
        scale = 720 / float(max(height, width))
        resized = cv.resize(
            frame_rgb, (int(width * scale), int(height * scale)), interpolation=cv.INTER_AREA
        )
    else:
        resized = frame_rgb

    try:
        osd = pytesseract.image_to_osd(Image.fromarray(resized), output_type=pytesseract.Output.DICT)
    except (pytesseract.TesseractError, ValueError, TypeError):
        return 0

    try:
        confidence = float(osd.get("orientation_conf", 0))
    except (TypeError, ValueError):
        confidence = 0
    if confidence < 10:
        return 0

    try:
        rotation = int(osd.get("rotate", 0))
    except (TypeError, ValueError):
        return 0

    rotation = rotation % 360
    if rotation in {0, 90, 180, 270}:
        return rotation
    return 0


def apply_rotation(frames: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation to video frames.

    Args:
        frames: (N, H, W, C) array of frames.
        rotation: Rotation in degrees (90, 180, or 270). 0 = no rotation.

    Returns:
        Rotated frames array with updated dimensions.
    """
    if rotation == 0:
        return frames

    # cv2.rotate codes for counter-clockwise rotation
    # (rotation metadata indicates how much CW rotation was applied during capture,
    # so we need to rotate CCW by the same amount to correct)
    if rotation == 90:
        code = cv.ROTATE_90_COUNTERCLOCKWISE
    elif rotation == 180:
        code = cv.ROTATE_180
    elif rotation == 270:
        code = cv.ROTATE_90_CLOCKWISE
    else:
        return frames

    rotated = [cv.rotate(frame, code) for frame in frames]
    return np.stack(rotated)


class MediaStream:
    """Keeps backward compatibility with earlier imperative usage."""

    def __init__(self):
        self.video_rgb: np.ndarray | None = None
        self.fps: float = 0.0

    def read_video(self, video_path: Path) -> np.ndarray:
        """Loads video and returns it in RGB format."""
        self.video_rgb, self.fps = read_video_rgb(video_path)
        return self.video_rgb
