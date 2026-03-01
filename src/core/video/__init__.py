"""Video I/O module.

Provides video reading and processing utilities including rotation detection
and correction for webcam and Teams recordings.
"""

from .media_stream import (
    MediaStream,
    read_video_rgb,
    probe_video_rotation,
    detect_frame_rotation,
    apply_rotation,
)

__all__ = [
    "MediaStream",
    "read_video_rgb",
    "probe_video_rotation",
    "detect_frame_rotation",
    "apply_rotation",
]
