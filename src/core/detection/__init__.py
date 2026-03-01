"""Pose detection module.

Provides BBoxDetector (YOLOX-only) for person detection.
"""


def __getattr__(name):
    """Lazy import for detector classes."""
    if name == "BBoxDetector":
        from .bbox_detector import BBoxDetector
        return BBoxDetector
    if name == "PoseDetectionResult":
        from .bbox_detector import PoseDetectionResult
        return PoseDetectionResult
    if name in ("SimpleIOUTracker", "OneEuroFilter", "RTMPOSE_MODELS"):
        from . import tracking
        return getattr(tracking, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BBoxDetector",
    "PoseDetectionResult",
    "SimpleIOUTracker",
    "OneEuroFilter",
    "RTMPOSE_MODELS",
]
