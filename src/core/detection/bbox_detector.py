"""YOLOX bounding box detector for SAM 3D pipeline.

Loads only the YOLOX person detector (skipping RTMPose pose estimation)
since SAM 3D only needs bounding boxes. ~2x faster than full RTMPose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .tracking import (
    SimpleIOUTracker,
    OneEuroFilter,
    RTMPOSE_MODELS,
)


@dataclass
class PoseDetectionResult:
    """Result from a pose detection run."""
    keypoints_2d: np.ndarray           # (N, K, 2) pixel coordinates
    keypoints_3d: np.ndarray | None    # (N, K, 3) or None
    visibility: np.ndarray             # (N, K)
    timestamps: np.ndarray             # (N,)
    image_size: tuple[int, int]        # (height, width)
    num_keypoints: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BBoxDetector:
    """YOLOX-only person detector for pipelines that only need bounding boxes.

    Reuses the same YOLOX models as BodyWithFeet but skips pose estimation.
    Includes IOU tracking and OneEuro smoothing on box coordinates.
    """

    def __init__(
        self,
        model_size: str = "m",
        device: str = "cuda",
        backend: str = "onnxruntime",
        use_tracking: bool = True,
        use_smoothing: bool = True,
        smoothing_min_cutoff: float = 10.0,
        smoothing_beta: float = 2.0,
        smoothing_d_cutoff: float = 1.0,
        tracking_max_age: int = 30,
        tracking_min_iou: float = 0.3,
        tracking_confirm_hits: int = 3,
    ):
        if model_size not in RTMPOSE_MODELS:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {list(RTMPOSE_MODELS.keys())}"
            )

        self.model_size = model_size
        self.device = device
        self.backend = backend
        self.use_tracking = use_tracking
        self.use_smoothing = use_smoothing
        self.smoothing_min_cutoff = smoothing_min_cutoff
        self.smoothing_beta = smoothing_beta
        self.smoothing_d_cutoff = smoothing_d_cutoff
        self.tracking_max_age = tracking_max_age
        self.tracking_min_iou = tracking_min_iou
        self.tracking_confirm_hits = tracking_confirm_hits
        self._det_model = None
        self._tracker = None

    def _ensure_model_loaded(self):
        """Lazy-load YOLOX detector only (no pose model)."""
        if self._det_model is not None:
            return

        from rtmlib import YOLOX, BodyWithFeet

        mode = RTMPOSE_MODELS[self.model_size]["mode"]
        mode_cfg = BodyWithFeet.MODE[mode]

        self._det_model = YOLOX(
            mode_cfg["det"],
            model_input_size=mode_cfg["det_input_size"],
            backend=self.backend,
            device=self.device,
        )
        print(f"[BBoxDetector] Loaded YOLOX (mode={mode}, det-only)")

    @property
    def name(self) -> str:
        return f"bbox-{self.model_size}"

    @property
    def provides_3d(self) -> bool:
        return False

    def detect(
        self,
        frames: np.ndarray,
        fps: float,
        visibility_min: float = 0.3,
    ) -> PoseDetectionResult:
        """Run YOLOX person detection on video frames.

        Args:
            frames: (N, H, W, 3) RGB frames.
            fps: Frame rate for timestamps.
            visibility_min: Not used (no keypoints), kept for API compat.

        Returns:
            PoseDetectionResult with bounding boxes in metadata.
            keypoints_2d is zeroed placeholder (1 keypoint).
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected (N, H, W, 3) frames, got {frames.shape}")

        self._ensure_model_loaded()

        n_frames, height, width, _ = frames.shape

        # Placeholder keypoints (1 dummy keypoint per frame)
        keypoints_2d = np.zeros((n_frames, 1, 2), dtype=np.float32)
        visibility = np.zeros((n_frames, 1), dtype=np.float32)
        timestamps = np.arange(n_frames, dtype=np.float32) / fps
        bboxes = np.zeros((n_frames, 4), dtype=np.float32)

        # Tracker for single-person selection
        tracker = None
        if self.use_tracking:
            tracker = SimpleIOUTracker(
                max_age=self.tracking_max_age,
                min_iou=self.tracking_min_iou,
                confirm_hits=self.tracking_confirm_hits,
            )

        primary_track_id = None
        detected_frames = []  # (frame_idx, bbox) for interpolation

        for idx in range(n_frames):
            frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
            det_boxes = self._det_model(frame_bgr)  # (P, 4) xyxy

            if len(det_boxes) == 0:
                continue

            if tracker is not None:
                # Wrap boxes as (bbox, kp_placeholder, sc_placeholder) for tracker API
                detections = [
                    (list(det_boxes[i]), np.zeros((1, 2)), np.zeros(1))
                    for i in range(len(det_boxes))
                ]
                tracks = tracker.update(detections)

                if tracks:
                    for track_id, bbox, _, _ in tracks:
                        if primary_track_id is None:
                            primary_track_id = track_id
                        if track_id == primary_track_id:
                            bboxes[idx] = bbox
                            detected_frames.append((idx, np.array(bbox)))
                            break
                elif primary_track_id is None:
                    # No confirmed tracks yet — use largest box
                    areas = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
                    best = int(np.argmax(areas))
                    bboxes[idx] = det_boxes[best]
                    detected_frames.append((idx, det_boxes[best].copy()))
            else:
                # No tracking — pick largest box
                areas = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
                best = int(np.argmax(areas))
                bboxes[idx] = det_boxes[best]
                detected_frames.append((idx, det_boxes[best].copy()))

        # Interpolate gaps between detected frames
        if len(detected_frames) >= 2:
            bboxes = self._interpolate_boxes(bboxes, detected_frames, n_frames)

        # Smooth box coordinates
        if self.use_smoothing and len(detected_frames) > 0:
            bboxes = self._smooth_boxes(bboxes, timestamps)

        print(f"[BBoxDetector] {len(detected_frames)}/{n_frames} frames detected")

        return PoseDetectionResult(
            keypoints_2d=keypoints_2d,
            keypoints_3d=None,
            visibility=visibility,
            timestamps=timestamps,
            image_size=(height, width),
            num_keypoints=1,
            metadata={
                "estimator": "bbox",
                "model_size": self.model_size,
                "backend": self.backend,
                "tracking": self.use_tracking,
                "primary_track_id": primary_track_id,
                "bounding_boxes": bboxes,
            },
        )

    def _interpolate_boxes(
        self,
        bboxes: np.ndarray,
        detected_frames: list[tuple[int, np.ndarray]],
        n_frames: int,
    ) -> np.ndarray:
        """Fill gaps between detected frames with linear interpolation."""
        det_indices = [d[0] for d in detected_frames]

        for i in range(len(det_indices) - 1):
            start = det_indices[i]
            end = det_indices[i + 1]
            if end - start <= 1:
                continue
            for gap in range(start + 1, end):
                t = (gap - start) / (end - start)
                bboxes[gap] = bboxes[start] * (1 - t) + bboxes[end] * t

        return bboxes

    def _smooth_boxes(
        self,
        bboxes: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """Apply OneEuro filter to box coordinates for temporal smoothing."""
        filters = [
            OneEuroFilter(self.smoothing_min_cutoff, self.smoothing_beta, self.smoothing_d_cutoff)
            for _ in range(4)
        ]
        smoothed = bboxes.copy()

        for idx in range(len(bboxes)):
            if np.all(bboxes[idx] == 0):
                continue
            t = float(timestamps[idx])
            for c in range(4):
                smoothed[idx, c] = filters[c](bboxes[idx, c], t)

        return smoothed
