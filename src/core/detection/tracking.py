"""Tracking and smoothing utilities for pose detection.

Provides IOU-based tracking and OneEuro temporal smoothing,
shared between BBoxDetector and other detection backends.
"""

from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """OneEuro filter for smoothing noisy signals.

    Combines low-pass filtering with adaptive cutoff based on signal speed.
    Great for reducing jitter in pose keypoints while preserving fast movements.

    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class SimpleIOUTracker:
    """Simple IOU-based tracker for single person tracking.

    Tracks the primary person across frames using bounding box IOU matching.
    """

    def __init__(self, max_age: int = 30, min_iou: float = 0.3, confirm_hits: int = 3):
        self.max_age = max_age
        self.min_iou = min_iou
        self.confirm_hits = confirm_hits
        self.tracks = {}
        self.next_id = 0

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def update(self, detections: list) -> list:
        """Update tracks with new detections.

        Args:
            detections: List of (bbox, keypoints, scores) tuples.

        Returns:
            List of (track_id, bbox, keypoints, scores) for matched tracks.
        """
        results = []

        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]

        matched_tracks = set()
        matched_dets = set()

        for det_idx, (bbox, kp, sc) in enumerate(detections):
            best_iou = self.min_iou
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._iou(bbox, track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid]['bbox'] = bbox
                self.tracks[best_tid]['age'] = 0
                self.tracks[best_tid]['hits'] += 1
                matched_tracks.add(best_tid)
                matched_dets.add(det_idx)

                if self.tracks[best_tid]['hits'] >= self.confirm_hits:
                    results.append((best_tid, bbox, kp, sc))

        for det_idx, (bbox, kp, sc) in enumerate(detections):
            if det_idx not in matched_dets:
                self.tracks[self.next_id] = {'bbox': bbox, 'age': 0, 'hits': 1}
                self.next_id += 1

        return results


# YOLOX model configurations (mode parameter for rtmlib BodyWithFeet)
RTMPOSE_MODELS = {
    "s": {
        "mode": "lightweight",
        "desc": "Small - fastest, lower accuracy (uses lightweight mode)",
    },
    "m": {
        "mode": "balanced",
        "desc": "Medium - balanced speed/accuracy (recommended)",
    },
    "l": {
        "mode": "performance",
        "desc": "Large - highest accuracy, slower",
    },
}
