"""Coordinate system transformations between pipeline conventions.

Conventions:
  Pipeline: X=forward, Y=up, Z=right (meters)
  Camera:   X=right, Y=down, Z=forward (millimeters)
"""

from __future__ import annotations

import numpy as np


# ── Rotation matrices for coordinate system transforms ──
# These are used across conversion modules (MHR markers, SAM 3D angles, etc.)

# MHR body-centric (X=right, Y=up, Z=backward) → Pipeline (X=forward, Y=up, Z=right)
MHR_BODY_TO_PIPELINE = np.array([
    [0,  0, -1],   # X_pipe = -Z_mhr (forward = -backward)
    [0,  1,  0],   # Y_pipe =  Y_mhr (up = up)
    [1,  0,  0],   # Z_pipe =  X_mhr (right = right)
], dtype=np.float64)

# Camera (X=right, Y=down, Z=forward) → Pipeline (X=forward, Y=up, Z=right)
# Used for SAM 3D camera-space outputs (joint_coords, keypoints_3d after MHRHead Y/Z flip)
CAMERA_TO_PIPELINE = np.array([
    [0,  0, 1],    # X_pipe = Z_cam (forward)
    [0, -1, 0],    # Y_pipe = -Y_cam (up = -down)
    [1,  0, 0],    # Z_pipe = X_cam (right)
], dtype=np.float64)


def camera_to_pipeline(coords_mm: np.ndarray) -> np.ndarray:
    """Camera(X=right, Y=down, Z=fwd) mm → Pipeline(X=fwd, Y=up, Z=right) m.

    Mapping: X_pipe = Z_cam, Y_pipe = -Y_cam, Z_pipe = X_cam.
    Also converts millimeters to meters.
    """
    out = np.empty_like(coords_mm, dtype=np.float64)
    out[..., 0] = coords_mm[..., 2] / 1000.0   # X_pipe(fwd)   = Z_cam(fwd) / 1000
    out[..., 1] = -coords_mm[..., 1] / 1000.0  # Y_pipe(up)    = -Y_cam(down) / 1000
    out[..., 2] = coords_mm[..., 0] / 1000.0   # Z_pipe(right) = X_cam(right) / 1000
    return out


def pipeline_to_camera(coords_m: np.ndarray) -> np.ndarray:
    """Pipeline(X=fwd, Y=up, Z=right) m → Camera(X=right, Y=down, Z=fwd) mm.

    Inverse of camera_to_pipeline. Converts meters to millimeters.
    """
    out = np.empty_like(coords_m)
    out[..., 0] = coords_m[..., 2] * 1000.0   # X_cam(right) = Z_pipe(right) * 1000
    out[..., 1] = -coords_m[..., 1] * 1000.0  # Y_cam(down)  = -Y_pipe(up) * 1000
    out[..., 2] = coords_m[..., 0] * 1000.0   # Z_cam(fwd)   = X_pipe(fwd) * 1000
    return out
