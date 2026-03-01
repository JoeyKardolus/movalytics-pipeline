"""MHR70 keypoint → OpenSim marker mapping for SAM 3D Body.

Maps the 70 MHR70 surface keypoints from SAM 3D Body to OpenSim-compatible
marker names for use with the LaiUhlrich2022 scaled model.

Includes strategic surface markers for rotational DOF constraint:
- Olecranon/cubital fossa for elbow rotation
- Acromion for shoulder rotation
- Index3/MiddleTip for forearm pronation/supination
- Feet markers for ankle rotation

Reference: SAM3D-OpenSim (github.com/AitorIriondo/SAM3D-OpenSim)

Coordinate conventions:
  Camera (SAM 3D): X=right, Y=down, Z=forward (meters)
  OpenSim/Pipeline: X=forward, Y=up, Z=right (meters)
  TRC output: millimeters
"""

from __future__ import annotations

import numpy as np

from src.shared.constants import MHR70_KEYPOINTS
from src.shared.coordinate_transforms import CAMERA_TO_PIPELINE


# MHR70 keypoint names in index order (0-69)
# From SAM3D-OpenSim keypoint_converter.py
MHR70_NAMES: list[str] = [
    # Body (0-20)
    "nose",                        # 0
    "left_eye",                    # 1
    "right_eye",                   # 2
    "left_ear",                    # 3
    "right_ear",                   # 4
    "left_shoulder",               # 5
    "right_shoulder",              # 6
    "left_elbow",                  # 7
    "right_elbow",                 # 8
    "left_hip",                    # 9
    "right_hip",                   # 10
    "left_knee",                   # 11
    "right_knee",                  # 12
    "left_ankle",                  # 13
    "right_ankle",                 # 14
    "left_big_toe",                # 15
    "left_small_toe",              # 16
    "left_heel",                   # 17
    "right_big_toe",               # 18
    "right_small_toe",             # 19
    "right_heel",                  # 20
    # Right hand (21-41)
    "right_thumb_tip",             # 21
    "right_thumb_first_joint",     # 22
    "right_thumb_second_joint",    # 23
    "right_thumb_third_joint",     # 24
    "right_index_tip",             # 25
    "right_index_first_joint",     # 26
    "right_index_second_joint",    # 27
    "right_index_third_joint",     # 28
    "right_middle_tip",            # 29
    "right_middle_first_joint",    # 30
    "right_middle_second_joint",   # 31
    "right_middle_third_joint",    # 32
    "right_ring_tip",              # 33
    "right_ring_first_joint",      # 34
    "right_ring_second_joint",     # 35
    "right_ring_third_joint",      # 36
    "right_pinky_tip",             # 37
    "right_pinky_first_joint",     # 38
    "right_pinky_second_joint",    # 39
    "right_pinky_third_joint",     # 40
    "right_wrist",                 # 41
    # Left hand (42-62)
    "left_thumb_tip",              # 42
    "left_thumb_first_joint",      # 43
    "left_thumb_second_joint",     # 44
    "left_thumb_third_joint",      # 45
    "left_index_tip",              # 46
    "left_index_first_joint",      # 47
    "left_index_second_joint",     # 48
    "left_index_third_joint",      # 49
    "left_middle_tip",             # 50
    "left_middle_first_joint",     # 51
    "left_middle_second_joint",    # 52
    "left_middle_third_joint",     # 53
    "left_ring_tip",               # 54
    "left_ring_first_joint",       # 55
    "left_ring_second_joint",      # 56
    "left_ring_third_joint",       # 57
    "left_pinky_tip",              # 58
    "left_pinky_first_joint",      # 59
    "left_pinky_second_joint",     # 60
    "left_pinky_third_joint",      # 61
    "left_wrist",                  # 62
    # Extra anatomical (63-69)
    "left_olecranon",              # 63
    "right_olecranon",             # 64
    "left_cubital_fossa",          # 65
    "right_cubital_fossa",         # 66
    "left_acromion",               # 67
    "right_acromion",              # 68
    "neck",                        # 69
]


# MHR70 index → OpenSim/TRC marker name (alias from shared constants)
MHR70_TO_OPENSIM = MHR70_KEYPOINTS


# Camera→OpenSim/Pipeline rotation (same transform)
_CAMERA_TO_OPENSIM = CAMERA_TO_PIPELINE


def _estimate_lean_angle(kp: np.ndarray) -> float:
    """Estimate median forward lean angle from pelvis→acromion spine vector.

    Args:
        kp: (N, K, 3) keypoints in OpenSim coords (X=fwd, Y=up, Z=right).

    Returns:
        Lean angle in degrees (positive = forward lean).
    """
    # Indices: 9/10=hips, 67/68=acromions
    angles = []
    for i in range(kp.shape[0]):
        pelvis = (kp[i, 9] + kp[i, 10]) / 2
        thorax = (kp[i, 67] + kp[i, 68]) / 2
        spine = thorax - pelvis
        # Project onto XY sagittal plane (X=forward, Y=up)
        spine_xy = np.array([spine[0], spine[1]])
        if np.linalg.norm(spine_xy) < 0.01:
            continue
        cos_a = np.dot(spine_xy, [0.0, 1.0]) / np.linalg.norm(spine_xy)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        # Sign: positive X = forward lean
        if spine[0] > 0:
            angles.append(angle)
        else:
            angles.append(-angle)
    return float(np.median(angles)) if angles else 0.0


def _correct_forward_lean(kp: np.ndarray) -> np.ndarray:
    """Correct systematic forward/backward lean by rotating around Z (lateral).

    Estimates the median spine-vs-vertical angle across all frames and applies
    a single correction rotation centered on the pelvis.

    Args:
        kp: (N, K, 3) keypoints in OpenSim coords (X=fwd, Y=up, Z=right).

    Returns:
        Corrected keypoints (copy).
    """
    angle = _estimate_lean_angle(kp)
    if abs(angle) < 1.0:
        return kp

    # Rotation around Z axis (lateral axis in OpenSim) by -angle to correct
    rad = np.radians(-angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    # Rotates in XY plane (forward/up), Z unchanged
    rot = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1],
    ])

    corrected = kp.copy()
    for i in range(corrected.shape[0]):
        pelvis = (corrected[i, 9] + corrected[i, 10]) / 2
        corrected[i] = (corrected[i] - pelvis) @ rot.T + pelvis

    return corrected


def _center_at_pelvis(kp: np.ndarray) -> np.ndarray:
    """Center each frame on pelvis XZ, keep Y (vertical) from ground alignment.

    Args:
        kp: (N, K, 3) keypoints in OpenSim coords.

    Returns:
        Centered keypoints (modified in-place).
    """
    for i in range(kp.shape[0]):
        pelvis = (kp[i, 9] + kp[i, 10]) / 2
        kp[i, :, 0] -= pelvis[0]  # X (forward)
        kp[i, :, 2] -= pelvis[2]  # Z (right)
    return kp


def map_sam3d_to_trc(
    keypoints_3d: np.ndarray,
    cam_t: np.ndarray,
    subject_height: float = 1.75,
) -> tuple[np.ndarray, list[str]]:
    """Convert SAM 3D MHR70 keypoints to TRC-compatible marker data.

    Transforms camera-space keypoints to OpenSim coordinates (mm), scales
    to subject height, corrects forward lean, ground-aligns, and centers
    on pelvis.

    Args:
        keypoints_3d: (N, 70, 3) MHR70 keypoints in body-model-relative space.
        cam_t: (N, 3) camera translation in camera space (meters).
        subject_height: Subject height in meters for scaling.

    Returns:
        (data_mm, marker_names) tuple:
          data_mm: (N, M, 3) in millimeters, OpenSim coords (X=fwd, Y=up, Z=right)
          marker_names: list of M marker names matching TRC columns
    """
    n_frames = keypoints_3d.shape[0]

    # Step 1: Transform body-relative → OpenSim coordinates
    # Body-relative coords (camera convention: X=right, Y=down, Z=forward)
    # → OpenSim: X=forward, Y=up, Z=right
    # NOTE: cam_t NOT added — body-relative coords are much smoother than
    # camera-space (hip jitter 3mm vs 26mm/frame). Matches reference approach.
    kp_opensim = np.einsum("ij,nkj->nki", _CAMERA_TO_OPENSIM, keypoints_3d)

    # Step 3: Scale to subject height
    # Use nose-to-ankle distance × 1.1 as full height estimate
    nose = kp_opensim[:, 0, :]       # nose
    l_ankle = kp_opensim[:, 13, :]   # left ankle
    r_ankle = kp_opensim[:, 14, :]   # right ankle
    ankle_mid = (l_ankle + r_ankle) / 2
    nose_to_ankle = np.linalg.norm(nose - ankle_mid, axis=1)
    median_height = np.median(nose_to_ankle[nose_to_ankle > 0.1])
    if median_height > 0.1:
        estimated_full = median_height * 1.1
        scale = subject_height / estimated_full
        kp_opensim = kp_opensim * scale

    # Step 4: Ground alignment — per-frame lowest foot Y=0
    foot_indices = [15, 16, 17, 18, 19, 20]  # all foot markers
    for i in range(n_frames):
        foot_y = kp_opensim[i, foot_indices, 1]
        min_y = np.min(foot_y)
        kp_opensim[i, :, 1] -= min_y

    # Step 6: Pelvis centering — discard cam_t XZ drift, keep Y from ground
    kp_opensim = _center_at_pelvis(kp_opensim)

    # Step 7: Select mapped keypoints + compute derived markers
    marker_names = []
    marker_data = []

    for idx in sorted(MHR70_TO_OPENSIM.keys()):
        marker_names.append(MHR70_TO_OPENSIM[idx])
        marker_data.append(kp_opensim[:, idx, :])

    # Derived markers (from SAM3D-OpenSim reference)
    # PelvisCenter = midpoint(LHip[9], RHip[10]) — anchors pelvis origin
    pelvis_center = (kp_opensim[:, 9] + kp_opensim[:, 10]) / 2
    marker_names.append("PelvisCenter")
    marker_data.append(pelvis_center)

    # Thorax = midpoint(LAcromion[67], RAcromion[68]) — mid-torso anchor
    thorax = (kp_opensim[:, 67] + kp_opensim[:, 68]) / 2
    marker_names.append("Thorax")
    marker_data.append(thorax)

    # SpineMid = midpoint(PelvisCenter, Thorax) — lumbar spine anchor
    spine_mid = (pelvis_center + thorax) / 2
    marker_names.append("SpineMid")
    marker_data.append(spine_mid)

    data = np.stack(marker_data, axis=1)  # (N, M, 3)

    # Step 8: Convert to millimeters (TRC convention)
    data_mm = data * 1000.0

    return data_mm, marker_names
