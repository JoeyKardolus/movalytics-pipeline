"""Convert per-frame MHR surface markers + joint centers + keypoints to TRC.

Takes marker positions extracted from MHR mesh vertices (via atlas),
MHR joint centers from SAM 3D skeleton, and MHR70 surface keypoints,
and produces an OpenSim-compatible TRC file.

Marker sources:
  41 surface (vertex lookup) + 2 computed HJC (Bell's method)
  + 10 MHR joint centers (from SAM 3D skeleton joint_coords)
  + 34 MHR70 surface keypoints (from MHR mesh regressor keypoints_3d)

Coordinate conventions:
  Surface markers (MHR body-centric): X=left, Y=up, Z=backward (meters)
    Verified: R hip at X=-0.078, L hip at X=+0.078 → X = subject's LEFT.
  Joint coords / keypoints (camera):  X=left, Y=down, Z=forward (meters)
    MHR body-centric with Y/Z flip applied by MHRHead.forward.
  Output (TRC/OpenSim):               X=forward, Y=up, Z=right (millimeters)

Transform verification (both paths yield identical pipeline coords):
  Body-centric P=(px,py,pz) → pipeline: (-pz, py, -px)
  Camera P_cam=(px,-py,-pz) → pipeline: (-pz, py, -px) ✓
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scipy.signal import detrend

from src.core.conversion.mhr_marker_atlas import compute_hjc_markers_batch
from src.core.conversion.trc_io import save_trc
from src.shared.constants import MHR70_KEYPOINTS, MHR_JOINT_CENTERS
from src.shared.coordinate_transforms import CAMERA_TO_PIPELINE, MHR_BODY_TO_PIPELINE
from src.shared.filtering import butterworth_lowpass

# Local aliases for readability
_MHR_BODY_TO_PIPELINE = MHR_BODY_TO_PIPELINE
_CAMERA_TO_PIPELINE = CAMERA_TO_PIPELINE
_MHR_JOINT_CENTERS = MHR_JOINT_CENTERS
_MHR70_KEYPOINTS = MHR70_KEYPOINTS


def mhr_markers_to_trc(
    marker_positions: np.ndarray,
    marker_names: list[str],
    subject_height: float,
    fps: float,
    output_path: Path | str,
    rest_vertices: np.ndarray | None = None,
    joint_coords: np.ndarray | None = None,
    keypoints_3d: np.ndarray | None = None,
    cam_t: np.ndarray | None = None,
) -> Path:
    """Convert per-frame MHR markers to TRC (up to 87 markers).

    Args:
        marker_positions: (N, M, 3) surface marker positions per frame.
            In MHR body-centric coords (X=left, Y=up, Z=backward), meters.
        marker_names: list of M surface marker names (matching dim 1).
        subject_height: Subject height in meters for metric scaling.
        fps: Video frame rate.
        output_path: Path for output TRC file.
        rest_vertices: (V, 3) MHR rest-pose mesh vertices for height estimate.
        joint_coords: (N, 127, 3) MHR joint positions, body-model-relative.
            Camera convention (X=left, Y=down, Z=forward). Optional.
        keypoints_3d: (N, 70, 3) MHR70 surface keypoints, body-model-relative.
            Camera convention (X=left, Y=down, Z=forward). Optional.
        cam_t: (N, 3) camera translation in camera convention (meters).
            Y delta (relative to frame 0) applied for vertical movement.

    Returns:
        Path to the written TRC file.
    """
    output_path = Path(output_path)
    n_frames, n_markers, _ = marker_positions.shape

    # ── 1. Transform surface markers: MHR body-centric → pipeline ──
    markers_pipe = np.einsum("ij,nkj->nki", _MHR_BODY_TO_PIPELINE, marker_positions)

    # ── 2. Extract and transform MHR joint centers (10 joints) ──
    jc_pipe = None
    jc_names: list[str] = []
    if joint_coords is not None:
        jc_indices = list(_MHR_JOINT_CENTERS.values())
        jc_names = list(_MHR_JOINT_CENTERS.keys())
        jc = joint_coords[:, jc_indices, :]  # (N, 10, 3) camera convention
        jc_pipe = np.einsum("ij,nkj->nki", _CAMERA_TO_PIPELINE, jc)

    # ── 3. Extract and transform MHR70 keypoints (34 keypoints) ──
    kp_pipe = None
    kp_names: list[str] = []
    if keypoints_3d is not None:
        kp_indices = sorted(_MHR70_KEYPOINTS.keys())
        kp_names = [_MHR70_KEYPOINTS[i] for i in kp_indices]
        kp = keypoints_3d[:, kp_indices, :]  # (N, 34, 3) camera convention
        kp_pipe = np.einsum("ij,nkj->nki", _CAMERA_TO_PIPELINE, kp)

    # ── 4. Height-based metric scaling ──
    if rest_vertices is not None:
        # Use rest-pose mesh bounding box height (Y axis in MHR = up)
        mesh_height = float(rest_vertices[:, 1].max() - rest_vertices[:, 1].min())
    else:
        # Fallback: estimate from per-frame marker range (Y axis in pipeline = up)
        y_range = markers_pipe[:, :, 1].max(axis=1) - markers_pipe[:, :, 1].min(axis=1)
        mesh_height = float(np.median(y_range))

    if mesh_height > 0.1:
        scale = subject_height / mesh_height
        markers_pipe = markers_pipe * scale
        if jc_pipe is not None:
            jc_pipe = jc_pipe * scale
        if kp_pipe is not None:
            kp_pipe = kp_pipe * scale
        print(f"[mhr-trc] Height scaling: mesh={mesh_height:.3f}m → "
              f"subject={subject_height:.2f}m (scale={scale:.3f})")

    # ── 5. cam_t Y for vertical dynamics + ground alignment ──
    # cam_t camera: X=left, Y=down, Z=forward. Pipeline Y = -cam_Y.
    # Detrend removes slow drift, Butterworth keeps stride oscillation,
    # final global min foot Y = 0 prevents ground penetration.
    foot_names = {
        "r_calc_study", "L_calc_study",
        "r_toe_study", "L_toe_study",
        "r_5meta_study", "L_5meta_study",
    }
    foot_indices = [i for i, n in enumerate(marker_names) if n in foot_names]

    if cam_t is not None and len(cam_t) == n_frames and n_frames > 12:
        cam_y_pipe = -cam_t[:, 1] * scale  # cam Y=down → pipe Y=up, scaled
        # Detrend: remove linear drift (camera slowly moving up/down)
        cam_y_dt = detrend(cam_y_pipe, type="linear")
        # Butterworth low-pass: keep stride frequency, kill perspective noise
        cam_y_smooth = butterworth_lowpass(cam_y_dt, cutoff_hz=6.0, fps=fps, order=2)

        # Apply as Y offset to all markers
        markers_pipe[:, :, 1] += cam_y_smooth[:, None]
        if jc_pipe is not None:
            jc_pipe[:, :, 1] += cam_y_smooth[:, None]
        if kp_pipe is not None:
            kp_pipe[:, :, 1] += cam_y_smooth[:, None]

        print(f"[mhr-trc] cam_t Y (detrend+butter): range="
              f"[{cam_y_smooth.min():.3f}, {cam_y_smooth.max():.3f}]m")

    # Global min foot Y = small offset above 0: ground plane baseline.
    # 5mm padding prevents IK solver from pushing skeleton through ground.
    if foot_indices:
        ground = np.min(markers_pipe[:, foot_indices, 1])
    else:
        ground = np.min(markers_pipe[:, :, 1])

    markers_pipe[:, :, 1] -= ground - 0.005
    if jc_pipe is not None:
        jc_pipe[:, :, 1] -= ground
    if kp_pipe is not None:
        kp_pipe[:, :, 1] -= ground

    foot_min_pf = markers_pipe[:, foot_indices, 1].min(axis=1) if foot_indices else markers_pipe[:, :, 1].min(axis=1)
    foot_max_pf = markers_pipe[:, foot_indices, 1].max(axis=1) if foot_indices else markers_pipe[:, :, 1].max(axis=1)
    print(f"[mhr-trc] Ground alignment: foot Y range="
          f"[{foot_min_pf.min():.3f}, {foot_max_pf.max():.3f}]m")

    # ── 7. Compute 2 HJC markers (Bell's method from ASIS/PSIS) ──
    hjc_positions, hjc_names = compute_hjc_markers_batch(
        markers_pipe, marker_names,
    )

    # ── 8. Concatenate all marker sources ──
    all_parts = [markers_pipe, hjc_positions]
    all_names = list(marker_names) + hjc_names

    if jc_pipe is not None:
        all_parts.append(jc_pipe)
        all_names.extend(jc_names)

    if kp_pipe is not None:
        all_parts.append(kp_pipe)
        all_names.extend(kp_names)

    all_positions = np.concatenate(all_parts, axis=1)

    # ── 9. Convert to millimeters (TRC convention) ──
    all_positions_mm = all_positions * 1000.0

    # ── 10. Write TRC ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_trc(all_positions_mm, all_names, output_path, fps)

    n_surface = n_markers
    n_jc = len(jc_names)
    n_kp = len(kp_names)
    n_total = all_positions_mm.shape[1]
    print(f"[mhr-trc] Wrote {n_total} markers ({n_surface} surface + "
          f"{len(hjc_names)} HJC + {n_jc} joint centers + "
          f"{n_kp} keypoints), {n_frames} frames → {output_path}")

    return output_path
