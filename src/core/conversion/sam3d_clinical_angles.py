"""Extract ISB-convention clinical DOFs from SAM 3D Body MHR global rotations.

Decomposes MHR 127-joint global rotation matrices into 14 clinical joint
groups (30 DOFs total) using local rotation extraction and Euler decomposition.

No FK solver needed — SAM 3D's MHR body model already enforces the
kinematic chain internally. We just read off the local rotations:
    R_local = R_parent.T @ R_child

Coordinate conventions:
  MHR body-centric: X=right, Y=up, Z=backward
  Clinical frame (via T_CLINICAL): X=forward, Y=up, Z=right
    Applied to 3-DOF and 2-DOF joints (parent frame ~ MHR global)
  MHR parent frame: used directly for hinge joints (knee, elbow)
    where the flexion axis is MHR-local Z (empirically verified)

ZXY intrinsic Euler in clinical frame:
  Z = flexion (about medio-lateral = MHR X)
  X = adduction (about A-P = -MHR Z)
  Y = axial rotation (about longitudinal = MHR Y)

ISB sign convention (applied via negate_isb flag):
  MHR Euler Z/X have opposite sign to ISB for most joints.
  Flex (col 0) and abd (col 1) are negated for: hip, knee, ankle, trunk,
  shoulder. NOT negated for: pelvis (global orientation), elbow (already
  ISB-correct). Applied before L/R mirroring.

Ankle decomposition:
  Dorsiflexion: 2dof from lowleg→foot (Z in clinical frame = mediolateral axis)
  Inversion/eversion: hinge from talocrural→subtalar (separate chain)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.shared.coordinate_transforms import MHR_BODY_TO_PIPELINE

from .sam3d_joint_map import MHR

# MHR body-centric → clinical frame (same as MHR → pipeline: X=fwd, Y=up, Z=right)
T_CLINICAL = MHR_BODY_TO_PIPELINE


# ---------------------------------------------------------------------------
# Joint definitions: (parent_idx, child_idx, output_key, is_left, decomp_type,
#                     negate_isb)
# decomp_type: "3dof" (ZXY Euler in clinical frame),
#              "hinge" (1-DOF flex in MHR parent frame),
#              "2dof" (ZX in clinical frame)
# negate_isb: True → negate flex (col 0) and abd (col 1) for ISB convention.
#   MHR ZXY Euler Z/X produce opposite signs to ISB for most joints.
#   Exceptions: pelvis (global orientation) and elbow (already ISB-correct).
# ---------------------------------------------------------------------------

_JOINT_DEFS: list[tuple[int, int, str, bool, str, bool]] = [
    # Pelvis: body_world → root (global orientation, signs already ISB-correct)
    (MHR["body_world"], MHR["root"],     "pelvis",     False, "3dof",  False),
    # Hips (ISB: flex/abd signs inverted vs MHR)
    (MHR["root"],       MHR["r_upleg"],  "hip_R",      False, "3dof",  True),
    (MHR["root"],       MHR["l_upleg"],  "hip_L",      True,  "3dof",  True),
    # Knees — hinge (MHR hinge Z is already ISB-positive for flexion)
    (MHR["r_upleg"],    MHR["r_lowleg"], "knee_R",     False, "hinge", False),
    (MHR["l_upleg"],    MHR["l_lowleg"], "knee_L",     True,  "hinge", False),
    # Ankles — dorsiflexion from lowleg→foot (2dof: flex from Z, abd from X in clinical frame)
    (MHR["r_lowleg"],   MHR["r_foot"],   "ankle_R",    False, "2dof", True),
    (MHR["l_lowleg"],   MHR["l_foot"],   "ankle_L",    True,  "2dof", True),
    # Ankles — inversion/eversion from talocrural→subtalar (separate chain)
    (MHR["r_talocrural"], MHR["r_subtalar"], "_ankle_abd_R", False, "hinge", False),
    (MHR["l_talocrural"], MHR["l_subtalar"], "_ankle_abd_L", True,  "hinge", False),
    # Trunk (L5/S1): root → spine0 (ISB: flex/abd signs inverted)
    (MHR["root"],       MHR["c_spine0"], "trunk",      False, "3dof",  True),
    # Shoulders (glenohumeral: clavicle → uparm, excludes scapulothoracic)
    # Using clavicle as parent matches GT model (humanoid_torque) which has
    # no scapular DOF — humerus is directly on the torso.
    # ISB: flex/abd signs inverted vs MHR.
    (MHR["r_clavicle"], MHR["r_uparm"],  "shoulder_R", False, "3dof",  True),
    (MHR["l_clavicle"], MHR["l_uparm"],  "shoulder_L", True,  "3dof",  True),
    # Elbows — hinge (signs already ISB-correct)
    (MHR["r_uparm"],    MHR["r_lowarm"], "elbow_R",    False, "hinge", False),
    (MHR["l_uparm"],    MHR["l_lowarm"], "elbow_L",    True,  "hinge", False),
    # Wrists (no ISB negation)
    (MHR["r_lowarm"],   MHR["r_wrist"],  "wrist_R",    False, "2dof",  False),
    (MHR["l_lowarm"],   MHR["l_wrist"],  "wrist_L",    True,  "2dof",  False),
]

# Column names per decomposition type
_COL_NAMES = {
    "3dof": ["flex_deg", "abd_deg", "rot_deg"],
    "hinge": ["flex_deg"],
    "2dof": ["flex_deg", "abd_deg"],
}

# Override column names for specific joints
_COL_OVERRIDES = {
    "pelvis": ["pelvis_flex_deg", "pelvis_abd_deg", "pelvis_rot_deg"],
    "trunk": ["trunk_flex_deg", "trunk_abd_deg", "trunk_rot_deg"],
    "wrist_R": ["wrist_flex_deg", "wrist_dev_deg"],
    "wrist_L": ["wrist_flex_deg", "wrist_dev_deg"],
    "_ankle_abd_R": ["ankle_abd_deg"],
    "_ankle_abd_L": ["ankle_abd_deg"],
}


# ---------------------------------------------------------------------------
# Decomposition functions
# ---------------------------------------------------------------------------

def _decompose_3dof(R: np.ndarray) -> np.ndarray:
    """Intrinsic ZXY Euler decomposition for 3-DOF joints.

    Input should be in clinical frame (via T_CLINICAL transform):
      Z rotation → flexion/extension (about medio-lateral axis)
      X rotation → adduction/abduction (about A-P axis)
      Y rotation → internal/external rotation (about longitudinal axis)

    Args:
        R: (N, 3, 3) rotation matrices in clinical frame.

    Returns:
        (N, 3) array of [flexion, adduction, rotation] in degrees.
    """
    r = Rotation.from_matrix(R)
    return r.as_euler("ZXY", degrees=True)


def _decompose_hinge(R: np.ndarray) -> np.ndarray:
    """Extract single flexion angle for hinge joints (knee, elbow).

    Input is in MHR parent frame (NOT clinical frame). Knee/elbow
    flexion is primarily about MHR-local Z axis (empirically verified).
    Uses ZXY Euler Z component.

    Args:
        R: (N, 3, 3) rotation matrices in MHR parent frame.

    Returns:
        (N, 1) flexion angle in degrees.
    """
    r = Rotation.from_matrix(R)
    euler = r.as_euler("ZXY", degrees=True)
    return euler[:, 0:1]  # Z component = flexion


def _decompose_2dof(R: np.ndarray) -> np.ndarray:
    """ZX decomposition for 2-DOF joints (ankle, wrist).

    Input should be in clinical frame (via T_CLINICAL transform).
    Z = flexion (about MHR X = mediolateral), X = abd/deviation.

    Args:
        R: (N, 3, 3) rotation matrices in clinical frame.

    Returns:
        (N, 2) array of [flexion, abduction/deviation] in degrees.
    """
    r = Rotation.from_matrix(R)
    euler = r.as_euler("ZXY", degrees=True)
    return euler[:, :2]


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_sam3d_clinical_angles(
    global_rots: np.ndarray,
    rest_global_rots: np.ndarray,
    fps: float,
    calibration_frames: int = 0,
) -> dict[str, pd.DataFrame]:
    """Extract clinical DOFs from MHR global rotation matrices.

    Args:
        global_rots: (N, 127, 3, 3) per-frame global rotations from SAM 3D.
        rest_global_rots: (127, 3, 3) rest-pose global rotations (zero body pose).
        fps: Frame rate in Hz.
        calibration_frames: If > 0, subtract the per-DOF median of the first
            N frames from all frames (static calibration). Matches clinical
            gait lab practice of using a standing trial as zero reference.

    Returns:
        Dict[str, pd.DataFrame] with 14 joint groups. Each DataFrame has
        time_s + angle columns in degrees. Compatible with
        save_comprehensive_angles_csv() and plot functions.
    """
    n_frames = global_rots.shape[0]
    time_s = np.arange(n_frames) / fps

    # Cast to float64 for precision
    global_rots = global_rots.astype(np.float64)
    rest_global_rots = rest_global_rots.astype(np.float64)

    # Detect invalid frames: zero/NaN rotation matrices (no detection)
    # Check root joint — if root is degenerate, whole frame is invalid.
    root_det = np.linalg.det(global_rots[:, 1])  # (N,)
    valid_mask = np.isfinite(root_det) & (np.abs(root_det) > 0.5)
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        print(f"[sam3d_angles] {n_invalid}/{n_frames} invalid frames (no detection)")

    valid_idx = np.where(valid_mask)[0]

    T = T_CLINICAL
    results: dict[str, pd.DataFrame] = {}

    for parent_idx, child_idx, output_key, is_left, decomp_type, negate_isb in _JOINT_DEFS:
        n_cols = {"3dof": 3, "hinge": 1, "2dof": 2}[decomp_type]
        angles = np.full((n_frames, n_cols), np.nan)

        if len(valid_idx) == 0:
            pass  # all NaN
        else:
            # Local rotation: R_local = R_parent^T @ R_child (valid frames only)
            R_parent = global_rots[valid_idx][:, parent_idx]  # (V, 3, 3)
            R_child = global_rots[valid_idx][:, child_idx]    # (V, 3, 3)
            R_local = np.einsum("nij,njk->nik", R_parent.transpose(0, 2, 1), R_child)

            # Rest-pose correction: R_motion = R_local @ R_rest_local^T
            R_rest_parent = rest_global_rots[parent_idx]  # (3, 3)
            R_rest_child = rest_global_rots[child_idx]    # (3, 3)
            R_rest_local = R_rest_parent.T @ R_rest_child  # (3, 3)
            R_motion = np.einsum("nij,jk->nik", R_local, R_rest_local.T)

            # Decompose valid frames
            if decomp_type == "hinge":
                angles[valid_idx] = _decompose_hinge(R_motion)
            else:
                R_clinical = np.einsum("ij,njk,lk->nil", T, R_motion, T)
                if decomp_type == "3dof":
                    angles[valid_idx] = _decompose_3dof(R_clinical)
                else:
                    angles[valid_idx] = _decompose_2dof(R_clinical)

        # ISB sign correction: negate flex (col 0) and abd (col 1).
        # MHR ZXY Euler produces opposite signs for flex/abd compared to ISB
        # for most joints. Applied before L/R mirroring.
        if negate_isb:
            angles[:, 0] *= -1  # flex
            if angles.shape[1] > 1:
                angles[:, 1] *= -1  # abd

        # Left-side convention: negate axes that are mirrored in the parent
        # frame. Applied after ISB negation. Which axes are mirrored depends
        # on the parent joint:
        #
        # Lower body (parent=root/pelvis or leg segments):
        #   Hips: abd and rot mirrored, flex not.
        #
        # Upper body (parent=per-side clavicle or arm segments):
        #   Shoulders: per-side clavicle parent → no negation needed.
        #   Wrists: L/R raw signs are consistent, no negation needed.
        #
        # Empirically verified from raw ZXY Euler decomposition of joey data.
        if is_left and decomp_type == "3dof":
            if output_key.startswith("shoulder"):
                pass  # per-side clavicle parent, no sign correction
            else:
                angles[:, 1] *= -1  # adduction
                angles[:, 2] *= -1  # rotation
        elif is_left and decomp_type == "2dof":
            if output_key.startswith("wrist"):
                pass  # wrist L/R same sign in forearm frame
            else:
                angles[:, 1] *= -1  # abd
        # Hinge joints: most (knee, elbow, ankle flex) don't need L/R sign
        # negation — MHR flex axis is consistent. But ankle abd (inversion/
        # eversion from talocrural→subtalar) mirrors L/R in the frontal plane.
        elif is_left and decomp_type == "hinge":
            if output_key.startswith("_ankle_abd"):
                angles[:, 0] *= -1  # inversion/eversion mirrors L/R

        # Pelvis: remove global heading (yaw) so rotation oscillates
        # around 0° during walking instead of showing camera-frame heading.
        # Unwrap first to fix ±180° discontinuities in Euler angles.
        if output_key == "pelvis":
            angles[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(angles[:, 2])))
            angles[:, 2] -= np.median(angles[:, 2])

        # Build column names
        if output_key in _COL_OVERRIDES:
            col_names = _COL_OVERRIDES[output_key]
        else:
            base = output_key.rsplit("_", 1)[0]  # "hip" from "hip_R"
            col_names = [f"{base}_{c}" for c in _COL_NAMES[decomp_type]]

        data = {"time_s": time_s}
        for i, col in enumerate(col_names):
            data[col] = angles[:, i]

        results[output_key] = pd.DataFrame(data)

    # Combine ankle flex (lowleg→foot) and ankle abd (talocrural→subtalar)
    # into unified ankle DataFrames with both columns.
    for side in ["R", "L"]:
        ankle_key = f"ankle_{side}"
        abd_key = f"_ankle_abd_{side}"
        if ankle_key in results and abd_key in results:
            abd_df = results[abd_key]
            abd_col = [c for c in abd_df.columns if c != "time_s"][0]
            results[ankle_key]["ankle_abd_deg"] = abd_df[abd_col].values
            del results[abd_key]

    # Optional static calibration: subtract per-DOF median of first N frames
    if calibration_frames > 0:
        n_cal = min(calibration_frames, n_frames)
        for df in results.values():
            for col in df.columns:
                if col == "time_s":
                    continue
                cal_median = df[col].iloc[:n_cal].median()
                df[col] -= cal_median
        print(f"[sam3d_angles] Calibrated zero from first {n_cal} frames")

    # Report
    n_groups = len(results)
    n_dofs = sum(len(df.columns) - 1 for df in results.values())
    print(f"[sam3d_angles] Extracted {n_dofs} DOFs across {n_groups} joint groups")

    return results
