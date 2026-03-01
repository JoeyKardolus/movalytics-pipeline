"""Convert OpenSim IK .mot angles to clinical joint-group CSVs.

Parses the .mot file and maps OpenSim coordinate names to clinical
joint groups in the same dict[str, DataFrame] format used by
extract_sam3d_clinical_angles(), so the same export and plotting
functions work for both sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# OpenSim .mot column → (clinical_group, clinical_column)
_MOT_MAP: dict[str, tuple[str, str]] = {
    # Pelvis
    "pelvis_tilt": ("pelvis", "pelvis_flex_deg"),
    "pelvis_list": ("pelvis", "pelvis_abd_deg"),
    "pelvis_rotation": ("pelvis", "pelvis_rot_deg"),
    # Hip
    "hip_flexion_r": ("hip_R", "hip_flex_deg"),
    "hip_adduction_r": ("hip_R", "hip_abd_deg"),
    "hip_rotation_r": ("hip_R", "hip_rot_deg"),
    "hip_flexion_l": ("hip_L", "hip_flex_deg"),
    "hip_adduction_l": ("hip_L", "hip_abd_deg"),
    "hip_rotation_l": ("hip_L", "hip_rot_deg"),
    # Knee
    "knee_angle_r": ("knee_R", "knee_flex_deg"),
    "knee_angle_l": ("knee_L", "knee_flex_deg"),
    # Ankle
    "ankle_angle_r": ("ankle_R", "ankle_flex_deg"),
    "ankle_angle_l": ("ankle_L", "ankle_flex_deg"),
    # Trunk (lumbar)
    "lumbar_extension": ("trunk", "trunk_flex_deg"),
    "lumbar_bending": ("trunk", "trunk_abd_deg"),
    "lumbar_rotation": ("trunk", "trunk_rot_deg"),
    # Shoulder
    "arm_flex_r": ("shoulder_R", "shoulder_flex_deg"),
    "arm_add_r": ("shoulder_R", "shoulder_abd_deg"),
    "arm_rot_r": ("shoulder_R", "shoulder_rot_deg"),
    "arm_flex_l": ("shoulder_L", "shoulder_flex_deg"),
    "arm_add_l": ("shoulder_L", "shoulder_abd_deg"),
    "arm_rot_l": ("shoulder_L", "shoulder_rot_deg"),
    # Elbow
    "elbow_flex_r": ("elbow_R", "elbow_flex_deg"),
    "elbow_flex_l": ("elbow_L", "elbow_flex_deg"),
}

# OpenSim → clinical sign corrections.
# With rest-pose IK calibration, most DOFs are directly correct.
# Only pelvis_tilt needs a sign flip (OpenSim posterior-positive
# vs clinical anterior-positive). Hip flexion convention matches.
_SIGN_CORRECTIONS: dict[str, float] = {
    # OpenSim pelvis_tilt positive = posterior tilt; clinical = anterior tilt
    "pelvis_tilt": -1.0,
}

# Ordered list of clinical groups (matches demo page expectations)
_GROUP_ORDER = [
    "pelvis", "hip_R", "hip_L", "knee_R", "knee_L",
    "ankle_R", "ankle_L", "trunk",
    "shoulder_R", "shoulder_L", "elbow_R", "elbow_L",
]


def _parse_mot(mot_path: Path) -> tuple[list[str], np.ndarray]:
    """Parse an OpenSim .mot file into column names and data array."""
    with open(mot_path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header row found in {mot_path}")

    columns = lines[header_idx].strip().split("\t")
    data = np.array(
        [[float(v) for v in line.strip().split("\t")]
         for line in lines[header_idx + 1:] if line.strip()]
    )
    return columns, data


def extract_opensim_clinical_angles(
    mot_path: Path | str,
    rest_mot_path: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Extract clinical joint angles from an OpenSim IK .mot file.

    If rest_mot_path is provided, the rest-pose angles are subtracted
    from all frames. This removes constant offsets caused by model
    shape mismatch (e.g. MHR pelvis proportions differ from OpenSim).

    Returns dict of DataFrames keyed by joint group name, in the same
    format as extract_sam3d_clinical_angles() so that
    save_comprehensive_angles_csv() and plot functions work unchanged.
    """
    mot_path = Path(mot_path)
    columns, data = _parse_mot(mot_path)

    # Parse rest-pose angles for calibration offset
    rest_offsets: dict[str, float] = {}
    if rest_mot_path is not None:
        rest_mot_path = Path(rest_mot_path)
        if rest_mot_path.exists():
            rest_cols, rest_data = _parse_mot(rest_mot_path)
            # Single frame (or average if multiple)
            for mot_name in _MOT_MAP:
                if mot_name in rest_cols:
                    idx = rest_cols.index(mot_name)
                    rest_offsets[mot_name] = float(np.mean(rest_data[:, idx]))

    time_col = data[:, columns.index("time")]

    # Build per-group DataFrames
    results: dict[str, pd.DataFrame] = {}

    for group in _GROUP_ORDER:
        group_cols: dict[str, np.ndarray] = {"time_s": time_col}

        for mot_name, (g, clinical_name) in _MOT_MAP.items():
            if g != group:
                continue
            if mot_name not in columns:
                continue
            col_idx = columns.index(mot_name)
            sign = _SIGN_CORRECTIONS.get(mot_name, 1.0)
            values = data[:, col_idx] - rest_offsets.get(mot_name, 0.0)
            group_cols[clinical_name] = values * sign

        # Only include groups that have at least one angle column
        if len(group_cols) > 1:
            results[group] = pd.DataFrame(group_cols)

    return results
