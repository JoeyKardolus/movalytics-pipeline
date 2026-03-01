"""ISB-standard clinical joint angle visualization for SAM 3D Body.

Fixed y-axis ranges per joint type, directional axis labels
(e.g., "Flexion (+) / Extension (−)"), and proper clinical layout.

Layout: 8 rows x 2 columns
  Row 0: Pelvis (spans both cols)
  Row 1: Trunk / Lumbar (spans both cols)
  Rows 2-7: Hip, Knee, Ankle, Shoulder, Elbow, Wrist (R | L)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# ---------------------------------------------------------------------------
# Directional axis labels: (positive_label, negative_label)
# ---------------------------------------------------------------------------

_AXIS_LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "pelvis": {
        "flex": ("Ant. Tilt", "Post. Tilt"),
        "abd":  ("Right Drop", "Left Drop"),
        "rot":  ("Right Fwd", "Left Fwd"),
    },
    "trunk": {
        "flex": ("Flexion", "Extension"),
        "abd":  ("Right Bend", "Left Bend"),
        "rot":  ("Right Rot", "Left Rot"),
    },
    "hip": {
        "flex": ("Flexion", "Extension"),
        "abd":  ("Abduction", "Adduction"),
        "rot":  ("Internal", "External"),
    },
    "knee": {
        "flex": ("Flexion", "Extension"),
    },
    "ankle": {
        "flex": ("Dorsiflexion", "Plantarflexion"),
        "abd":  ("Inversion", "Eversion"),
    },
    "shoulder": {
        "flex": ("Flexion", "Extension"),
        "abd":  ("Abduction", "Adduction"),
        "rot":  ("Internal", "External"),
    },
    "elbow": {
        "flex": ("Flexion", "Extension"),
    },
    "wrist": {
        "flex": ("Flexion", "Extension"),
        "dev":  ("Radial", "Ulnar"),
    },
}


# ---------------------------------------------------------------------------
# Clinical display labels per DOF column
# ---------------------------------------------------------------------------

_CLINICAL_LABELS: dict[str, dict[str, str]] = {
    "pelvis": {
        "pelvis_flex_deg": "Ant/Post Tilt",
        "pelvis_abd_deg": "Lateral List",
        "pelvis_rot_deg": "Axial Rotation",
    },
    "hip": {
        "hip_flex_deg": "Flex/Extension",
        "hip_abd_deg": "Abd/Adduction",
        "hip_rot_deg": "Int/Ext Rotation",
    },
    "knee": {
        "knee_flex_deg": "Flex/Extension",
    },
    "ankle": {
        "ankle_flex_deg": "Dorsi/Plantarflex",
        "ankle_abd_deg": "Inv/Eversion",
    },
    "trunk": {
        "trunk_flex_deg": "Flex/Extension",
        "trunk_abd_deg": "Lateral Bending",
        "trunk_rot_deg": "Axial Rotation",
    },
    "shoulder": {
        "shoulder_flex_deg": "Flex/Extension",
        "shoulder_abd_deg": "Abd/Adduction",
        "shoulder_rot_deg": "Int/Ext Rotation",
    },
    "elbow": {
        "elbow_flex_deg": "Flex/Extension",
    },
    "wrist": {
        "wrist_flex_deg": "Flex/Extension",
        "wrist_dev_deg": "Radial/Ulnar Dev",
    },
}


# ---------------------------------------------------------------------------
# Colors and layout
# ---------------------------------------------------------------------------

_COLORS = {
    "flex": "#2171B5",
    "abd":  "#E6550D",
    "rot":  "#31A354",
    "dev":  "#756BB1",
}

# Physiological ROM limits (AAOS/ISB) — degrees, matching OpenSim sign conventions.
# Values beyond these indicate tracking/IK errors.
_PHYSIO_ROM: dict[str, tuple[float, float]] = {
    "hip_flex_deg":      (-30, 130),
    "hip_abd_deg":       (-50, 50),
    "hip_rot_deg":       (-50, 50),
    "knee_flex_deg":     (-10, 150),
    "ankle_flex_deg":    (-50, 30),
    "ankle_abd_deg":     (-35, 35),
    "pelvis_flex_deg":   (-20, 35),
    "pelvis_abd_deg":    (-30, 30),
    "pelvis_rot_deg":    (-35, 35),
    "trunk_flex_deg":    (-35, 80),
    "trunk_abd_deg":     (-45, 45),
    "trunk_rot_deg":     (-50, 50),
    "shoulder_flex_deg": (-60, 180),
    "shoulder_abd_deg":  (-45, 180),
    "shoulder_rot_deg":  (-100, 100),
    "elbow_flex_deg":    (-10, 155),
}


def _load_normative_data(
    normative_path: Path,
) -> dict[str, tuple[float, float]]:
    """Load normative gait data and return static min/max bands per angle.

    The Schwartz 2008 data is gait-cycle-percentage indexed. For general
    (non-gait) activities, we collapse to the full range across the gait
    cycle: (min of lo, max of hi). This gives a "healthy walking range"
    reference band.

    Returns:
        Dict mapping angle column name → (band_lo, band_hi) in degrees.
    """
    with open(normative_path) as f:
        data = json.load(f)

    bands: dict[str, tuple[float, float]] = {}
    for angle_name, vals in data.items():
        lo = min(vals["lo"])
        hi = max(vals["hi"])
        bands[angle_name] = (lo, hi)
    return bands

_LAYOUT = [
    ("pelvis",     "Pelvis",           "pelvis",   0, None),
    ("trunk",      "Trunk / Lumbar",   "trunk",    1, None),
    ("hip_R",      "Hip (Right)",      "hip",      2, 0),
    ("hip_L",      "Hip (Left)",       "hip",      2, 1),
    ("knee_R",     "Knee (Right)",     "knee",     3, 0),
    ("knee_L",     "Knee (Left)",      "knee",     3, 1),
    ("ankle_R",    "Ankle (Right)",    "ankle",    4, 0),
    ("ankle_L",    "Ankle (Left)",     "ankle",    4, 1),
    ("shoulder_R", "Shoulder (Right)", "shoulder", 5, 0),
    ("shoulder_L", "Shoulder (Left)",  "shoulder", 5, 1),
    ("elbow_R",    "Elbow (Right)",    "elbow",    6, 0),
    ("elbow_L",    "Elbow (Left)",     "elbow",    6, 1),
    # Wrists excluded — no hand detection
]


def _dof_key(col_name: str) -> str:
    """Extract DOF type key from column name (flex, abd, rot, dev)."""
    for key in ("flex", "abd", "rot", "dev"):
        if key in col_name:
            return key
    return "flex"


def _get_y_limits(
    data_ranges: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Data-adaptive y-axis limits with padding.

    Fits the axes tightly around the actual data with comfortable margins.
    Only includes zero line if the data is near zero (within 2x data span).
    """
    if not data_ranges:
        return -10.0, 10.0

    d_lo = min(lo for lo, _ in data_ranges.values())
    d_hi = max(hi for _, hi in data_ranges.values())
    span = d_hi - d_lo
    margin = max(5.0, 0.15 * span)

    y_min = d_lo - margin
    y_max = d_hi + margin

    # Include zero only if data is near zero (within 2x span)
    if d_lo >= 0 and d_lo < 2 * span:
        y_min = min(y_min, -margin * 0.3)
    if d_hi <= 0 and abs(d_hi) < 2 * span:
        y_max = max(y_max, margin * 0.3)

    return y_min, y_max


def _build_ylabel(joint_type: str, dof_keys: list[str]) -> str:
    """Build directional y-axis label from the DOFs plotted on this axis."""
    axis_labels = _AXIS_LABELS.get(joint_type, {})
    parts = []
    for dk in dof_keys:
        if dk in axis_labels:
            pos, neg = axis_labels[dk]
            parts.append(f"{pos} (+) / {neg} (\u2212)")

    if not parts:
        return "Angle (deg)"

    # If multiple DOFs, use the first (flex usually) for label
    return f"{parts[0]}\n(deg)"


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_sam3d_clinical_angles(
    angle_results: dict[str, pd.DataFrame],
    output_path: Path | None = None,
    title_prefix: str = "",
    dpi: int = 150,
    normative_path: Path | None = None,
) -> None:
    """ISB-standard clinical joint angle visualization for SAM 3D Body.

    Args:
        angle_results: Dict of DataFrames from extract_sam3d_clinical_angles().
        output_path: Save path (None = plt.show()).
        title_prefix: Video name prefix for title.
        dpi: Output resolution.
        normative_path: Path to normative gait data JSON (Schwartz 2008).
            When provided, renders gray bands showing healthy walking range
            and red dashed lines at physiological ROM limits if exceeded.
    """
    # Load normative bands if available
    norm_bands: dict[str, tuple[float, float]] = {}
    if normative_path and normative_path.exists():
        try:
            norm_bands = _load_normative_data(normative_path)
        except Exception as e:
            print(f"[sam3d_viz] Warning: could not load normative data: {e}")

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(7, 2, hspace=0.5, wspace=0.35)
    has_normative = False

    for output_key, panel_title, joint_type, row, col in _LAYOUT:
        if output_key not in angle_results:
            continue

        df = angle_results[output_key]
        if "time_s" not in df.columns:
            continue

        times = df["time_s"].values
        labels = _CLINICAL_LABELS.get(joint_type, {})

        # Create subplot
        if col is None:
            ax = fig.add_subplot(gs[row, :])
        else:
            ax = fig.add_subplot(gs[row, col])

        # Collect data for y-range computation
        dof_keys = []
        data_ranges: dict[str, tuple[float, float]] = {}
        has_data = False

        for col_name in df.columns:
            if col_name == "time_s":
                continue
            angles = df[col_name].values
            if np.all(np.isnan(angles)):
                continue

            dk = _dof_key(col_name)
            dof_keys.append(dk)
            d_min = float(np.nanmin(angles))
            d_max = float(np.nanmax(angles))
            data_ranges[dk] = (d_min, d_max)

            has_data = True

            # Normative band (behind data line)
            if col_name in norm_bands:
                n_lo, n_hi = norm_bands[col_name]
                ax.axhspan(n_lo, n_hi, color="#B0B0B0", alpha=0.18,
                           zorder=0)
                has_normative = True

            # Physiological ROM limits — only draw if data exceeds them
            if col_name in _PHYSIO_ROM:
                rom_lo, rom_hi = _PHYSIO_ROM[col_name]
                if d_min < rom_lo:
                    ax.axhline(y=rom_lo, color="red", linestyle="--",
                               linewidth=1.0, alpha=0.6, zorder=1)
                if d_max > rom_hi:
                    ax.axhline(y=rom_hi, color="red", linestyle="--",
                               linewidth=1.0, alpha=0.6, zorder=1)

            # Data line
            label = labels.get(col_name, col_name)
            color = _COLORS.get(dk, "#333333")
            ax.plot(times, angles, label=label, color=color,
                    linewidth=1.5, alpha=0.9, zorder=2)

        if has_data:
            ax.set_title(panel_title, fontsize=11, fontweight="bold", loc="left")
            ax.set_xlabel("Time (s)", fontsize=9)

            # Data-adaptive y-axis range
            y_lo, y_hi = _get_y_limits(data_ranges)
            ax.set_ylim(y_lo, y_hi)

            # Directional y-axis label
            ax.set_ylabel(_build_ylabel(joint_type, dof_keys), fontsize=8)

            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            ax.tick_params(labelsize=8)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.7, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{panel_title}\n(No data)",
                    ha="center", va="center", fontsize=10, color="gray",
                    transform=ax.transAxes)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

    # Title
    n_dofs = sum(
        sum(1 for c in df.columns if c != "time_s")
        for df in angle_results.values()
    )
    title = f"Clinical Joint Angles ({n_dofs} DOF)"
    if title_prefix:
        title = f"{title_prefix} \u2014 {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    # Legend note for normative bands
    if has_normative:
        fig.text(0.5, 0.002,
                 "Gray bands = normative walking range (Schwartz 2008)  |  "
                 "Red dashes = physiological ROM limits",
                 ha="center", fontsize=8, color="#666666")

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[sam3d_viz] Saved: {output_path}")
        plt.close()
    else:
        plt.show()
