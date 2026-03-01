#!/usr/bin/env python3
"""Validate MHR marker atlas against biomechanical standards.

Quantifies marker placement quality by checking pelvis geometry, hip joint
centers, torso proportions, and limb marker distances against published
anatomical norms and the OpenSim LaiUhlrich2022 generic model.

With --motion flag, also compares per-frame velocity of surface markers vs
MHR skeleton joints to diagnose motion amplification.

Input: data/output/<video>/ directory with *_sam3d.npz + *_mhr_markers.trc.
Output: Table of checks with measured value, expected range, PASS/WARN/FAIL.

Coordinate conventions:
  MHR body-centric (rest-pose): X=right, Y=up, Z=backward (meters)
  MHR joint_coords (camera):    X=right, Y=down, Z=forward (meters)
  rest_joint_coords:            Same as camera convention (from worker output)
  TRC (pipeline):               X=forward, Y=up, Z=right (millimeters)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# MHR joint indices (from sam3d_joint_map.py)
_MHR = {
    "body_world": 0, "root": 1,
    "l_upleg": 2, "l_lowleg": 3, "l_foot": 4, "l_ball": 8,
    "r_upleg": 18, "r_lowleg": 19, "r_foot": 20, "r_ball": 24,
    "c_spine0": 34, "c_spine1": 35, "c_spine2": 36, "c_spine3": 37,
    "r_uparm": 39, "r_lowarm": 40,
    "l_uparm": 75, "l_lowarm": 76,
    "c_neck": 110, "c_head": 113,
}


def _status(val: float, lo: float, hi: float) -> str:
    """Return PASS/WARN/FAIL with color based on range check."""
    if lo <= val <= hi:
        return "\033[32mPASS\033[0m"
    elif lo * 0.8 <= val <= hi * 1.2:
        return "\033[33mWARN\033[0m"
    else:
        return "\033[31mFAIL\033[0m"


def _cm(m: float) -> float:
    """Convert meters to centimeters."""
    return m * 100.0


def validate(
    npz_path: Path,
    atlas_path: Path | None = None,
) -> list[dict]:
    """Run all validation checks.

    Args:
        npz_path: Path to SAM 3D NPZ with rest_vertices, rest_joint_coords.
        atlas_path: Path to mhr_marker_atlas.json. Auto-detected if None.

    Returns:
        List of check result dicts.
    """
    data = np.load(str(npz_path), allow_pickle=True)
    # rest_vertices and rest_joint_coords are in MHR body-centric convention:
    # X=right, Y=up, Z=backward (meters). No flip needed.
    rest_verts_bc = data["rest_vertices"]         # (V, 3)
    rest_joints_bc = data["rest_joint_coords"]    # (127, 3)

    # Load marker atlas
    if atlas_path is None:
        atlas_path = (
            Path(__file__).parent.parent.parent / "src" / "core" / "conversion"
            / "mhr_marker_atlas.json"
        )
    if atlas_path.exists():
        atlas = json.loads(atlas_path.read_text())
    else:
        # Fallback: import from module
        from src.core.conversion.mhr_marker_atlas import MHR_SURFACE_MARKERS
        atlas = MHR_SURFACE_MARKERS

    def _marker(name: str) -> np.ndarray:
        """Get marker position from rest-pose vertices (body-centric)."""
        return rest_verts_bc[atlas[name]]

    def _joint(name: str) -> np.ndarray:
        """Get joint position (body-centric)."""
        return rest_joints_bc[_MHR[name]]

    checks: list[dict] = []

    def _check(category: str, name: str, value: float, lo: float, hi: float,
               unit: str = "cm"):
        status = _status(value, lo, hi)
        checks.append({
            "category": category, "name": name,
            "value": value, "lo": lo, "hi": hi,
            "unit": unit, "status": status,
        })

    # ═══════════════════════════════════════════════════════════════════
    # Pelvis geometry
    # ═══════════════════════════════════════════════════════════════════
    rasi = _marker("r.ASIS_study")
    lasi = _marker("L.ASIS_study")
    rpsi = _marker("r.PSIS_study")
    lpsi = _marker("L.PSIS_study")

    asis_dist = _cm(np.linalg.norm(rasi - lasi))
    _check("Pelvis", "ASIS-ASIS distance", asis_dist, 24.0, 28.0)

    psis_dist = _cm(np.linalg.norm(rpsi - lpsi))
    _check("Pelvis", "PSIS-PSIS distance", psis_dist, 8.0, 14.0)

    mid_asis = (rasi + lasi) / 2
    mid_psis = (rpsi + lpsi) / 2
    # Sagittal depth: forward component (Z in body-centric = backward, negate)
    asis_psis_vec = mid_asis - mid_psis
    # In body-centric: X=right, Y=up, Z=backward
    # Sagittal depth is the Z component (forward-backward distance)
    sagittal_depth = _cm(abs(asis_psis_vec[2]))
    _check("Pelvis", "ASIS-PSIS sagittal depth", sagittal_depth, 8.0, 14.0)

    # Pelvis tilt: ASIS-PSIS height difference
    # Positive tilt = ASIS lower than PSIS (anterior tilt)
    asis_y = (rasi[1] + lasi[1]) / 2
    psis_y = (rpsi[1] + lpsi[1]) / 2
    tilt_deg = np.degrees(np.arctan2(psis_y - asis_y, abs(asis_psis_vec[2])))
    _check("Pelvis", "Pelvis anterior tilt (rest)", tilt_deg, 4.0, 16.0, "deg")

    # ═══════════════════════════════════════════════════════════════════
    # HJC validation
    # ═══════════════════════════════════════════════════════════════════
    mhr_lhjc = _joint("l_upleg")
    mhr_rhjc = _joint("r_upleg")

    # Bell's method HJC from ASIS/PSIS
    pelvis_width = np.linalg.norm(rasi - lasi)
    z_axis = (rasi - lasi) / (pelvis_width + 1e-8)
    temp = mid_asis - mid_psis
    x_axis = temp / (np.linalg.norm(temp) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    x_axis = np.cross(y_axis, z_axis)

    bell_rhjc = mid_asis + pelvis_width * (
        -0.19 * x_axis - 0.30 * y_axis + 0.36 * z_axis
    )
    bell_lhjc = mid_asis + pelvis_width * (
        -0.19 * x_axis - 0.30 * y_axis - 0.36 * z_axis
    )

    rhjc_err = _cm(np.linalg.norm(mhr_rhjc - bell_rhjc))
    lhjc_err = _cm(np.linalg.norm(mhr_lhjc - bell_lhjc))
    _check("HJC", "R HJC: MHR vs Bell's", rhjc_err, 0.0, 3.0)
    _check("HJC", "L HJC: MHR vs Bell's", lhjc_err, 0.0, 3.0)

    # ASIS height relative to HJC (ASIS should be ~7-8cm above)
    rasi_above_rhjc = _cm(rasi[1] - mhr_rhjc[1])
    lasi_above_lhjc = _cm(lasi[1] - mhr_lhjc[1])
    _check("HJC", "R ASIS above R HJC", rasi_above_rhjc, 5.0, 10.0)
    _check("HJC", "L ASIS above L HJC", lasi_above_lhjc, 5.0, 10.0)

    # HJC offset vector in pelvis local frame (compare to OpenSim generic)
    # OpenSim generic: (-0.056, -0.079, ±0.077) in OpenSim coords (X=fwd,Y=up,Z=right)
    pelvis_center = _joint("root")
    rhjc_offset = mhr_rhjc - pelvis_center
    lhjc_offset = mhr_lhjc - pelvis_center
    # Body-centric (X=right,Y=up,Z=back) → OpenSim (X=fwd,Y=up,Z=right)
    # X_osim = -Z_mhr, Y_osim = Y_mhr, Z_osim = X_mhr
    rhjc_osim = np.array([-rhjc_offset[2], rhjc_offset[1], rhjc_offset[0]])
    lhjc_osim = np.array([-lhjc_offset[2], lhjc_offset[1], lhjc_offset[0]])
    print(f"\n  R HJC offset (OpenSim frame): X={rhjc_osim[0]:.4f}, "
          f"Y={rhjc_osim[1]:.4f}, Z={rhjc_osim[2]:.4f}")
    print(f"  L HJC offset (OpenSim frame): X={lhjc_osim[0]:.4f}, "
          f"Y={lhjc_osim[1]:.4f}, Z={lhjc_osim[2]:.4f}")
    print(f"  OpenSim generic reference:     X=-0.056, Y=-0.079, Z=±0.077")

    # ═══════════════════════════════════════════════════════════════════
    # Torso/trunk geometry
    # ═══════════════════════════════════════════════════════════════════
    c7 = _marker("C7_study")
    r_shoulder = _marker("r_shoulder_study")
    l_shoulder = _marker("L_shoulder_study")
    shoulder_mid = (r_shoulder + l_shoulder) / 2
    pelvis_mid = (mhr_lhjc + mhr_rhjc) / 2

    # Trunk angle: pelvis→C7 vs vertical (Y-axis) in sagittal plane
    spine_vec = c7 - pelvis_mid
    # In body-centric: sagittal = Y-Z plane (Y=up, Z=backward)
    spine_yz = np.array([spine_vec[2], spine_vec[1]])  # (backward, up)
    cos_a = np.dot(spine_yz, [0.0, 1.0]) / (np.linalg.norm(spine_yz) + 1e-8)
    trunk_angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    if spine_vec[2] > 0:  # backward lean
        trunk_angle = -trunk_angle
    _check("Torso", "Trunk angle (rest)", abs(trunk_angle), 0.0, 8.0, "deg")

    # C7 relative to shoulder midpoint (C7 should be slightly posterior)
    c7_vs_shoulder = c7 - shoulder_mid
    c7_posterior = _cm(-c7_vs_shoulder[2])  # posterior = -Z in body-centric
    _check("Torso", "C7 posterior to shoulders", c7_posterior, 2.0, 8.0)

    # Shoulder width vs pelvis width ratio
    shoulder_width = _cm(np.linalg.norm(r_shoulder - l_shoulder))
    ratio = shoulder_width / asis_dist if asis_dist > 0 else 0
    _check("Torso", "Shoulder/pelvis width ratio", ratio, 1.3, 1.9, "ratio")

    # C7 height above pelvis center
    c7_height = _cm(c7[1] - pelvis_mid[1])
    _check("Torso", "C7 height above pelvis", c7_height, 38.0, 52.0)

    # ═══════════════════════════════════════════════════════════════════
    # Limb markers
    # ═══════════════════════════════════════════════════════════════════
    # Knee epicondyle separation
    r_knee = _marker("r_knee_study")
    r_mknee = _marker("r_mknee_study")
    l_knee = _marker("L_knee_study")
    l_mknee = _marker("L_mknee_study")
    r_knee_sep = _cm(np.linalg.norm(r_knee - r_mknee))
    l_knee_sep = _cm(np.linalg.norm(l_knee - l_mknee))
    _check("Limbs", "R knee epicondyle separation", r_knee_sep, 7.0, 12.0)
    _check("Limbs", "L knee epicondyle separation", l_knee_sep, 7.0, 12.0)

    # Ankle malleolus separation
    r_ankle = _marker("r_ankle_study")
    r_mankle = _marker("r_mankle_study")
    l_ankle = _marker("L_ankle_study")
    l_mankle = _marker("L_mankle_study")
    r_ankle_sep = _cm(np.linalg.norm(r_ankle - r_mankle))
    l_ankle_sep = _cm(np.linalg.norm(l_ankle - l_mankle))
    _check("Limbs", "R ankle malleolus separation", r_ankle_sep, 5.0, 9.0)
    _check("Limbs", "L ankle malleolus separation", l_ankle_sep, 5.0, 9.0)

    # Marker-to-MHR-joint distances (surface markers should be near joint center)
    joint_marker_pairs = [
        ("R knee lateral", "r_knee_study", "r_lowleg"),
        ("L knee lateral", "L_knee_study", "l_lowleg"),
        ("R ankle lateral", "r_ankle_study", "r_foot"),
        ("L ankle lateral", "L_ankle_study", "l_foot"),
        ("R elbow lateral", "r_lelbow_study", "r_lowarm"),
        ("L elbow lateral", "L_lelbow_study", "l_lowarm"),
    ]
    for label, marker_name, joint_name in joint_marker_pairs:
        m = _marker(marker_name)
        j = _joint(joint_name)
        dist = _cm(np.linalg.norm(m - j))
        _check("Limbs", f"{label} → joint center", dist, 0.0, 6.0)

    # ═══════════════════════════════════════════════════════════════════
    # Overall mesh stats
    # ═══════════════════════════════════════════════════════════════════
    mesh_height = rest_verts_bc[:, 1].max() - rest_verts_bc[:, 1].min()
    _check("Mesh", "Total mesh height", _cm(mesh_height), 150.0, 200.0)

    head_to_foot = _cm(_joint("c_head")[1] - min(
        _joint("l_ball")[1], _joint("r_ball")[1]
    ))
    _check("Mesh", "Head-to-foot joint height", head_to_foot, 150.0, 200.0)

    return checks


def print_results(checks: list[dict]) -> None:
    """Pretty-print validation results as a table."""
    print("\n" + "=" * 75)
    print("MARKER ATLAS VALIDATION REPORT")
    print("=" * 75)

    current_cat = ""
    for c in checks:
        if c["category"] != current_cat:
            current_cat = c["category"]
            print(f"\n── {current_cat} ──")

        val_str = f"{c['value']:8.2f} {c['unit']:5s}"
        range_str = f"[{c['lo']:.1f} - {c['hi']:.1f}]"
        print(f"  {c['name']:40s} {val_str}  {range_str:16s}  {c['status']}")

    # Summary
    n_pass = sum(1 for c in checks if "PASS" in c["status"])
    n_warn = sum(1 for c in checks if "WARN" in c["status"])
    n_fail = sum(1 for c in checks if "FAIL" in c["status"])
    print(f"\n{'=' * 75}")
    print(f"Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL  "
          f"(total {len(checks)} checks)")
    print("=" * 75)


def motion_diagnostic(trc_path: Path) -> None:
    """Compare per-frame velocity of pelvis marker sources in TRC.

    Reads the TRC file and computes frame-to-frame displacement (velocity)
    for each pelvis marker source:
      - ASIS/PSIS surface markers (4 markers, from mesh vertices)
      - Bell's HJC (2 markers, computed from ASIS/PSIS)
      - MHR hip joints (2 markers, from SAM 3D skeleton)

    Prints velocity stats and ratio (surface/skeleton) to diagnose
    whether surface markers amplify pelvis movement.
    """
    # Add project root to sys.path for imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.core.conversion.trc_io import load_trc

    positions, marker_names, fps = load_trc(trc_path)
    # positions: (N, M, 3) in mm
    n_frames = positions.shape[0]

    name_to_idx = {n: i for i, n in enumerate(marker_names)}

    # Pelvis marker groups
    asis_psis_names = ["r.ASIS_study", "L.ASIS_study",
                       "r.PSIS_study", "L.PSIS_study"]
    bell_hjc_names = ["RHJC_study", "LHJC_study"]
    mhr_hip_names = ["MHR_RHip", "MHR_LHip"]

    def _get_group_positions(names: list[str]) -> np.ndarray | None:
        indices = [name_to_idx.get(n) for n in names]
        if any(i is None for i in indices):
            missing = [n for n, i in zip(names, indices) if i is None]
            print(f"  WARNING: markers not found in TRC: {missing}")
            return None
        return positions[:, indices, :]  # (N, K, 3)

    asis_psis = _get_group_positions(asis_psis_names)
    bell_hjc = _get_group_positions(bell_hjc_names)
    mhr_hip = _get_group_positions(mhr_hip_names)

    if asis_psis is None or mhr_hip is None:
        print("ERROR: Cannot run motion diagnostic — missing markers.")
        return

    def _velocities(pos: np.ndarray) -> np.ndarray:
        """Per-frame centroid velocity (mm/frame)."""
        centroid = pos.mean(axis=1)  # (N, 3)
        return np.linalg.norm(np.diff(centroid, axis=0), axis=1)  # (N-1,)

    vel_surface = _velocities(asis_psis)
    vel_skeleton = _velocities(mhr_hip)
    vel_bell = _velocities(bell_hjc) if bell_hjc is not None else None

    print(f"\n{'=' * 75}")
    print("PELVIS MOTION DIAGNOSTIC")
    print(f"{'=' * 75}")
    print(f"  TRC: {trc_path}")
    print(f"  Frames: {n_frames}, FPS: {fps:.1f}")

    print(f"\n── Per-frame centroid velocity (mm/frame) ──")
    fmt = "  {:<30s}  median={:7.2f}  P95={:7.2f}  max={:7.2f}"
    print(fmt.format(
        "ASIS/PSIS (4 surface)",
        np.median(vel_surface), np.percentile(vel_surface, 95),
        vel_surface.max()))
    if vel_bell is not None:
        print(fmt.format(
            "Bell's HJC (2 derived)",
            np.median(vel_bell), np.percentile(vel_bell, 95),
            vel_bell.max()))
    print(fmt.format(
        "MHR hip joints (2 skeleton)",
        np.median(vel_skeleton), np.percentile(vel_skeleton, 95),
        vel_skeleton.max()))

    # Velocity ratio: surface / skeleton (>1.0 means surface moves more)
    # Avoid division by zero for static frames
    mask = vel_skeleton > 0.1  # only compare frames with actual movement
    if mask.sum() > 0:
        ratio = vel_surface[mask] / vel_skeleton[mask]
        print(f"\n── Velocity ratio: surface / skeleton (moving frames only) ──")
        print(f"  Moving frames: {mask.sum()} / {len(vel_skeleton)}")
        print(f"  Median ratio:  {np.median(ratio):.2f}x")
        print(f"  P95 ratio:     {np.percentile(ratio, 95):.2f}x")
        print(f"  Max ratio:     {ratio.max():.2f}x")
        if np.median(ratio) > 1.3:
            print(f"  \033[33m→ Surface markers move {np.median(ratio):.1f}x more "
                  f"than skeleton — consider increasing MHR hip weights\033[0m")
        elif np.median(ratio) < 0.7:
            print(f"  \033[33m→ Skeleton moves more than surface — "
                  f"check joint_coords noise\033[0m")
        else:
            print(f"  \033[32m→ Surface and skeleton move similarly\033[0m")
    else:
        print(f"\n  No significant pelvis movement detected (all frames < 0.1 mm)")

    # Per-frame displacement breakdown by axis (X=forward, Y=up, Z=right in TRC)
    surf_centroid = asis_psis.mean(axis=1)  # (N, 3)
    skel_centroid = mhr_hip.mean(axis=1)    # (N, 3)
    surf_range = surf_centroid.max(axis=0) - surf_centroid.min(axis=0)
    skel_range = skel_centroid.max(axis=0) - skel_centroid.min(axis=0)

    print(f"\n── Total range of motion (mm) ──")
    axes = ["X(fwd)", "Y(up)", "Z(right)"]
    print(f"  {'':30s}  {'X(fwd)':>8s}  {'Y(up)':>8s}  {'Z(right)':>8s}")
    print(f"  {'ASIS/PSIS centroid':30s}  {surf_range[0]:8.1f}  "
          f"{surf_range[1]:8.1f}  {surf_range[2]:8.1f}")
    print(f"  {'MHR hip centroid':30s}  {skel_range[0]:8.1f}  "
          f"{skel_range[1]:8.1f}  {skel_range[2]:8.1f}")
    if np.any(skel_range > 0.1):
        axis_ratio = np.where(skel_range > 0.1, surf_range / skel_range, 0)
        print(f"  {'Ratio (surface/skeleton)':30s}  {axis_ratio[0]:8.2f}  "
              f"{axis_ratio[1]:8.2f}  {axis_ratio[2]:8.2f}")

    print(f"{'=' * 75}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate MHR marker atlas against biomechanical standards"
    )
    parser.add_argument("--dir", required=True,
                        help="Output directory with SAM 3D NPZ")
    parser.add_argument("--npz", default=None,
                        help="Override path to SAM 3D NPZ file")
    parser.add_argument("--atlas", default=None,
                        help="Override path to mhr_marker_atlas.json")
    parser.add_argument("--motion", action="store_true",
                        help="Run pelvis motion diagnostic from TRC file")
    args = parser.parse_args()

    run_dir = Path(args.dir)
    if args.npz:
        npz_path = Path(args.npz)
    else:
        # Auto-detect NPZ in directory
        npz_files = list(run_dir.glob("*_sam3d.npz"))
        if not npz_files:
            print(f"ERROR: No *_sam3d.npz found in {run_dir}", file=sys.stderr)
            sys.exit(1)
        npz_path = npz_files[0]

    atlas_path = Path(args.atlas) if args.atlas else None

    print(f"NPZ: {npz_path}")
    checks = validate(npz_path, atlas_path)
    print_results(checks)

    if args.motion:
        # Auto-detect TRC in directory
        trc_files = list(run_dir.glob("*_mhr_markers.trc"))
        if not trc_files:
            print(f"ERROR: No *_mhr_markers.trc found in {run_dir}",
                  file=sys.stderr)
            sys.exit(1)
        motion_diagnostic(trc_files[0])


if __name__ == "__main__":
    main()
