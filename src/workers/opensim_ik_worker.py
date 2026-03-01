#!/usr/bin/env python3
"""OpenSim Scale + IK worker — runs in conda opensim env (Python 3.12 + OpenSim 4.5.2).

Loads LaiUhlrich2022.osim (Rajagopal + Lai/Uhlrich refinements), adds 43 clinical
markers programmatically, scales per-segment from SAM 3D rest-pose geometry,
then runs inverse kinematics on the scaled model.

Model: LaiUhlrich2022 (MIT Use Agreement)
Marker set: 43 SAM4Dcap markers (41 vertex + 2 computed HJC)
Scaling: Per-segment from SAM 3D rest-pose joint positions (MHR body model)
Reference: SAM4Dcap (Cho et al. 2024)

Usage:
    conda run -n opensim python src/workers/opensim_ik_worker.py \
        --trc path/to/markers.trc --output-dir path/to/output \
        --height 1.78 --mass 70 --npz path/to/sam3d.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import opensim as osim
from scipy.spatial.transform import Rotation

# Add project root to path for src/ imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.shared.coordinate_transforms import MHR_BODY_TO_PIPELINE  # noqa: E402
from src.shared.filtering import butterworth_lowpass  # noqa: E402
from src.shared.opensim_constants import (  # noqa: E402
    FK_BODY_COLORS as _FK_BODY_COLORS,
    FK_SKELETON_EDGES as _FK_SKELETON_EDGES,
    FK_SKIP_BODIES as _FK_SKIP_BODIES,
    MHR_SEGMENTS as _MHR_SEGMENTS,
    OPENSIM_MARKERS as MARKERS,
    OPENSIM_MARKER_WEIGHTS as MARKER_WEIGHTS,
    OSIM_SEGMENT_BODY_PAIRS as _OSIM_SEGMENT_BODY_PAIRS,
    TRANSLATION_DOFS as _TRANSLATION_DOFS,
)


def get_trc_time_range(trc_path: str) -> tuple[float, float]:
    """Read TRC header to get time range."""
    with open(trc_path, "r") as f:
        lines = f.readlines()
    keys = lines[1].strip().split("\t")
    vals = lines[2].strip().split("\t")
    kv = dict(zip(keys, vals))
    data_rate = float(kv["DataRate"])
    num_frames = int(kv["NumFrames"])
    start_time = 0.0
    end_time = (num_frames - 1) / data_rate
    return start_time, end_time


def _compute_segment_scales(
    rest_joints: np.ndarray,
    subject_height: float,
    model: osim.Model,
) -> dict[str, float]:
    """Compute per-segment scale factors from SAM 3D rest-pose geometry.

    Args:
        rest_joints: (127, 3) MHR rest-pose joint positions (meters).
        subject_height: Subject height in meters.
        model: OpenSim generic model (initialized).

    Returns:
        Dict mapping segment name → scale factor.
    """
    state = model.initSystem()
    model.realizePosition(state)
    body_set = model.getBodySet()

    def osim_body_pos(name: str) -> np.ndarray:
        b = body_set.get(name)
        p = b.getPositionInGround(state)
        return np.array([p.get(0), p.get(1), p.get(2)])

    # MHR total height: head top to lowest foot point
    mhr_head_y = rest_joints[113, 1]  # c_head Y
    mhr_foot_y = min(rest_joints[8, 1], rest_joints[24, 1])  # l_ball / r_ball Y
    mhr_height = mhr_head_y - mhr_foot_y
    height_scale = subject_height / mhr_height

    print(f"[opensim-ik] MHR rest-pose height: {mhr_height:.3f}m → "
          f"subject: {subject_height:.2f}m (height_scale={height_scale:.3f})")

    scales = {}
    for seg_name, seg_def in _MHR_SEGMENTS.items():
        # MHR segment length (average of all pairs)
        mhr_lengths = []
        for a, b in seg_def["pairs"]:
            mhr_lengths.append(np.linalg.norm(rest_joints[a] - rest_joints[b]))
        mhr_len = np.mean(mhr_lengths)

        # OpenSim generic segment length
        osim_pair = _OSIM_SEGMENT_BODY_PAIRS[seg_name]
        osim_len = np.linalg.norm(osim_body_pos(osim_pair[0]) - osim_body_pos(osim_pair[1]))

        if osim_len < 1e-6:
            scales[seg_name] = height_scale
            continue

        # Per-segment scale = (MHR proportion) * (overall height scale)
        seg_scale = (mhr_len / osim_len) * height_scale
        scales[seg_name] = seg_scale
        print(f"  {seg_name:12s}: MHR={mhr_len:.4f}m  OpenSim={osim_len:.4f}m  "
              f"scale={seg_scale:.3f}")

    return scales


def _apply_per_segment_scales(
    model: osim.Model,
    segment_scales: dict[str, float],
    mass: float,
) -> None:
    """Apply per-segment scale factors to an OpenSim model.

    Scales body geometry, joint frame translations, and marker locations.
    Bodies not covered by any segment measurement get the median scale.

    Args:
        model: OpenSim model to scale (modified in-place).
        segment_scales: Dict mapping segment name → scale factor.
        mass: Subject mass in kg.
    """
    # Build body_name → per-axis scale
    body_scale_map: dict[str, list[float]] = {}
    median_scale = float(np.median(list(segment_scales.values())))

    for seg_name, seg_def in _MHR_SEGMENTS.items():
        s = segment_scales[seg_name]
        for body_name in seg_def["bodies"]:
            if body_name not in body_scale_map:
                body_scale_map[body_name] = [median_scale, median_scale, median_scale]
            for ax in seg_def["axes"]:
                body_scale_map[body_name][ax] = s

    body_set = model.getBodySet()
    joint_set = model.getJointSet()
    marker_set = model.getMarkerSet()

    # Scale each body's geometry
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        name = body.getName()
        sx, sy, sz = body_scale_map.get(name, [median_scale] * 3)
        body.scaleAttachedGeometry(osim.Vec3(sx, sy, sz))

    # Scale joint frame translations
    # Each joint has parent and child offset frames. The translation in the
    # parent frame should scale with the PARENT body's scale, and the child
    # frame translation with the CHILD body's scale.
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        for frame_idx in range(2):
            try:
                frame = joint.get_frames(frame_idx)
                # Determine which body this frame belongs to
                socket = frame.getSocket("parent")
                parent_path = socket.getConnecteePath()
                body_name = parent_path.split("/")[-1]
                sx, sy, sz = body_scale_map.get(body_name, [median_scale] * 3)
                t = frame.get_translation()
                frame.set_translation(osim.Vec3(
                    t.get(0) * sx,
                    t.get(1) * sy,
                    t.get(2) * sz,
                ))
            except Exception:
                pass

    # Scale marker locations (in their parent body's frame)
    for i in range(marker_set.getSize()):
        m = marker_set.get(i)
        body_name = m.getParentFrame().getName()
        sx, sy, sz = body_scale_map.get(body_name, [median_scale] * 3)
        loc = m.get_location()
        m.set_location(osim.Vec3(
            loc.get(0) * sx,
            loc.get(1) * sy,
            loc.get(2) * sz,
        ))

    # Scale mass proportionally
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        body.setMass(body.getMass() * (mass / 75.337))  # LaiUhlrich2022 default mass


def _run_scale_tool(
    generic_model_path: Path,
    trc_path: Path,
    output_dir: Path,
    mass: float,
    height: float,
    npz_path: Path | None = None,
) -> Path:
    """Scale the generic model using SAM 3D rest-pose per-segment scaling.

    When an NPZ file with rest_joint_coords is available, computes per-segment
    scale factors from the MHR body model's rest-pose joint positions. This
    captures individual body proportions (long legs, short torso, etc.) from
    the neural shape model, independent of depth ambiguity.

    Falls back to uniform height-based scaling if no NPZ is provided.

    Args:
        generic_model_path: Path to generic .osim with 43 markers added.
        trc_path: Path to TRC file with 43 markers (mm).
        output_dir: Directory for output files.
        mass: Subject mass in kg.
        height: Subject height in meters.
        npz_path: Path to SAM 3D NPZ with rest_joint_coords (optional).

    Returns:
        Path to scaled .osim model.
    """
    model = osim.Model(str(generic_model_path))

    # Try per-segment scaling from rest-pose geometry
    if npz_path is not None and npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=True)
        if "rest_joint_coords" in data:
            rest_joints = data["rest_joint_coords"]
            print(f"[opensim-ik] Per-segment scaling from SAM 3D rest pose "
                  f"({rest_joints.shape[0]} joints)")
            segment_scales = _compute_segment_scales(rest_joints, height, model)
            # Re-load model (initSystem above consumed it)
            model = osim.Model(str(generic_model_path))
            _apply_per_segment_scales(model, segment_scales, mass)
        else:
            print("[opensim-ik] NPZ has no rest_joint_coords, falling back to "
                  "uniform scaling")
            _apply_uniform_scale(model, height, mass)
    else:
        print("[opensim-ik] No NPZ provided, using uniform height-based scaling")
        _apply_uniform_scale(model, height, mass)

    model.finalizeConnections()
    model.initSystem()

    scaled_model_path = output_dir / f"{trc_path.stem}_scaled.osim"
    model.printToXML(str(scaled_model_path))
    print(f"[opensim-ik] Scaled model → {scaled_model_path.name}")
    return scaled_model_path


def _apply_uniform_scale(model: osim.Model, height: float, mass: float) -> None:
    """Fallback: uniform height-based scaling when no rest-pose data available."""
    state = model.initSystem()
    model.realizePosition(state)
    marker_set = model.getMarkerSet()
    c7 = marker_set.get("C7_study")
    r_calc = marker_set.get("r_calc_study")
    c7_pos = c7.getLocationInGround(state)
    calc_pos = r_calc.getLocationInGround(state)
    model_c7_height = c7_pos.get(1) - calc_pos.get(1)
    model_height = model_c7_height / 0.82
    s = height / model_height
    print(f"[opensim-ik] Uniform scaling: model={model_height:.3f}m → "
          f"subject={height:.2f}m (scale={s:.3f})")

    body_set = model.getBodySet()
    for i in range(body_set.getSize()):
        body_set.get(i).scaleAttachedGeometry(osim.Vec3(s, s, s))
    for i in range(model.getJointSet().getSize()):
        joint = model.getJointSet().get(i)
        for fi in range(2):
            try:
                frame = joint.get_frames(fi)
                t = frame.get_translation()
                frame.set_translation(osim.Vec3(
                    t.get(0) * s, t.get(1) * s, t.get(2) * s))
            except Exception:
                pass
    for i in range(marker_set.getSize()):
        m = marker_set.get(i)
        loc = m.get_location()
        m.set_location(osim.Vec3(
            loc.get(0) * s, loc.get(1) * s, loc.get(2) * s))
    for i in range(body_set.getSize()):
        body_set.get(i).setMass(body_set.get(i).getMass() * (mass / 75.337))


def _extract_sam3d_pelvis_yaw(npz_path: Path, n_frames: int) -> np.ndarray | None:
    """Extract pelvis yaw (rotation around Y) from SAM 3D global_rots.

    MHR body-centric: X=right, Y=up, Z=backward.
    OpenSim: X=forward, Y=up, Z=right.
    Transform: R_osim = T @ R_mhr @ T.T where T maps MHR→OpenSim axes.
    Pelvis yaw = atan2(R_osim[0,2], R_osim[0,0]) in degrees.
    """
    try:
        npz = np.load(npz_path, allow_pickle=True)
        global_rots = npz["global_rots"]  # (N, 127, 3, 3)
        pelvis_rots = global_rots[:, 1]   # (N, 3, 3) — joint 1 = pelvis
    except Exception:
        return None

    # MHR body-centric → OpenSim coordinate transform
    T = MHR_BODY_TO_PIPELINE

    n = min(len(pelvis_rots), n_frames)
    yaw = np.zeros(n)
    for i in range(n):
        R = T @ pelvis_rots[i] @ T.T
        yaw[i] = np.degrees(np.arctan2(R[0, 2], R[0, 0]))

    return yaw[:n_frames] if n >= n_frames else None


def _export_fk_bodies(
    model_path: Path,
    mot_path: Path,
    output_path: Path,
) -> None:
    """Export body transforms from IK result via forward kinematics.

    Loads the scaled OpenSim model and .mot file, runs FK for each frame,
    and saves body positions + rotations + geometry info for Three.js mesh rendering.

    Coordinate system: OpenSim (X=forward, Y=up, Z=right), meters.

    Args:
        model_path: Path to scaled .osim model.
        mot_path: Path to .mot IK result.
        output_path: Path for output .npz file.
    """
    model = osim.Model(str(model_path))

    # Add geometry search path so we can query mesh info
    geometry_dir = model_path.parent.parent.parent / "models" / "opensim" / "Geometry"
    if not geometry_dir.exists():
        geometry_dir = Path(__file__).resolve().parent.parent / "models" / "opensim" / "Geometry"
    if geometry_dir.exists():
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geometry_dir))

    state = model.initSystem()

    # Read .mot data (same parser as _unwrap_mot)
    with open(mot_path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header found in {mot_path}")

    header = lines[header_idx].strip().split("\t")
    data = np.array(
        [[float(v) for v in line.strip().split("\t")] for line in lines[header_idx + 1:]]
    )

    times = data[:, 0]
    n_frames = len(times)

    # Map model coordinates to .mot columns
    coord_set = model.getCoordinateSet()
    coord_map: list[tuple[int, int, bool]] = []  # (coord_idx, col_idx, is_rotational)
    for ci in range(coord_set.getSize()):
        name = coord_set.get(ci).getName()
        if name in header:
            col_idx = header.index(name)
            is_rot = name not in _TRANSLATION_DOFS
            coord_map.append((ci, col_idx, is_rot))

    # Select bodies for export
    body_set = model.getBodySet()
    export_bodies: list[str] = []
    for i in range(body_set.getSize()):
        name = body_set.get(i).getName()
        if name not in _FK_SKIP_BODIES:
            export_bodies.append(name)

    n_bodies = len(export_bodies)
    positions = np.zeros((n_frames, n_bodies, 3))
    rotations = np.zeros((n_frames, n_bodies, 4))  # quaternion [w, x, y, z]

    # Extract attached geometry info per body (mesh file, scale factors)
    # Used by demo page to render bone meshes in Three.js
    geom_info: list[list[tuple[str, list[float]]]] = []  # per body: list of (mesh_file, [sx,sy,sz])
    for bname in export_bodies:
        body = body_set.get(bname)
        body_geoms: list[tuple[str, list[float]]] = []
        try:
            n_geom = body.getPropertyByName("attached_geometry").size()
            for gi in range(n_geom):
                geom = body.get_attached_geometry(gi)
                mesh = osim.Mesh.safeDownCast(geom)
                if mesh:
                    mesh_file = mesh.get_mesh_file()
                    sf = mesh.get_scale_factors()
                    body_geoms.append((
                        mesh_file,
                        [sf.get(0), sf.get(1), sf.get(2)],
                    ))
        except Exception:
            pass
        geom_info.append(body_geoms)

    t0 = time.perf_counter()
    for fi in range(n_frames):
        # Set all coordinates from .mot row
        for ci, col_idx, is_rot in coord_map:
            val = data[fi, col_idx]
            if is_rot:
                val = np.radians(val)  # .mot stores degrees
            coord_set.get(ci).setValue(state, val)

        model.realizePosition(state)

        # Read body transforms
        for bi, bname in enumerate(export_bodies):
            body = body_set.get(bname)
            xform = body.getTransformInGround(state)
            p = xform.p()
            positions[fi, bi] = [p.get(0), p.get(1), p.get(2)]
            # Rotation matrix → quaternion
            R = xform.R()
            q = R.convertRotationToQuaternion()
            rotations[fi, bi] = [q.get(0), q.get(1), q.get(2), q.get(3)]

    dt = time.perf_counter() - t0

    # Build edge indices
    body_idx = {name: i for i, name in enumerate(export_bodies)}
    edges = [
        [body_idx[p], body_idx[c]]
        for p, c in _FK_SKELETON_EDGES
        if p in body_idx and c in body_idx
    ]

    # Body colors
    colors = [_FK_BODY_COLORS.get(name, "#999999") for name in export_bodies]

    # Edge colors (use child body's color)
    edge_colors = [
        _FK_BODY_COLORS.get(c, "#999999")
        for p, c in _FK_SKELETON_EDGES
        if p in body_idx and c in body_idx
    ]

    fps = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 25.0

    # Serialize geometry info as JSON string (NPZ can't store nested lists of tuples)
    import json
    geom_json = json.dumps(geom_info)

    np.savez_compressed(
        output_path,
        body_positions=positions,
        body_rotations=rotations,
        body_names=np.array(export_bodies),
        edges=np.array(edges),
        edge_colors=np.array(edge_colors),
        colors=np.array(colors),
        geometry_info=np.array(geom_json),  # JSON string
        fps=fps,
        n_frames=n_frames,
        times=times,
    )
    print(f"[opensim-ik] FK bodies → {output_path.name} "
          f"({n_frames} frames, {n_bodies} bodies, {dt:.1f}s)")


def _fix_ik_outlier_frames(
    data: np.ndarray,
    header: list[str],
    threshold_deg: float = 45.0,
) -> int:
    """Detect and interpolate sustained IK outlier frames.

    Uses pelvis orientation as the outlier indicator (most sensitive to
    body-wide IK failures). Compares each frame's pelvis quaternion to
    the global median. Frames exceeding the geodesic threshold, plus
    adjacent high-velocity transition frames, have ALL rotational DOFs
    replaced via linear interpolation from valid neighbors.

    Catches sustained multi-frame flips (e.g., 90° tilt over 60+ frames)
    that per-frame velocity-based despike cannot detect.

    Assumes outlier frames comprise < 50% of total.

    Args:
        data: (N, n_cols) MOT data array (modified in-place, degrees).
        header: Column names from MOT file.
        threshold_deg: Geodesic distance from global median to flag outlier.

    Returns:
        Number of outlier frames fixed.
    """
    pelvis_names = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
    pelvis_idxs = [header.index(n) for n in pelvis_names if n in header]
    if len(pelvis_idxs) != 3:
        return 0

    eulers = data[:, pelvis_idxs]  # (N, 3) degrees
    T = len(eulers)
    if T < 3:
        return 0

    # Euler (degrees) -> quaternion
    rots = Rotation.from_euler("ZXY", eulers, degrees=True)
    quats = rots.as_quat()  # (T, 4) [x, y, z, w]

    # Sign continuity (q ≡ -q)
    for t in range(1, T):
        if np.dot(quats[t], quats[t - 1]) < 0:
            quats[t] = -quats[t]

    # Global median quaternion (component-wise + renormalize).
    # Robust as long as outlier frames are < 50% of total.
    median_quat = np.median(quats, axis=0)
    median_quat /= max(np.linalg.norm(median_quat), 1e-8)

    # Geodesic distance to global median
    threshold_rad = np.radians(threshold_deg)
    valid = np.ones(T, dtype=bool)
    n_outliers = 0
    for t in range(T):
        dot_val = np.clip(np.abs(np.dot(quats[t], median_quat)), 0, 1)
        angle = 2 * np.arccos(dot_val)
        if angle > threshold_rad:
            valid[t] = False
            n_outliers += 1

    if n_outliers == 0:
        return 0

    # Grow outlier regions into adjacent transition frames (high tilt velocity).
    # The geodesic threshold catches the plateau but not the gradual ramps
    # into/out of it. Extend outlier regions until tilt velocity drops below
    # 3 deg/frame, so interpolation endpoints sit on stable frames.
    tilt_col = pelvis_idxs[0]  # pelvis_tilt is first in ZXY
    tilt_vel = np.abs(np.diff(data[:, tilt_col]))
    vel_threshold = 3.0  # deg/frame
    changed = True
    while changed:
        changed = False
        for t in range(1, T):
            if not valid[t] and valid[t - 1] and tilt_vel[t - 1] > vel_threshold:
                valid[t - 1] = False
                changed = True
        for t in range(T - 2, -1, -1):
            if not valid[t] and valid[t + 1] and tilt_vel[t] > vel_threshold:
                valid[t + 1] = False
                changed = True

    n_total = int((~valid).sum())

    # Interpolate ALL rotational DOFs (not just pelvis) for outlier frames.
    # The IK solver produces a coupled solution — if pelvis is wrong, all
    # other joints are wrong too. Fixing pelvis alone shifts the error
    # to other DOFs.
    valid_indices = np.where(valid)[0]
    invalid_indices = np.where(~valid)[0]
    if len(valid_indices) < 2:
        return 0

    skip = {"time"} | set(_TRANSLATION_DOFS)
    for col_idx, name in enumerate(header):
        if name in skip:
            continue
        data[invalid_indices, col_idx] = np.interp(
            invalid_indices, valid_indices, data[valid_indices, col_idx],
        )
    return n_total


def _unwrap_mot(mot_path: Path, npz_path: Path | None = None) -> None:
    """Unwrap accumulated rotational angles in MOT file.

    OpenSim's InverseKinematicsTool outputs raw Euler angles that can
    accumulate across revolutions and flip between equivalent Euler
    decompositions (gimbal ambiguity). This post-processing:
    1. Fixes pelvis tilt/rotation flips via quaternion outlier detection
    2. Fixes all 3-DOF ZXY joints via quaternion sign-continuity
    3. Unwraps + normalizes all remaining rotational DOFs to [-180, 180]
    """
    with open(mot_path, "r") as f:
        lines = f.readlines()

    # Find header line (starts with "time")
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            header_idx = i
            break
    if header_idx is None:
        return

    header = lines[header_idx].strip().split("\t")
    data_lines = lines[header_idx + 1:]
    if not data_lines:
        return

    # Parse data into array
    data = np.array(
        [[float(v) for v in line.strip().split("\t")] for line in data_lines]
    )

    # ── Pelvis: geodesic-distance quaternion outlier detection ──
    # Catches sustained multi-frame tilt/rotation flips that despike misses.
    # Separate from ZXY sign continuity (which causes 360° Euler swings on pelvis).
    n_ik_fixed = _fix_ik_outlier_frames(data, header, threshold_deg=45.0)
    if n_ik_fixed > 0:
        print(f"[opensim-ik] IK outlier fix: {n_ik_fixed} frames interpolated (all DOFs)")

    # ── 3-DOF ZXY joints: quaternion sign continuity ──
    # LaiUhlrich2022 uses ZXY Euler for most 3-DOF joints.
    _ZXY_GROUPS = [
        # Pelvis handled separately above via geodesic outlier detection
        ("hip_flexion_r", "hip_adduction_r", "hip_rotation_r"),
        ("hip_flexion_l", "hip_adduction_l", "hip_rotation_l"),
        ("lumbar_extension", "lumbar_bending", "lumbar_rotation"),
        ("arm_flex_r", "arm_add_r", "arm_rot_r"),
        ("arm_flex_l", "arm_add_l", "arm_rot_l"),
    ]

    quat_fixed_dofs: set[str] = set()
    for group_names in _ZXY_GROUPS:
        group_idxs = [header.index(n) for n in group_names if n in header]
        if len(group_idxs) != 3:
            continue
        eulers = data[:, group_idxs]  # (N, 3) in degrees
        rots = Rotation.from_euler("ZXY", eulers, degrees=True)
        quats = rots.as_quat()  # (N, 4) as [x, y, z, w]

        # Ensure quaternion sign continuity
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]

        # Convert back to ZXY Euler
        rots_fixed = Rotation.from_quat(quats)
        eulers_fixed = rots_fixed.as_euler("ZXY", degrees=True)
        data[:, group_idxs] = eulers_fixed
        quat_fixed_dofs.update(group_names)

    n_groups = len([g for g in _ZXY_GROUPS if all(n in header for n in g)])
    print(f"[opensim-ik] Quaternion sign continuity: {len(quat_fixed_dofs)} DOFs "
          f"in {n_groups} ZXY groups")

    # ── All other rotational DOFs: unwrap + normalize ──
    skip_dofs = set(_TRANSLATION_DOFS) | quat_fixed_dofs
    for col_idx, name in enumerate(header):
        if col_idx == 0 or name in skip_dofs:
            continue
        rad = np.radians(data[:, col_idx])
        unwrapped = np.unwrap(rad)
        # Keep unwrapped (continuous) — don't normalize to [-180,180]
        # which would re-introduce discontinuities at ±180°
        data[:, col_idx] = np.degrees(unwrapped)

    # ── Despike: replace outlier frames with interpolated values ──
    # Detects frames where any DOF jumps > threshold from neighbors.
    times = data[:, 0]
    dt = np.median(np.diff(times))
    fps = 1.0 / dt if dt > 0 else 25.0

    total_fixed = 0
    for _pass in range(3):  # multi-pass to catch cascading spikes
        pass_fixed = 0
        for col_idx, name in enumerate(header):
            if col_idx == 0 or name in _TRANSLATION_DOFS:
                continue
            col = data[:, col_idx].copy()
            vel = np.abs(np.diff(col))
            median_vel = np.median(vel)
            threshold = max(median_vel * 3.0, 5.0)  # 3x median or 5°/frame
            spike = np.zeros(len(col), dtype=bool)
            for i in range(1, len(col) - 1):
                if vel[i - 1] > threshold or vel[i] > threshold:
                    spike[i] = True
            if spike.any():
                good = np.where(~spike)[0]
                bad = np.where(spike)[0]
                if len(good) > 1:
                    data[bad, col_idx] = np.interp(bad, good, col[good])
                    pass_fixed += len(bad)
        total_fixed += pass_fixed
        if pass_fixed == 0:
            break
    print(f"[opensim-ik] Despike: {total_fixed} frames interpolated ({_pass + 1} passes)")

    # ── Smooth all rotational DOFs (Butterworth low-pass 6 Hz) ──
    cutoff = 6.0  # Hz
    if fps > 2 * cutoff and data.shape[0] > 12:
        for col_idx, name in enumerate(header):
            if col_idx == 0 or name in _TRANSLATION_DOFS:
                continue
            data[:, col_idx] = butterworth_lowpass(
                data[:, col_idx], cutoff_hz=cutoff, fps=fps, order=2,
            )

    # Rewrite data lines
    new_lines = lines[:header_idx + 1]
    for row in data:
        new_lines.append("\t".join(f"{v:.8f}" for v in row) + "\n")

    with open(mot_path, "w") as f:
        f.writelines(new_lines)

    print(f"[opensim-ik] Unwrapped rotational angles in {mot_path.name}")


def run_ik(
    trc_path: Path,
    output_dir: Path,
    height: float,
    mass: float,
    npz_path: Path | None = None,
    skip_fk: bool = False,
) -> Path:
    """Scale model + run OpenSim IK on TRC data using LaiUhlrich2022.

    Up to 87 markers: 43 surface/HJC + 10 MHR joint centers + 34 MHR70 keypoints.
    Pipeline: add markers → per-segment scale from rest pose → IK on scaled model.

    Args:
        trc_path: Path to TRC file with marker positions (mm).
        output_dir: Directory for output files.
        height: Subject height in meters.
        mass: Subject mass in kg.
        npz_path: Path to SAM 3D NPZ with rest_joint_coords for per-segment scaling.

    Returns:
        Path to output .mot file.
    """
    # Find LaiUhlrich2022.osim model
    model_path = _PROJECT_ROOT / "models" / "opensim" / "LaiUhlrich2022.osim"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Add geometry search path
    geometry_dir = model_path.parent / "Geometry"
    if geometry_dir.exists():
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geometry_dir))

    # ── Step 1: Add markers to generic model ──
    model = osim.Model(str(model_path))

    for name, body_name, x, y, z in MARKERS:
        try:
            body = model.getBodySet().get(body_name)
            marker = osim.Marker()
            marker.setName(name)
            marker.setParentFrame(body)
            marker.set_location(osim.Vec3(x, y, z))
            model.addMarker(marker)
        except Exception as e:
            print(f"  Warning: Could not add marker {name} on {body_name}: {e}",
                  file=sys.stderr)

    model.finalizeConnections()
    model.initSystem()
    print(f"[opensim-ik] LaiUhlrich2022 loaded with "
          f"{model.getMarkerSet().getSize()} markers, "
          f"{model.getCoordinateSet().getSize()} DOFs")

    # Save generic model with markers (ScaleTool needs this as input)
    generic_with_markers = output_dir / f"{trc_path.stem}_generic.osim"
    model.printToXML(str(generic_with_markers))

    # ── Step 2: Scale model via per-segment rest-pose scaling ──
    scaled_model_path = _run_scale_tool(
        generic_with_markers, trc_path, output_dir, mass, height,
        npz_path=npz_path,
    )

    # Load the scaled model for IK
    scaled_model = osim.Model(str(scaled_model_path))
    scaled_model.initSystem()

    # Save as the final output model (overwrite generic)
    final_model_path = output_dir / f"{trc_path.stem}.osim"
    scaled_model.printToXML(str(final_model_path))
    print(f"[opensim-ik] Scaled model → {final_model_path.name}")

    # ── Step 3: IK on scaled model ──
    time_range = get_trc_time_range(str(trc_path))
    mot_path = output_dir / f"{trc_path.stem}_ik.mot"

    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(scaled_model)
    ik_tool.setMarkerDataFileName(str(trc_path))
    ik_tool.setStartTime(time_range[0])
    ik_tool.setEndTime(time_range[1])
    ik_tool.setOutputMotionFileName(str(mot_path))
    ik_tool.setResultsDir(str(output_dir))
    ik_tool.set_accuracy(1e-5)

    # ── Set per-marker IK weights ──
    task_set = ik_tool.getIKTaskSet()
    for name, _, _, _, _ in MARKERS:
        weight = MARKER_WEIGHTS.get(name, 5.0)
        marker_task = osim.IKMarkerTask()
        marker_task.setName(name)
        marker_task.setApply(True)
        marker_task.setWeight(weight)
        task_set.cloneAndAppend(marker_task)

    # No pelvis coordinate constraints — MHR joint centers at weight 50
    # dominate pelvis positioning (skeleton > surface mesh deformation).
    # Surface markers (ASIS/PSIS=15, Bell's HJC=10) provide orientation.

    print(f"[opensim-ik] Running IK: {time_range[0]:.3f}s - {time_range[1]:.3f}s")
    t0 = time.perf_counter()
    ik_tool.run()
    dt = time.perf_counter() - t0
    print(f"[opensim-ik] IK completed in {dt:.1f}s")

    # Unwrap accumulated Euler angles to [-180, 180]
    if mot_path.exists():
        _unwrap_mot(mot_path, npz_path=npz_path)

    # Clean up intermediate files
    generic_with_markers.unlink(missing_ok=True)
    scaled_path = output_dir / f"{trc_path.stem}_scaled.osim"
    scaled_path.unlink(missing_ok=True)

    if mot_path.exists():
        if not skip_fk:
            # Export FK body positions for demo visualization
            fk_path = output_dir / f"{trc_path.stem}_fk_bodies.npz"
            try:
                _export_fk_bodies(final_model_path, mot_path, fk_path)
            except Exception as e:
                print(f"[opensim-ik] FK export failed (non-fatal): {e}", file=sys.stderr)

        print(f"MOT_PATH={mot_path}")
        return mot_path
    else:
        raise RuntimeError(f"IK did not produce output: {mot_path}")


def main():
    parser = argparse.ArgumentParser(description="OpenSim IK worker (LaiUhlrich2022)")
    parser.add_argument("--trc", default=None, help="Input TRC file")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--height", type=float, default=1.75, help="Subject height (m)")
    parser.add_argument("--mass", type=float, default=70.0, help="Subject mass (kg)")
    parser.add_argument("--npz", default=None,
                        help="SAM 3D NPZ with rest_joint_coords for per-segment scaling")
    parser.add_argument("--skip-fk", action="store_true",
                        help="Skip FK body export (saves ~10s)")
    parser.add_argument("--fk-only", action="store_true",
                        help="Only export FK body positions from existing model + mot")
    parser.add_argument("--model", default=None, help="Path to .osim model (for --fk-only)")
    parser.add_argument("--mot", default=None, help="Path to .mot file (for --fk-only)")
    parser.add_argument("--fk-output", default=None, help="Path for FK output NPZ (for --fk-only)")
    args = parser.parse_args()

    if args.fk_only:
        if not args.model or not args.mot:
            print("ERROR: --fk-only requires --model and --mot", file=sys.stderr)
            sys.exit(1)
        model_path = Path(args.model).resolve()
        mot_path = Path(args.mot).resolve()
        if args.fk_output:
            fk_path = Path(args.fk_output).resolve()
        else:
            fk_path = mot_path.parent / f"{mot_path.stem.replace('_ik', '')}_fk_bodies.npz"
        _export_fk_bodies(model_path, mot_path, fk_path)
        return

    if not args.trc or not args.output_dir:
        print("ERROR: --trc and --output-dir required (or use --fk-only)", file=sys.stderr)
        sys.exit(1)

    trc_path = Path(args.trc).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(args.npz).resolve() if args.npz else None

    if not trc_path.exists():
        print(f"ERROR: TRC file not found: {trc_path}", file=sys.stderr)
        sys.exit(1)

    mot_path = run_ik(trc_path, output_dir, args.height, args.mass, npz_path=npz_path,
                      skip_fk=args.skip_fk)
    print(f"[opensim-ik] Output: {mot_path}")


if __name__ == "__main__":
    main()
