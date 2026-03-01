#!/usr/bin/env python3
"""Automated anatomical landmark detection on MHR mesh geometry.

Places 41 clinical surface markers at bony prominences / surface extrema
identifiable purely from mesh shape — no SMPL dependency.

Algorithm:
  1. Load MHR rest-pose mesh (vertices, faces, joint coordinates)
  2. Assign each vertex to nearest MHR joint → segment map
  3. For each of 21 right-side + 1 midline markers, find best vertex
     via geometric queries (surface extrema near joint centers)
  4. Mirror right→left across X=0 plane
  5. Validate L/R symmetry

Coordinate conventions (empirically verified from rest-pose data):
  MHR rest-pose vertices:
    X: negative = body's right side, positive = body's left side
    Y: up (ground ~0.06, head ~1.77)
    Z: positive = anterior (front/chest), negative = posterior (back)
  Mirror plane: X=0 (sagittal midplane)

Usage:
    # Called by build_mhr_atlas.py --auto
    from scripts.auto_site_markers import auto_site_all
    vertex_map = auto_site_all(rest_vertices, rest_joints, verbose=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# MHR joint indices (from sam3d_joint_map.py)
_MHR_JOINTS = {
    "body_world": 0,
    "root": 1,
    "l_upleg": 2,
    "l_lowleg": 3,
    "l_foot": 4,
    "l_ball": 8,
    "r_upleg": 18,
    "r_lowleg": 19,
    "r_foot": 20,
    "r_ball": 24,
    "c_spine0": 34,
    "c_spine1": 35,
    "c_spine2": 36,
    "c_spine3": 37,
    "r_clavicle": 38,
    "r_uparm": 39,
    "r_lowarm": 40,
    "r_wrist": 42,
    "l_clavicle": 74,
    "l_uparm": 75,
    "l_lowarm": 76,
    "l_wrist": 78,
    "c_neck": 110,
    "c_head": 113,
}


def _assign_segments(
    vertices: np.ndarray,
    joints: np.ndarray,
    joint_indices: list[int],
) -> np.ndarray:
    """Assign each vertex to nearest joint.

    Args:
        vertices: (V, 3) mesh vertices.
        joints: (127, 3) joint positions.
        joint_indices: Which joint indices to use for assignment.

    Returns:
        (V,) array of joint indices, one per vertex.
    """
    joint_positions = joints[joint_indices]
    tree = cKDTree(joint_positions)
    _, nearest = tree.query(vertices)
    return np.array([joint_indices[i] for i in nearest])


def _vertices_near_joints(
    vertices: np.ndarray,
    joints: np.ndarray,
    joint_names: list[str],
    radius: float,
    segment_map: np.ndarray | None = None,
    expand_radius: float = 0.0,
) -> np.ndarray:
    """Get boolean mask of vertices near specified joints.

    Uses segment assignment + distance filter for robust region selection.

    Args:
        vertices: (V, 3) mesh vertices.
        joints: (127, 3) joint positions.
        joint_names: List of MHR joint names to include.
        radius: Max distance from any specified joint.
        segment_map: (V,) segment assignments. If provided, includes all
            vertices assigned to these joints plus distance-filtered neighbors.
        expand_radius: Extra radius for including adjacent-segment vertices.

    Returns:
        (V,) boolean mask.
    """
    joint_idxs = [_MHR_JOINTS[n] for n in joint_names]
    joint_pos = joints[joint_idxs]

    # Distance to closest specified joint
    dists = np.min(
        np.linalg.norm(vertices[:, None, :] - joint_pos[None, :, :], axis=2),
        axis=1,
    )
    mask = dists < radius

    # Optionally include all segment-assigned vertices
    if segment_map is not None:
        seg_mask = np.isin(segment_map, joint_idxs)
        mask = mask | seg_mask

        # Expand to adjacent segments within radius
        if expand_radius > 0:
            expanded = dists < (radius + expand_radius)
            mask = mask | expanded

    return mask


def _find_extremum(
    vertices: np.ndarray,
    mask: np.ndarray,
    axis: int,
    direction: str,
    extra_weight_axis: int | None = None,
    extra_weight_dir: str | None = None,
    extra_weight: float = 0.0,
) -> int:
    """Find vertex at surface extremum within masked region.

    Args:
        vertices: (V, 3) mesh vertices.
        mask: (V,) boolean mask for search region.
        axis: Which axis to find extremum on (0=X, 1=Y, 2=Z).
        direction: "min" or "max".
        extra_weight_axis: Optional secondary axis for tie-breaking.
        extra_weight_dir: "min" or "max" for secondary axis.
        extra_weight: Weight for secondary axis contribution.

    Returns:
        Vertex index (into full vertex array).
    """
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        raise ValueError("No vertices in search region")

    vals = vertices[candidates, axis].copy()

    # Add secondary axis contribution for tie-breaking
    if extra_weight_axis is not None and extra_weight > 0:
        secondary = vertices[candidates, extra_weight_axis]
        if extra_weight_dir == "min":
            secondary = -secondary
        vals = vals + extra_weight * secondary

    if direction == "max":
        best = candidates[np.argmax(vals)]
    else:
        best = candidates[np.argmin(vals)]

    return int(best)


def _find_cluster_markers(
    vertices: np.ndarray,
    mask: np.ndarray,
    proximal_joint: np.ndarray,
    distal_joint: np.ndarray,
    lateral_sign: float,
) -> list[int]:
    """Find 3 cluster markers on anterior surface of a limb segment.

    Places markers at 1/3, 1/2, 2/3 along the segment length,
    on the anterior surface (max Z), spread laterally.

    Args:
        vertices: (V, 3) mesh vertices.
        mask: (V,) boolean mask for segment vertices.
        proximal_joint: (3,) position of proximal joint.
        distal_joint: (3,) position of distal joint.
        lateral_sign: -1 for right side, +1 for left side.

    Returns:
        List of 3 vertex indices [v1, v2, v3].
    """
    candidates = np.where(mask)[0]
    if len(candidates) < 3:
        return [int(candidates[0])] * 3 if len(candidates) > 0 else [0, 0, 0]

    seg_vec = distal_joint - proximal_joint
    seg_len = float(np.linalg.norm(seg_vec))
    if seg_len < 1e-6:
        return [int(candidates[0])] * 3

    seg_dir = seg_vec / seg_len

    # Project vertices onto segment axis (0 = proximal, 1 = distal)
    projections = (vertices[candidates] - proximal_joint) @ seg_dir / seg_len

    result = []
    # Fractions along the segment: 1/3, 1/2, 2/3
    # Lateral offsets: center, lateral, medial (forms triangle)
    for frac, lat_offset in [(0.33, 0.0), (0.5, 0.01), (0.67, 0.0)]:
        # Find vertices near this fraction of the segment
        height_mask = np.abs(projections - frac) < 0.15
        if not height_mask.any():
            height_mask = np.abs(projections - frac) < 0.3

        local_cands = candidates[height_mask]
        if len(local_cands) == 0:
            result.append(int(candidates[0]))
            continue

        # Score: prefer anterior (max Z) and slightly lateral
        # Z axis: +Z = anterior (front of body)
        z_vals = vertices[local_cands, 2]
        x_vals = vertices[local_cands, 0]

        # Anterior surface: high Z score
        z_score = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)

        # Lateral offset for triangle spread
        target_x = np.mean(x_vals) + lateral_sign * lat_offset
        x_score = -np.abs(x_vals - target_x)
        x_score = (x_score - x_score.min()) / (x_score.max() - x_score.min() + 1e-8)

        score = z_score * 0.7 + x_score * 0.3
        best = local_cands[np.argmax(score)]
        result.append(int(best))

    return result


def auto_site_right_side(
    vertices: np.ndarray,
    joints: np.ndarray,
    segment_map: np.ndarray,
    verbose: bool = False,
) -> dict[str, int]:
    """Auto-site the 20 right-side markers + 1 midline (C7).

    MHR coordinate conventions:
      X: negative = body's right side
      Y: positive = up
      Z: positive = anterior (front), negative = posterior (back)

    Args:
        vertices: (V, 3) rest-pose mesh vertices.
        joints: (127, 3) rest-pose joint positions.
        segment_map: (V,) vertex-to-joint assignment.
        verbose: Print per-marker details.

    Returns:
        Dict of 21 marker names → vertex indices.
    """
    result: dict[str, int] = {}

    def _site(name: str, vid: int) -> None:
        result[name] = vid
        if verbose:
            pos = vertices[vid]
            print(f"  {name:25s} v={vid:6d} "
                  f"[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    # ── Helper: get joint position ──
    def _j(name: str) -> np.ndarray:
        return joints[_MHR_JOINTS[name]]

    # ================================================================
    # Pelvis markers
    # ================================================================

    # Right ASIS: most anterior (+Z), lateral (X < 0), near hip height
    r_hip = _j("r_upleg")
    pelvis_mask = _vertices_near_joints(
        vertices, joints, ["root", "r_upleg"], radius=0.15,
        segment_map=segment_map, expand_radius=0.05,
    )
    # Restrict to right side, near hip height
    asis_mask = (
        pelvis_mask
        & (vertices[:, 0] < -0.02)  # right side (X < 0)
        & (np.abs(vertices[:, 1] - r_hip[1]) < 0.06)  # near hip height
    )
    vid = _find_extremum(
        vertices, asis_mask, axis=2, direction="max",  # most anterior (+Z)
        extra_weight_axis=0, extra_weight_dir="min", extra_weight=0.3,  # prefer lateral
    )
    _site("r.ASIS_study", vid)

    # Right PSIS: posterior (-Z), at least 4cm from midline, near ASIS height.
    # PSIS sits on the iliac crest, not the spine — must be lateral enough.
    r_asis_pos = vertices[result["r.ASIS_study"]]
    psis_mask = (
        pelvis_mask
        & (vertices[:, 0] < -0.04)  # at least 4cm from midline (not spine)
        & (vertices[:, 1] > r_asis_pos[1] - 0.02)  # at or above ASIS
        & (vertices[:, 1] < r_asis_pos[1] + 0.06)  # not too high
    )
    vid = _find_extremum(
        vertices, psis_mask, axis=2, direction="min",  # most posterior (-Z)
        extra_weight_axis=0, extra_weight_dir="min", extra_weight=0.5,  # strongly prefer lateral
    )
    _site("r.PSIS_study", vid)

    # ================================================================
    # Knee markers (lateral + medial epicondyles)
    # ================================================================

    r_knee_pos = _j("r_lowleg")
    r_hip_pos = _j("r_upleg")
    knee_mask = _vertices_near_joints(
        vertices, joints, ["r_lowleg"], radius=0.06,
        segment_map=segment_map, expand_radius=0.03,
    )
    # Also include femur-side vertices near the knee
    knee_mask = knee_mask | (
        _vertices_near_joints(vertices, joints, ["r_upleg"], radius=0.08)
        & (np.abs(vertices[:, 1] - r_knee_pos[1]) < 0.05)
    )
    # Epicondyles are at or slightly above the knee joint center
    epicondyle_mask = (
        knee_mask
        & (vertices[:, 1] > r_knee_pos[1] - 0.03)
        & (vertices[:, 1] < r_knee_pos[1] + 0.04)
    )
    if epicondyle_mask.sum() < 5:
        epicondyle_mask = knee_mask & (
            np.abs(vertices[:, 1] - r_knee_pos[1]) < 0.04
        )

    # Right lateral knee: most lateral (min X) at epicondyle level
    vid = _find_extremum(vertices, epicondyle_mask, axis=0, direction="min")
    _site("r_knee_study", vid)

    # Right medial knee: most medial (max X) at epicondyle level
    vid = _find_extremum(vertices, epicondyle_mask, axis=0, direction="max")
    _site("r_mknee_study", vid)

    # ================================================================
    # Ankle markers (lateral + medial malleoli)
    # ================================================================

    r_ankle_pos = _j("r_foot")
    ankle_mask = _vertices_near_joints(
        vertices, joints, ["r_foot", "r_lowleg"], radius=0.06,
        segment_map=segment_map, expand_radius=0.02,
    )
    ankle_mask = ankle_mask & (np.abs(vertices[:, 1] - r_ankle_pos[1]) < 0.04)

    # Right lateral ankle: most lateral (min X)
    vid = _find_extremum(vertices, ankle_mask, axis=0, direction="min")
    _site("r_ankle_study", vid)

    # Right medial ankle: most medial (max X)
    vid = _find_extremum(vertices, ankle_mask, axis=0, direction="max")
    _site("r_mankle_study", vid)

    # ================================================================
    # Foot markers (toe, 5th metatarsal, calcaneus)
    # ================================================================

    r_ball_pos = _j("r_ball")
    r_ankle = _j("r_foot")

    foot_mask = _vertices_near_joints(
        vertices, joints, ["r_foot", "r_ball"], radius=0.12,
        segment_map=segment_map, expand_radius=0.03,
    )
    # Restrict to foot level (below ankle)
    foot_mask = foot_mask & (vertices[:, 1] < r_ankle[1] + 0.02)

    # Right toe: most anterior (+Z) on foot
    toe_mask = foot_mask & (vertices[:, 2] > r_ball_pos[2] - 0.02)
    if toe_mask.sum() < 5:
        toe_mask = foot_mask
    vid = _find_extremum(
        vertices, toe_mask, axis=2, direction="max",  # most anterior
        extra_weight_axis=1, extra_weight_dir="min", extra_weight=0.2,  # prefer ground level
    )
    _site("r_toe_study", vid)

    # Right 5th metatarsal: most lateral (min X) on distal foot
    meta5_mask = foot_mask & (vertices[:, 2] > r_ankle[2])  # distal to ankle in Z
    if meta5_mask.sum() < 5:
        meta5_mask = foot_mask
    vid = _find_extremum(
        vertices, meta5_mask, axis=0, direction="min",  # most lateral (min X)
        extra_weight_axis=1, extra_weight_dir="min", extra_weight=0.3,  # prefer ground level
    )
    _site("r_5meta_study", vid)

    # Right calcaneus: most posterior (-Z) on foot, at ground level
    # The calcaneus is the heel bone — lowest + most posterior point on the foot
    calc_mask = foot_mask & (vertices[:, 2] < r_ankle[2])  # posterior to ankle in Z
    if calc_mask.sum() < 5:
        calc_mask = foot_mask
    # Find ground-level (min Y) posterior (-Z) vertices
    # Use combined score: low Y + low Z
    calc_cands = np.where(calc_mask)[0]
    y_vals = vertices[calc_cands, 1]
    z_vals = vertices[calc_cands, 2]
    # Normalize both to [0, 1] where 0 = best
    y_score = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-8)
    z_score = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
    # Calcaneus = lowest (min Y) AND most posterior (min Z)
    # Weight ground level heavily — the heel marker goes on the back of the heel near ground
    combined = y_score * 0.6 + z_score * 0.4
    vid = int(calc_cands[np.argmin(combined)])
    _site("r_calc_study", vid)

    # ================================================================
    # Shoulder marker
    # ================================================================

    r_shoulder_pos = _j("r_uparm")
    shoulder_mask = _vertices_near_joints(
        vertices, joints, ["r_clavicle", "r_uparm"], radius=0.10,
        segment_map=segment_map, expand_radius=0.03,
    )
    # Acromion: near or above the uparm joint height, lateral
    shoulder_mask = (
        shoulder_mask
        & (vertices[:, 0] < r_shoulder_pos[0] + 0.03)  # right side
        & (vertices[:, 1] > r_shoulder_pos[1] - 0.02)  # at or above shoulder joint
        & (np.linalg.norm(vertices - r_shoulder_pos, axis=1) < 0.10)
    )

    # Right shoulder (acromion): combined lateral + superior scoring
    sh_cands = np.where(shoulder_mask)[0]
    x_score = -vertices[sh_cands, 0]  # more lateral = higher
    y_score = vertices[sh_cands, 1]   # more superior = higher
    x_norm = (x_score - x_score.min()) / (x_score.max() - x_score.min() + 1e-8)
    y_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-8)
    combined = x_norm * 0.5 + y_norm * 0.5
    vid = int(sh_cands[np.argmax(combined)])
    _site("r_shoulder_study", vid)

    # ================================================================
    # C7 (midline, posterior upper spine)
    # ================================================================

    c_neck_pos = _j("c_neck")
    c_spine3_pos = _j("c_spine3")
    c7_mask = _vertices_near_joints(
        vertices, joints, ["c_neck", "c_spine3"], radius=0.08,
        segment_map=segment_map,
    )
    # Strict midline (|X| < 0.01) and near neck level
    c7_mask = (
        c7_mask
        & (np.abs(vertices[:, 0]) < 0.01)  # strict midline
        & (vertices[:, 1] > c_spine3_pos[1] - 0.03)
        & (vertices[:, 1] < c_neck_pos[1] + 0.03)
    )
    # C7 is at the posterior (back) of the neck — most posterior (-Z)
    vid = _find_extremum(vertices, c7_mask, axis=2, direction="min")
    _site("C7_study", vid)

    # ================================================================
    # Elbow markers (lateral + medial epicondyles)
    # ================================================================

    r_elbow_pos = _j("r_lowarm")
    elbow_mask = _vertices_near_joints(
        vertices, joints, ["r_lowarm", "r_uparm"], radius=0.06,
        segment_map=segment_map, expand_radius=0.02,
    )
    # Tight radial constraint from elbow joint to avoid torso vertices
    elbow_dist = np.linalg.norm(vertices - r_elbow_pos, axis=1)
    elbow_mask = elbow_mask & (elbow_dist < 0.06) & (
        np.abs(vertices[:, 1] - r_elbow_pos[1]) < 0.04
    )

    # Right lateral elbow: most lateral (min X)
    vid = _find_extremum(vertices, elbow_mask, axis=0, direction="min")
    _site("r_lelbow_study", vid)

    # Right medial elbow: most medial (max X)
    vid = _find_extremum(vertices, elbow_mask, axis=0, direction="max")
    _site("r_melbow_study", vid)

    # ================================================================
    # Wrist markers (lateral + medial styloid processes)
    # ================================================================

    r_wrist_pos = _j("r_wrist")
    wrist_mask = _vertices_near_joints(
        vertices, joints, ["r_wrist", "r_lowarm"], radius=0.06,
        segment_map=segment_map, expand_radius=0.02,
    )
    wrist_mask = wrist_mask & (
        np.linalg.norm(vertices - r_wrist_pos, axis=1) < 0.05
    )

    # Right lateral wrist: most lateral (min X)
    vid = _find_extremum(vertices, wrist_mask, axis=0, direction="min")
    _site("r_lwrist_study", vid)

    # Right medial wrist: most medial (max X)
    vid = _find_extremum(vertices, wrist_mask, axis=0, direction="max")
    _site("r_mwrist_study", vid)

    # ================================================================
    # Thigh cluster markers (3 on anterior femur)
    # ================================================================

    thigh_mask = _vertices_near_joints(
        vertices, joints, ["r_upleg", "r_lowleg"], radius=0.10,
        segment_map=segment_map,
    )
    # Right side only
    thigh_mask = thigh_mask & (vertices[:, 0] < 0)

    thigh_vids = _find_cluster_markers(
        vertices, thigh_mask,
        proximal_joint=_j("r_upleg"),
        distal_joint=_j("r_lowleg"),
        lateral_sign=-1.0,  # right side = negative X
    )
    _site("r_thigh1_study", thigh_vids[0])
    _site("r_thigh2_study", thigh_vids[1])
    _site("r_thigh3_study", thigh_vids[2])

    # ================================================================
    # Shank cluster markers (3 on anterior tibia)
    # ================================================================

    shank_mask = _vertices_near_joints(
        vertices, joints, ["r_lowleg", "r_foot"], radius=0.10,
        segment_map=segment_map,
    )
    shank_mask = shank_mask & (vertices[:, 0] < 0)

    shank_vids = _find_cluster_markers(
        vertices, shank_mask,
        proximal_joint=_j("r_lowleg"),
        distal_joint=_j("r_foot"),
        lateral_sign=-1.0,
    )
    _site("r_sh1_study", shank_vids[0])
    _site("r_sh2_study", shank_vids[1])
    _site("r_sh3_study", shank_vids[2])

    return result


def _mirror_right_to_left(
    right_map: dict[str, int],
    vertices: np.ndarray,
    verbose: bool = False,
) -> dict[str, int]:
    """Mirror right-side markers to left side across X=0 sagittal plane.

    Right side has negative X, left side has positive X.
    Mirror: negate X coordinate, find nearest vertex.

    Returns:
        Dict of left-side marker names → vertex indices.
    """
    tree = cKDTree(vertices)
    left_map: dict[str, int] = {}

    # Right→Left name mapping
    _r_to_l = {
        "r.ASIS_study": "L.ASIS_study",
        "r.PSIS_study": "L.PSIS_study",
        "r_shoulder_study": "L_shoulder_study",
        "r_knee_study": "L_knee_study",
        "r_mknee_study": "L_mknee_study",
        "r_ankle_study": "L_ankle_study",
        "r_mankle_study": "L_mankle_study",
        "r_toe_study": "L_toe_study",
        "r_5meta_study": "L_5meta_study",
        "r_calc_study": "L_calc_study",
        "r_lelbow_study": "L_lelbow_study",
        "r_melbow_study": "L_melbow_study",
        "r_lwrist_study": "L_lwrist_study",
        "r_mwrist_study": "L_mwrist_study",
        "r_thigh1_study": "L_thigh1_study",
        "r_thigh2_study": "L_thigh2_study",
        "r_thigh3_study": "L_thigh3_study",
        "r_sh1_study": "L_sh1_study",
        "r_sh2_study": "L_sh2_study",
        "r_sh3_study": "L_sh3_study",
    }

    for r_name, l_name in _r_to_l.items():
        if r_name not in right_map:
            continue

        r_vid = right_map[r_name]
        r_pos = vertices[r_vid].copy()

        # Mirror X across sagittal plane
        mirrored = r_pos.copy()
        mirrored[0] = -mirrored[0]

        # Find nearest vertex to mirrored position
        _, l_vid = tree.query(mirrored)
        l_vid = int(l_vid)
        left_map[l_name] = l_vid

        if verbose:
            l_pos = vertices[l_vid]
            sym_err = float(np.linalg.norm(vertices[l_vid] - mirrored))
            print(f"  {r_name:25s} → {l_name:25s} v={l_vid:6d} "
                  f"sym_err={sym_err * 1000:.2f}mm")

    return left_map


def auto_site_all(
    vertices: np.ndarray,
    joints: np.ndarray,
    verbose: bool = False,
) -> dict[str, int]:
    """Auto-site all 41 surface markers on MHR mesh.

    Args:
        vertices: (V, 3) rest-pose mesh vertices.
        joints: (127, 3) rest-pose joint positions.
        verbose: Print per-marker details.

    Returns:
        Dict of 41 marker names → MHR vertex indices.
    """
    # 1. Assign vertices to segments
    # Use main kinematic joints (skip twist/procedural joints)
    segment_joints = [
        _MHR_JOINTS[n] for n in [
            "root", "r_upleg", "r_lowleg", "r_foot", "r_ball",
            "l_upleg", "l_lowleg", "l_foot", "l_ball",
            "c_spine0", "c_spine1", "c_spine2", "c_spine3",
            "r_clavicle", "r_uparm", "r_lowarm", "r_wrist",
            "l_clavicle", "l_uparm", "l_lowarm", "l_wrist",
            "c_neck", "c_head",
        ]
    ]
    if verbose:
        print("[auto-site] Assigning vertices to segments...")
    segment_map = _assign_segments(vertices, joints, segment_joints)

    # 2. Site right-side markers + C7
    if verbose:
        print("[auto-site] Siting right-side markers (20) + midline (1)...")
    right_map = auto_site_right_side(vertices, joints, segment_map, verbose)

    # 3. Mirror to left side
    if verbose:
        print("[auto-site] Mirroring right → left (20 pairs)...")
    left_map = _mirror_right_to_left(right_map, vertices, verbose)

    # 4. Combine
    vertex_map = {}
    vertex_map.update(right_map)
    vertex_map.update(left_map)

    # 5. Summary
    if verbose:
        print(f"\n[auto-site] Total: {len(vertex_map)} markers sited")

        # Check symmetry
        from src.core.conversion.mhr_marker_atlas import LR_PAIRS
        sym_errors = []
        for r_name, l_name in LR_PAIRS:
            if r_name in vertex_map and l_name in vertex_map:
                r_pos = vertices[vertex_map[r_name]]
                l_pos = vertices[vertex_map[l_name]]
                # Perfect symmetry: l_pos = (-r_pos[0], r_pos[1], r_pos[2])
                expected = np.array([-r_pos[0], r_pos[1], r_pos[2]])
                err = float(np.linalg.norm(l_pos - expected))
                sym_errors.append(err)
        if sym_errors:
            errs = np.array(sym_errors) * 1000
            print(f"  L/R symmetry: mean={errs.mean():.2f}mm, "
                  f"max={errs.max():.2f}mm, "
                  f"<1mm={int((errs < 1).sum())}/{len(errs)}")

    return vertex_map


def auto_site_from_directory(
    mesh_dir: Path,
    verbose: bool = False,
) -> dict[str, int]:
    """Auto-site markers from a pipeline output directory.

    Loads rest_vertices.npy and rest_joint_coords.npy from mesh_dir.

    Args:
        mesh_dir: Directory containing rest-pose data.
        verbose: Print details.

    Returns:
        Dict of 41 marker names → MHR vertex indices.
    """
    verts_path = mesh_dir / "rest_vertices.npy"
    joints_path = mesh_dir / "rest_joint_coords.npy"

    if verts_path.exists() and joints_path.exists():
        vertices = np.load(verts_path)
        joints = np.load(joints_path)
    else:
        # Fallback: load from *_sam3d.npz
        npz_files = sorted(mesh_dir.glob("*_sam3d.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"No rest_vertices.npy or *_sam3d.npz in {mesh_dir}")
        npz = np.load(npz_files[0], allow_pickle=True)
        vertices = npz["rest_vertices"]
        joints = npz["rest_joint_coords"]

    if verbose:
        print(f"[auto-site] Loaded mesh: {vertices.shape[0]} vertices, "
              f"{joints.shape[0]} joints from {mesh_dir}")

    return auto_site_all(vertices, joints, verbose)
