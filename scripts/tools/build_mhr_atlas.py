#!/usr/bin/env python3
"""Build MHR marker atlas: symmetry enforcer for 43-marker clinical set.

Two modes:
  --auto: Geometric auto-siting (primary method, no manual input needed)
  --input: Load picker JSON (manual refinement / backup)

Auto-siting algorithm:
  1. Load MHR rest-pose mesh (vertices, faces, joint coordinates)
  2. Assign vertices to segments via nearest-joint
  3. For each marker, find best vertex via geometric surface extrema queries
  4. Mirror right→left across X=0 (sagittal midplane)
  5. Validate symmetry, output atlas files

Manual siting algorithm:
  1. Load picker output JSON (from manual siting session)
  2. Load MHR rest-pose mesh vertices
  3. For each L/R pair: mirror reference side across X=0, find nearest vertex
  4. For midline marker (C7): snap to nearest vertex with |X| < threshold
  5. Output updated atlas .py + .npy + .json

MHR mesh has fixed topology (18,439 vertices). Vertex indices from one
siting work universally across all subjects/body shapes.

Coordinate conventions:
  MHR rest-pose vertices:
    X: negative = body's right side, positive = body's left side
    Y: up (ground ~0.06, head ~1.77)
    Z: positive = anterior (front/chest), negative = posterior (back)
  Mirror plane: X=0 (sagittal midplane)

Usage:
    # Auto-site (primary method)
    uv run python scripts/build_mhr_atlas.py \\
      --auto --mesh data/output/Max/ --verbose

    # From picker JSON (manual refinement)
    uv run python scripts/build_mhr_atlas.py \\
      --input mapping.json --mesh data/output/Max/ --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.core.conversion.mhr_marker_atlas import (
    LR_PAIRS,
    MIDLINE_MARKERS,
    MHR_SURFACE_MARKERS,
)


def _load_rest_vertices(mesh_dir: Path) -> np.ndarray:
    """Load MHR rest-pose vertices from pipeline output.

    Returns:
        (V, 3) vertices in MHR body-centric coords: X=right, Y=up, Z=backward.
    """
    verts_path = mesh_dir / "rest_vertices.npy"
    if verts_path.exists():
        return np.load(verts_path)

    npz_files = list(mesh_dir.glob("*_sam3d.npz"))
    if npz_files:
        npz = np.load(npz_files[0], allow_pickle=True)
        if "rest_vertices" in npz:
            return npz["rest_vertices"]

    raise FileNotFoundError(
        f"No rest_vertices.npy or *_sam3d.npz in {mesh_dir}. "
        f"Run the pipeline with --3d-method sam3d first."
    )


def _find_nearest_vertex(position: np.ndarray, vertices: np.ndarray) -> int:
    """Find nearest mesh vertex to a 3D position. Returns vertex index."""
    dists = np.linalg.norm(vertices - position, axis=1)
    return int(np.argmin(dists))


def _load_picker_json(json_path: Path) -> dict[str, int]:
    """Load marker siting from picker output JSON.

    Expected format: {"mapping": {"marker_name": {"vertex": int, ...}, ...}}

    Returns:
        Dict of marker name → vertex index (only entries with valid vertex data).
    """
    data = json.loads(json_path.read_text())
    mapping = data.get("mapping", {})

    result: dict[str, int] = {}
    for name, entry in mapping.items():
        if entry and isinstance(entry, dict) and "vertex" in entry:
            result[name] = int(entry["vertex"])

    return result


def build_atlas(
    picker_json: Path,
    mesh_dir: Path,
    reference_side: str = "right",
    midline_threshold: float = 0.005,
    verbose: bool = False,
) -> dict[str, int]:
    """Build the 41-marker atlas with symmetry enforcement.

    Args:
        picker_json: Path to picker output JSON with manual siting.
        mesh_dir: Pipeline output directory with rest_vertices.npy.
        reference_side: Which side was manually sited ("right" or "left").
        midline_threshold: Max |X| for midline marker snapping (meters).
        verbose: Print per-marker details.

    Returns:
        Dict of 41 marker names → MHR vertex indices.
    """
    print("[atlas] Loading MHR rest-pose mesh...")
    rest_vertices = _load_rest_vertices(mesh_dir)
    print(f"  Vertices: {rest_vertices.shape}")

    print(f"[atlas] Loading picker siting from {picker_json}...")
    picker_map = _load_picker_json(picker_json)
    print(f"  Loaded {len(picker_map)} sited markers")

    vertex_map: dict[str, int] = {}
    symmetry_errors: list[float] = []

    # ── Process L/R pairs ──
    print(f"[atlas] Enforcing symmetry (reference={reference_side})...")
    for r_name, l_name in LR_PAIRS:
        if reference_side == "right":
            ref_name, mirror_name = r_name, l_name
        else:
            ref_name, mirror_name = l_name, r_name

        if ref_name not in picker_map:
            print(f"  WARNING: reference marker '{ref_name}' not sited, skipping pair")
            continue

        ref_vid = picker_map[ref_name]
        ref_pos = rest_vertices[ref_vid]

        # Mirror across X=0 (sagittal plane)
        # MHR body-centric: X=right, so mirroring X gives left side
        mirrored_pos = ref_pos.copy()
        mirrored_pos[0] = -mirrored_pos[0]

        # Find nearest vertex to mirrored position
        mirror_vid = _find_nearest_vertex(mirrored_pos, rest_vertices)
        mirror_pos = rest_vertices[mirror_vid]

        # Compute symmetry error
        sym_error = float(np.linalg.norm(mirror_pos - mirrored_pos))
        symmetry_errors.append(sym_error)

        vertex_map[ref_name] = ref_vid
        vertex_map[mirror_name] = mirror_vid

        if verbose:
            print(f"  {ref_name:25s} v={ref_vid:6d} → mirror → "
                  f"{mirror_name:25s} v={mirror_vid:6d}  "
                  f"sym_err={sym_error * 1000:.2f}mm")

    # ── Process midline markers ──
    print(f"[atlas] Processing {len(MIDLINE_MARKERS)} midline marker(s)...")
    for name in MIDLINE_MARKERS:
        if name in picker_map:
            vid = picker_map[name]
            pos = rest_vertices[vid]
            # Optionally snap to nearest vertex near midline
            if abs(pos[0]) > midline_threshold:
                # Find nearest vertex with |X| < threshold
                midline_mask = np.abs(rest_vertices[:, 0]) < midline_threshold
                if midline_mask.any():
                    midline_verts = rest_vertices.copy()
                    midline_verts[~midline_mask] = 1e6  # penalize off-midline
                    vid = _find_nearest_vertex(pos, midline_verts)
                    if verbose:
                        print(f"  {name}: snapped to midline vertex {vid} "
                              f"(was off-center by {abs(pos[0]) * 1000:.1f}mm)")
            vertex_map[name] = vid
            if verbose:
                mid_pos = rest_vertices[vid]
                print(f"  {name:25s} v={vid:6d}  X={mid_pos[0] * 1000:.1f}mm")
        else:
            print(f"  WARNING: midline marker '{name}' not sited")

    # ── Summary ──
    n_expected = len(MHR_SURFACE_MARKERS)
    n_mapped = len(vertex_map)
    print(f"\n[atlas] Mapped {n_mapped}/{n_expected} markers")

    if symmetry_errors:
        errors_mm = np.array(symmetry_errors) * 1000
        print(f"  Symmetry error: mean={errors_mm.mean():.2f}mm, "
              f"max={errors_mm.max():.2f}mm, "
              f"<1mm={int((errors_mm < 1).sum())}/{len(errors_mm)}")

    missing = set(MHR_SURFACE_MARKERS.keys()) - set(vertex_map.keys())
    if missing:
        print(f"  Missing markers: {sorted(missing)}")

    # ── Generate output files ──
    _save_atlas_files(vertex_map, _project_root, verbose)

    return vertex_map


def _save_atlas_files(
    vertex_map: dict[str, int],
    project_root: Path,
    verbose: bool = False,
) -> None:
    """Save atlas .py, .npy, and .json files."""
    conv_dir = project_root / "src" / "core" / "conversion"

    # 1. Generate atlas Python module
    atlas_path = conv_dir / "mhr_marker_atlas.py"
    _generate_atlas_module(vertex_map, atlas_path)

    # 2. Save marker indices .npy (sorted by name for deterministic order)
    marker_names = sorted(vertex_map.keys())
    marker_indices = np.array([vertex_map[n] for n in marker_names], dtype=np.int64)
    indices_path = conv_dir / "mhr_marker_indices.npy"
    np.save(indices_path, marker_indices)
    print(f"[atlas] Saved indices ({len(marker_indices)},) → {indices_path}")

    # 3. Save marker names .json
    names_path = conv_dir / "mhr_marker_names.json"
    names_path.write_text(json.dumps(marker_names, indent=2))
    print(f"[atlas] Saved names → {names_path}")

    # 4. Save atlas lookup .json (marker_name → vertex_index, for picker verification)
    atlas_json_path = conv_dir / "mhr_marker_atlas.json"
    atlas_json_path.write_text(json.dumps(vertex_map, indent=2))
    print(f"[atlas] Saved atlas JSON → {atlas_json_path}")


def _generate_atlas_module(
    vertex_map: dict[str, int],
    output_path: Path,
) -> None:
    """Generate Python atlas module with vertex indices."""
    # Read the current module to preserve the class/function definitions
    # Only update the MHR_SURFACE_MARKERS dict values
    entries: list[str] = []

    # Group markers by category for readability
    categories = [
        ("Pelvis", ["r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study"]),
        ("Spine/Torso", ["C7_study"]),
        ("Shoulders", ["r_shoulder_study", "L_shoulder_study"]),
        ("Knees", ["r_knee_study", "r_mknee_study", "L_knee_study", "L_mknee_study"]),
        ("Ankles", ["r_ankle_study", "r_mankle_study", "L_ankle_study", "L_mankle_study"]),
        ("Feet", ["r_toe_study", "r_5meta_study", "r_calc_study",
                  "L_toe_study", "L_5meta_study", "L_calc_study"]),
        ("Elbows", ["r_lelbow_study", "r_melbow_study", "L_lelbow_study", "L_melbow_study"]),
        ("Wrists", ["r_lwrist_study", "r_mwrist_study", "L_lwrist_study", "L_mwrist_study"]),
        ("Thigh clusters", ["r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                           "L_thigh1_study", "L_thigh2_study", "L_thigh3_study"]),
        ("Shank clusters", ["r_sh1_study", "r_sh2_study", "r_sh3_study",
                           "L_sh1_study", "L_sh2_study", "L_sh3_study"]),
    ]

    for cat_name, names in categories:
        entries.append(f"    # ── {cat_name} ──")
        for name in names:
            vid = vertex_map.get(name, 0)
            entries.append(f'    "{name}": {vid},')

    n_mapped = sum(1 for n in vertex_map if vertex_map[n] != 0)
    n_total = len(vertex_map)
    timestamp = datetime.now(timezone.utc).isoformat()

    code = f'''"""MHR 18439-vertex mesh → 43-marker clinical atlas (LaiUhlrich2022 / SAM4Dcap).

41 surface markers (vertex lookup) + 2 computed hip joint centers (Bell's method).
Marker names use SAM4Dcap `*_study` convention for direct TRC↔OpenSim matching.

Generated by: scripts/build_mhr_atlas.py (symmetry enforcer)
Mapped: {n_mapped}/{n_total} surface markers.
Generated: {timestamp}
"""

from __future__ import annotations

import numpy as np

# Surface markers: SAM4Dcap study name → MHR vertex index
# 41 markers: 20 right + 20 left + 1 midline (C7)
MHR_SURFACE_MARKERS: dict[str, int] = {{
{chr(10).join(entries)}
}}

# Ordered marker names (deterministic iteration order)
MHR_SURFACE_MARKER_NAMES: list[str] = list(MHR_SURFACE_MARKERS.keys())

# Vertex indices as array for fast numpy indexing
MHR_MARKER_INDICES: np.ndarray = np.array(
    list(MHR_SURFACE_MARKERS.values()), dtype=np.int64
)

# L/R pairs: (right_name, left_name) for symmetry enforcement
LR_PAIRS: list[tuple[str, str]] = [
    ("r.ASIS_study", "L.ASIS_study"),
    ("r.PSIS_study", "L.PSIS_study"),
    ("r_shoulder_study", "L_shoulder_study"),
    ("r_knee_study", "L_knee_study"),
    ("r_mknee_study", "L_mknee_study"),
    ("r_ankle_study", "L_ankle_study"),
    ("r_mankle_study", "L_mankle_study"),
    ("r_toe_study", "L_toe_study"),
    ("r_5meta_study", "L_5meta_study"),
    ("r_calc_study", "L_calc_study"),
    ("r_lelbow_study", "L_lelbow_study"),
    ("r_melbow_study", "L_melbow_study"),
    ("r_lwrist_study", "L_lwrist_study"),
    ("r_mwrist_study", "L_mwrist_study"),
    ("r_thigh1_study", "L_thigh1_study"),
    ("r_thigh2_study", "L_thigh2_study"),
    ("r_thigh3_study", "L_thigh3_study"),
    ("r_sh1_study", "L_sh1_study"),
    ("r_sh2_study", "L_sh2_study"),
    ("r_sh3_study", "L_sh3_study"),
]

# Midline markers (not mirrored)
MIDLINE_MARKERS: list[str] = ["C7_study"]

# Right-side markers to site manually (picker places these, auto-mirrors to left)
RIGHT_SIDE_MARKERS: list[str] = [r for r, _ in LR_PAIRS]


def compute_hjc_markers(
    vertices: np.ndarray,
    surface: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Compute hip joint center markers via Bell's method (Bell et al. 1990).

    Args:
        vertices: MHR mesh vertices (V, 3) — single frame or rest pose.
            In MHR body-centric coords: X=right, Y=up, Z=backward.
        surface: Override surface marker dict. Defaults to MHR_SURFACE_MARKERS.

    Returns:
        Dict with RHJC_study and LHJC_study positions (3,).
    """
    if surface is None:
        surface = MHR_SURFACE_MARKERS

    def _get(name: str) -> np.ndarray:
        return vertices[surface[name]]

    rasi = _get("r.ASIS_study")
    lasi = _get("L.ASIS_study")
    rpsi = _get("r.PSIS_study")
    lpsi = _get("L.PSIS_study")
    asis_mid = (rasi + lasi) / 2
    psis_mid = (rpsi + lpsi) / 2
    pelvis_width = float(np.linalg.norm(rasi - lasi))

    # Pelvis coordinate system (ISB: X=fwd, Y=up, Z=right)
    z_axis = (rasi - lasi) / (pelvis_width + 1e-8)
    temp = asis_mid - psis_mid
    x_axis = temp / (np.linalg.norm(temp) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    x_axis = np.cross(y_axis, z_axis)

    # Bell coefficients (fraction of pelvis width)
    result: dict[str, np.ndarray] = {{}}
    result["RHJC_study"] = asis_mid + pelvis_width * (
        -0.19 * x_axis - 0.30 * y_axis + 0.36 * z_axis
    )
    result["LHJC_study"] = asis_mid + pelvis_width * (
        -0.19 * x_axis - 0.30 * y_axis - 0.36 * z_axis
    )
    return result


def compute_hjc_markers_batch(
    surface_positions: np.ndarray,
    marker_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Compute HJC markers for a batch of frames.

    Args:
        surface_positions: (N, 41, 3) per-frame surface marker positions.
        marker_names: list of 41 surface marker names (matching dim 1).

    Returns:
        (hjc_positions, hjc_names):
          hjc_positions: (N, 2, 3) per-frame HJC markers [RHJC, LHJC].
          hjc_names: ["RHJC_study", "LHJC_study"].
    """
    name_to_idx = {{n: i for i, n in enumerate(marker_names)}}

    def _get(name: str) -> np.ndarray:
        return surface_positions[:, name_to_idx[name]]  # (N, 3)

    rasi = _get("r.ASIS_study")
    lasi = _get("L.ASIS_study")
    rpsi = _get("r.PSIS_study")
    lpsi = _get("L.PSIS_study")
    asis_mid = (rasi + lasi) / 2
    psis_mid = (rpsi + lpsi) / 2
    pelvis_width = np.linalg.norm(rasi - lasi, axis=1, keepdims=True)  # (N, 1)

    z_axis = (rasi - lasi) / (pelvis_width + 1e-8)
    temp = asis_mid - psis_mid
    x_axis = temp / (np.linalg.norm(temp, axis=1, keepdims=True) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-8)
    x_axis = np.cross(y_axis, z_axis)

    rhip = asis_mid + pelvis_width * (
        -0.19 * x_axis - 0.30 * y_axis + 0.36 * z_axis
    )
    lhip = asis_mid + pelvis_width * (
        -0.19 * x_axis - 0.30 * y_axis - 0.36 * z_axis
    )

    hjc_positions = np.stack([rhip, lhip], axis=1)  # (N, 2, 3)
    hjc_names = ["RHJC_study", "LHJC_study"]
    return hjc_positions, hjc_names


def extract_all_markers(
    vertices: np.ndarray,
    surface: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Extract all 43 markers (41 surface + 2 HJC) from MHR mesh vertices.

    Args:
        vertices: MHR mesh vertices (V, 3) — single frame.
        surface: Override surface marker dict. Defaults to MHR_SURFACE_MARKERS.

    Returns:
        Dict of marker name → (3,) position (43 total).
    """
    if surface is None:
        surface = MHR_SURFACE_MARKERS

    result: dict[str, np.ndarray] = {{}}
    for name, vid in surface.items():
        result[name] = vertices[vid].copy()

    result.update(compute_hjc_markers(vertices, surface))
    return result
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    print(f"[atlas] Generated atlas module: {n_mapped}/{n_total} markers → {output_path}")


def build_atlas_auto(
    mesh_dir: Path,
    verbose: bool = False,
) -> dict[str, int]:
    """Build the 41-marker atlas via geometric auto-siting.

    Args:
        mesh_dir: Pipeline output directory with rest_vertices.npy + rest_joint_coords.npy.
        verbose: Print per-marker details.

    Returns:
        Dict of 41 marker names → MHR vertex indices.
    """
    from scripts.tools.auto_site_markers import auto_site_from_directory

    print("[atlas] Running geometric auto-siting...")
    vertex_map = auto_site_from_directory(mesh_dir, verbose=verbose)

    print(f"\n[atlas] Auto-sited {len(vertex_map)} markers")

    # Save atlas files
    _save_atlas_files(vertex_map, _project_root, verbose)

    return vertex_map


def main():
    parser = argparse.ArgumentParser(
        description="Build MHR marker atlas: symmetry enforcer for 43-marker clinical set"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--auto", action="store_true",
        help="Geometric auto-siting (primary method, no manual input needed)"
    )
    mode.add_argument(
        "--input",
        help="Picker output JSON (mhr_marker_mapping_43.json)"
    )
    parser.add_argument(
        "--mesh", required=True,
        help="Pipeline output directory with rest_vertices.npy"
    )
    parser.add_argument(
        "--reference-side", choices=["right", "left"], default="right",
        help="Which side was manually sited (default: right, only for --input)"
    )
    parser.add_argument(
        "--midline-threshold", type=float, default=0.005,
        help="Max |X| distance for midline snapping in meters (default: 5mm)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-marker details"
    )
    args = parser.parse_args()

    if args.auto:
        build_atlas_auto(
            mesh_dir=Path(args.mesh),
            verbose=args.verbose,
        )
    else:
        build_atlas(
            picker_json=Path(args.input),
            mesh_dir=Path(args.mesh),
            reference_side=args.reference_side,
            midline_threshold=args.midline_threshold,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
