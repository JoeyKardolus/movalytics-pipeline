#!/usr/bin/env python3
"""Export MHR rest-pose mesh as JSON for the interactive marker picker.

43-marker clinical set (LaiUhlrich2022 / SAM4Dcap):
  41 surface markers sited on MHR mesh vertices
  2 computed HJC markers (Bell's method from ASIS/PSIS)

Usage:
    uv run python scripts/export_mhr_mesh_json.py data/output/Max/
    # outputs: data/output/Max/mhr_mesh.json

    uv run python scripts/export_mhr_mesh_json.py --json-to-atlas mapping.json
    # converts picker labeling output → src/core/conversion/mhr_marker_atlas.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── 43-marker clinical set (SAM4Dcap / LaiUhlrich2022) ──
# 41 surface markers grouped by body segment for picker sidebar.
# Marker names use SAM4Dcap `*_study` naming convention.
# Right-side markers are sited manually; left-side auto-mirrors across X=0.
SURFACE_MARKERS: dict[str, list[str]] = {
    "Pelvis": [
        "r.ASIS_study", "L.ASIS_study",
        "r.PSIS_study", "L.PSIS_study",
    ],
    "Knee": [
        "r_knee_study", "r_mknee_study",
        "L_knee_study", "L_mknee_study",
    ],
    "Ankle": [
        "r_ankle_study", "r_mankle_study",
        "L_ankle_study", "L_mankle_study",
    ],
    "Foot": [
        "r_toe_study", "r_5meta_study", "r_calc_study",
        "L_toe_study", "L_5meta_study", "L_calc_study",
    ],
    "Shoulder": [
        "r_shoulder_study", "L_shoulder_study",
    ],
    "Spine": ["C7_study"],
    "Elbow": [
        "r_lelbow_study", "r_melbow_study",
        "L_lelbow_study", "L_melbow_study",
    ],
    "Wrist": [
        "r_lwrist_study", "r_mwrist_study",
        "L_lwrist_study", "L_mwrist_study",
    ],
    "Thigh Cluster": [
        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
    ],
    "Shank Cluster": [
        "r_sh1_study", "r_sh2_study", "r_sh3_study",
        "L_sh1_study", "L_sh2_study", "L_sh3_study",
    ],
}

# Flat set for quick lookup
SURFACE_MARKER_SET: set[str] = set()
for _names in SURFACE_MARKERS.values():
    SURFACE_MARKER_SET.update(_names)

# Computed markers (NOT shown in picker — auto-computed from ASIS/PSIS)
INTERNAL_MARKERS: dict[str, dict] = {
    "RHJC_study": {
        "method": "bell_hjc", "side": "r",
        "needs": ["r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study"],
    },
    "LHJC_study": {
        "method": "bell_hjc", "side": "l",
        "needs": ["r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study"],
    },
}

# L/R auto-symmetry pairs for picker (right → left across X=0)
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

# OpenSim body assignment for each marker (for reference display in picker)
MARKER_BODY: dict[str, str] = {
    "r.ASIS_study": "pelvis", "L.ASIS_study": "pelvis",
    "r.PSIS_study": "pelvis", "L.PSIS_study": "pelvis",
    "C7_study": "torso",
    "r_shoulder_study": "torso", "L_shoulder_study": "torso",
    "r_knee_study": "femur_r", "r_mknee_study": "femur_r",
    "L_knee_study": "femur_l", "L_mknee_study": "femur_l",
    "r_ankle_study": "tibia_r", "r_mankle_study": "tibia_r",
    "L_ankle_study": "tibia_l", "L_mankle_study": "tibia_l",
    "r_toe_study": "calcn_r", "r_5meta_study": "calcn_r", "r_calc_study": "calcn_r",
    "L_toe_study": "calcn_l", "L_5meta_study": "calcn_l", "L_calc_study": "calcn_l",
    "r_lelbow_study": "humerus_r", "r_melbow_study": "humerus_r",
    "L_lelbow_study": "humerus_l", "L_melbow_study": "humerus_l",
    "r_lwrist_study": "radius_r", "r_mwrist_study": "radius_r",
    "L_lwrist_study": "radius_l", "L_mwrist_study": "radius_l",
    "r_thigh1_study": "femur_r", "r_thigh2_study": "femur_r", "r_thigh3_study": "femur_r",
    "L_thigh1_study": "femur_l", "L_thigh2_study": "femur_l", "L_thigh3_study": "femur_l",
    "r_sh1_study": "tibia_r", "r_sh2_study": "tibia_r", "r_sh3_study": "tibia_r",
    "L_sh1_study": "tibia_l", "L_sh2_study": "tibia_l", "L_sh3_study": "tibia_l",
}


def export_mhr_mesh_json(
    output_dir: Path,
    mesh_output: Path | None = None,
) -> Path:
    """Export MHR mesh from pipeline output to viewer JSON.

    Args:
        output_dir: Pipeline output directory containing rest_vertices.npy etc,
            or a _sam3d.npz file.
        mesh_output: Output JSON path. Defaults to output_dir/mhr_mesh.json.

    Returns:
        Path to the output JSON file.
    """
    output_dir = Path(output_dir)

    # Try loading from individual .npy files first, then from .npz
    verts = None
    faces = None

    npy_verts = output_dir / "rest_vertices.npy"
    npz_files = list(output_dir.glob("*_sam3d.npz"))

    if npy_verts.exists():
        verts = np.load(output_dir / "rest_vertices.npy")
        faces_path = output_dir / "rest_faces.npy"
        if faces_path.exists():
            faces = np.load(faces_path)
    elif npz_files:
        npz = np.load(npz_files[0], allow_pickle=True)
        if "rest_vertices" in npz:
            verts = npz["rest_vertices"]
        if "rest_faces" in npz:
            faces = npz["rest_faces"]

    if verts is None:
        raise FileNotFoundError(
            f"No rest_vertices.npy or *_sam3d.npz with rest_vertices in {output_dir}. "
            f"Run the pipeline with --3d-method sam3d first."
        )

    n_verts = verts.shape[0]
    n_faces = faces.shape[0] if faces is not None else 0

    # Build JSON structure (flat arrays for compact transfer)
    mesh_json = {
        "meta": {
            "vertex_count": n_verts,
            "face_count": n_faces,
            "model": "MHR",
            "source": str(output_dir),
            "marker_set": "SAM4Dcap_43",
        },
        "vertices": verts.flatten().tolist(),
        "faces": faces.flatten().tolist() if faces is not None else [],
        # Include marker definitions for the picker
        "marker_groups": {g: names for g, names in SURFACE_MARKERS.items()},
        "lr_pairs": LR_PAIRS,
        "midline_markers": ["C7_study"],
        "marker_bodies": MARKER_BODY,
    }

    if mesh_output is None:
        mesh_output = output_dir / "mhr_mesh.json"

    mesh_output.parent.mkdir(parents=True, exist_ok=True)
    mesh_output.write_text(json.dumps(mesh_json, separators=(",", ":")))
    print(f"[export] MHR mesh JSON: {n_verts} vertices, {n_faces} faces, "
          f"{len(SURFACE_MARKER_SET)} markers → {mesh_output}")

    return mesh_output


def json_to_atlas(
    mapping_json: Path,
    atlas_output: Path | None = None,
) -> Path:
    """Convert labeling JSON from viewer → run symmetry enforcer → generate atlas.

    For the 43-marker set, this delegates to build_mhr_atlas.py's symmetry enforcer.
    The picker JSON has right-side + midline markers; left side is auto-mirrored.

    Args:
        mapping_json: Exported mhr_marker_mapping.json from the viewer.
        atlas_output: Unused (output path controlled by build_mhr_atlas.py).
    """
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.build_mhr_atlas import build_atlas

    # The picker JSON should include a mesh_dir field, or we infer from path
    data = json.loads(mapping_json.read_text())
    mesh_dir = data.get("mesh_dir", None)
    if mesh_dir is None:
        # Try to find rest_vertices.npy in the same directory
        json_dir = mapping_json.parent
        if (json_dir / "rest_vertices.npy").exists():
            mesh_dir = str(json_dir)
        else:
            print("ERROR: Cannot determine mesh directory. "
                  "Use scripts/build_mhr_atlas.py directly with --mesh flag.")
            return mapping_json

    build_atlas(
        picker_json=mapping_json,
        mesh_dir=Path(mesh_dir),
        reference_side="right",
        verbose=True,
    )

    atlas_path = project_root / "src" / "core" / "conversion" / "mhr_marker_atlas.py"
    return atlas_path


def main():
    parser = argparse.ArgumentParser(
        description="Export MHR mesh JSON for marker picker (43-marker clinical set)"
    )
    parser.add_argument(
        "output_dir", nargs="?", default=None,
        help="Pipeline output directory (e.g., data/output/Max/)"
    )
    parser.add_argument(
        "--mesh-output", type=str, default=None,
        help="Output path for mesh JSON (default: <output_dir>/mhr_mesh.json)"
    )
    parser.add_argument(
        "--json-to-atlas", type=str, default=None,
        help="Convert labeling JSON to Python atlas module (via symmetry enforcer)"
    )
    args = parser.parse_args()

    # Atlas conversion mode
    if args.json_to_atlas:
        json_to_atlas(Path(args.json_to_atlas))
        return

    if not args.output_dir:
        parser.error("output_dir is required (unless using --json-to-atlas)")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: Directory not found: {output_dir}")
        return

    export_mhr_mesh_json(
        output_dir,
        Path(args.mesh_output) if args.mesh_output else None,
    )


if __name__ == "__main__":
    main()
