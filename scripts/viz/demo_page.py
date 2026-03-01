"""Generate a shareable standalone HTML demo page for the OpenSim FK pipeline.

Auto-discovers all processed videos in data/output/ that have:
  - <name>_fk_bodies.npz
  - <name>_sam3d_angles.png OR joint_angles/ directory
  - matching input video in data/input/

Generates a single standalone HTML file with:
  - Three.js bone mesh animations (auto-orbit, video-synced)
  - Synced input video playback
  - Interactive Chart.js clinical angle charts (L/R overlay, normative bands)

Usage:
    uv run python scripts/viz/demo_page.py
    uv run python scripts/viz/demo_page.py --output-dir data/demo --videos joey walking
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_OUTPUT = PROJECT_ROOT / "data" / "output"
DATA_INPUT = PROJECT_ROOT / "data" / "input"
GEOMETRY_DIR = PROJECT_ROOT / "models" / "opensim" / "Geometry"
DATA_NORMATIVE = PROJECT_ROOT / "data" / "normative"

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Activity labels per demo video (used for normative data selection + display).
# Auto-detected from video name keywords, or set per-video via
# data/output/<name>/metadata.json  {"activity": "walking"}
_ACTIVITY_KEYWORDS: dict[str, str] = {
    "walk": "walking",
    "run": "running",
    "jog": "running",
    "treadmill": "running",
    "sprint": "running",
    "bike": "cycling",
    "cycle": "cycling",
    "pushup": "pushup",
    "push_up": "pushup",
    "squat": "squat",
    "jump": "jumprope",
}


def _detect_activity(name: str) -> str:
    """Detect activity from video name or metadata.json sidecar."""
    # Check for metadata.json sidecar first
    meta_path = DATA_OUTPUT / name / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if "activity" in meta:
                return meta["activity"]
        except (json.JSONDecodeError, KeyError):
            pass
    # Keyword matching on video name
    lower = name.lower()
    for keyword, activity in _ACTIVITY_KEYWORDS.items():
        if keyword in lower:
            return activity
    return "general"

# Target skeleton fps (downsampled from native fps to keep HTML file size down)
_DEMO_FPS = 15.0

# Bodies to skip entirely (no hand detection, too many finger bone meshes)
_SKIP_BODIES = {"hand_r", "hand_l"}


# ---------------------------------------------------------------------------
# VTP mesh parser
# ---------------------------------------------------------------------------


def parse_vtp(path: Path) -> tuple[list[float], list[int]]:
    """Parse VTP file -> (flat_vertices [x0,y0,z0,...], flat_face_indices [i0,i1,i2,...]).

    Handles mixed triangle/quad polygons by triangulating quads into two triangles.
    """
    tree = ET.parse(path)
    piece = tree.getroot().find(".//{http://www.w3.org/2001/XMLSchema-instance}Piece")
    if piece is None:
        piece = tree.getroot().find(".//Piece")
    if piece is None:
        raise ValueError(f"No <Piece> element found in {path}")

    # Parse vertices
    points = piece.find(".//Points/DataArray")
    if points is None or points.text is None:
        raise ValueError(f"No Points/DataArray found in {path}")
    vertices = [float(x) for x in points.text.split()]

    # Parse connectivity and offsets
    connectivity_el = piece.find('.//Polys/DataArray[@Name="connectivity"]')
    offsets_el = piece.find('.//Polys/DataArray[@Name="offsets"]')
    if connectivity_el is None or connectivity_el.text is None:
        raise ValueError(f"No connectivity DataArray found in {path}")
    if offsets_el is None or offsets_el.text is None:
        raise ValueError(f"No offsets DataArray found in {path}")

    conn = [int(x) for x in connectivity_el.text.split()]
    offsets = [int(x) for x in offsets_el.text.split()]

    # Triangulate: split polygons using offsets
    faces: list[int] = []
    prev_off = 0
    for off in offsets:
        poly = conn[prev_off:off]
        n = len(poly)
        if n == 3:
            faces.extend(poly)
        elif n == 4:
            # Split quad into two triangles: [0,1,2] and [0,2,3]
            faces.extend([poly[0], poly[1], poly[2]])
            faces.extend([poly[0], poly[2], poly[3]])
        elif n > 4:
            # Fan triangulation from first vertex
            for i in range(1, n - 1):
                faces.extend([poly[0], poly[i], poly[i + 1]])
        prev_off = off

    return vertices, faces


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _find_input_video(name: str) -> Path | None:
    """Find input video file for a given base name."""
    for ext in _VIDEO_EXTENSIONS:
        p = DATA_INPUT / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _find_fk_bodies(subdir: Path, name: str) -> Path | None:
    """Find FK bodies NPZ, preferring the newer mhr_markers variant."""
    candidates = [
        subdir / f"{name}_mhr_markers_fk_bodies.npz",
        subdir / f"{name}_fk_bodies.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _discover_videos(names: list[str] | None = None) -> list[str]:
    """Return list of video base names that have all required output files."""
    candidates: list[str] = []

    if not DATA_OUTPUT.exists():
        return candidates

    for subdir in sorted(DATA_OUTPUT.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        # Filter to requested names if provided
        if names and name not in names:
            continue
        # Check required output files (FK bodies may be named
        # <name>_fk_bodies.npz or <name>_mhr_markers_fk_bodies.npz)
        npz = _find_fk_bodies(subdir, name)
        angle_dir = subdir / "joint_angles"
        if npz is None or not angle_dir.is_dir():
            continue
        # Check input video exists
        if _find_input_video(name) is None:
            continue
        candidates.append(name)

    return candidates


def _load_fk_data(name: str) -> dict:
    """Load and prepare per-video FK body data for the demo JSON."""
    npz_path = _find_fk_bodies(DATA_OUTPUT / name, name)
    assert npz_path is not None, f"No FK bodies NPZ found for {name}"
    data = dict(np.load(npz_path, allow_pickle=True))

    body_positions: np.ndarray = data["body_positions"]   # (N, n_bodies, 3)
    body_rotations: np.ndarray = data["body_rotations"]   # (N, n_bodies, 4) [w,x,y,z]
    body_names: list[str] = list(data["body_names"])
    edges: np.ndarray = data["edges"]                      # (n_edges, 2)
    edge_colors: list[str] = list(data["edge_colors"])
    colors: list[str] = list(data["colors"])
    geometry_info_str: str = str(data["geometry_info"])
    fps = float(data["fps"])
    n_frames_orig = int(data["n_frames"])

    # Parse geometry info: list of lists per body
    # Each entry: [(mesh_file, [sx, sy, sz]), ...]
    geometry_info: list[list] = json.loads(geometry_info_str)

    # Build body index mask: skip hand_r and hand_l
    n_bodies_orig = len(body_names)
    keep_mask = [i for i in range(n_bodies_orig) if body_names[i] not in _SKIP_BODIES]
    old_to_new = {old: new for new, old in enumerate(keep_mask)}

    # Filter bodies
    body_positions = body_positions[:, keep_mask, :]
    body_rotations = body_rotations[:, keep_mask, :]
    filtered_names = [body_names[i] for i in keep_mask]
    filtered_colors = [colors[i] for i in keep_mask]
    filtered_geometry = [geometry_info[i] for i in keep_mask]

    # Filter edges: only keep edges where both endpoints are in keep_mask
    filtered_edges = []
    filtered_edge_colors = []
    for ei, (p, c) in enumerate(edges):
        p_int, c_int = int(p), int(c)
        if p_int in old_to_new and c_int in old_to_new:
            filtered_edges.append([old_to_new[p_int], old_to_new[c_int]])
            filtered_edge_colors.append(edge_colors[ei])

    n_bodies = len(filtered_names)

    # Downsample to ~_DEMO_FPS
    step = max(1, int(round(fps / _DEMO_FPS)))
    body_positions = body_positions[::step]
    body_rotations = body_rotations[::step]
    effective_fps = fps / step
    n_frames_ds = len(body_positions)

    # Round: positions to 3 decimal places, quaternions to 4
    body_positions = np.round(body_positions, 3)
    body_rotations = np.round(body_rotations, 4)

    # Flatten transforms per frame: [px,py,pz,qw,qx,qy,qz] * n_bodies
    transforms = []
    for fi in range(n_frames_ds):
        frame_flat: list[float] = []
        for bi in range(n_bodies):
            px, py, pz = body_positions[fi, bi].tolist()
            qw, qx, qy, qz = body_rotations[fi, bi].tolist()
            frame_flat.extend([px, py, pz, qw, qx, qy, qz])
        transforms.append(frame_flat)

    # Build body info with mesh assignments
    bodies = []
    all_mesh_names: set[str] = set()
    for bi in range(n_bodies):
        mesh_list = []
        for mesh_file, scale in filtered_geometry[bi]:
            mesh_name = Path(mesh_file).stem
            mesh_list.append({"name": mesh_name, "s": scale})
            all_mesh_names.add(mesh_name)
        bodies.append({
            "name": filtered_names[bi],
            "meshes": mesh_list,
            "color": filtered_colors[bi],
        })

    return {
        "fps": round(effective_fps, 4),
        "native_fps": round(fps, 4),
        "n_frames": n_frames_ds,
        "n_bodies": n_bodies,
        "bodies": bodies,
        "edges": filtered_edges,
        "edge_colors": filtered_edge_colors,
        "transforms": transforms,
        "mesh_names": all_mesh_names,
    }


def _load_angle_data(name: str, step: int) -> dict | None:
    """Load joint angle CSVs, downsample, and return structured dict for JSON.

    Args:
        name: Video base name.
        step: Downsample step (same as skeleton).

    Returns:
        Dict with ``time_s`` list and ``joints`` mapping, or ``None``.
    """
    angle_dir = DATA_OUTPUT / name / "joint_angles"
    if not angle_dir.is_dir():
        return None

    joint_names = [
        "pelvis", "trunk",
        "hip_R", "hip_L", "knee_R", "knee_L",
        "ankle_R", "ankle_L",
        "shoulder_R", "shoulder_L",
        "elbow_R", "elbow_L",
    ]

    joints: dict[str, dict[str, list[float]]] = {}
    time_s: list[float] | None = None

    for jname in joint_names:
        csv_path = angle_dir / f"{name}_angles_{jname}.csv"
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        ds_rows = rows[::step]

        if time_s is None:
            time_s = [round(float(r["time_s"]), 3) for r in ds_rows]

        cols = [k for k in rows[0].keys() if k != "time_s"]
        joint_data: dict[str, list[float]] = {}
        for col in cols:
            joint_data[col] = [round(float(r[col]), 1) for r in ds_rows]

        joints[jname] = joint_data

    if time_s is None or not joints:
        return None

    return {"time_s": time_s, "joints": joints}


# Schwartz2008 DOF name → (joint_group, axis, sign) in the normative JSON.
# sign: +1 keeps original Schwartz convention, -1 negates (and swaps lo/hi).
_SCHWARTZ_MAP: dict[str, tuple[str, str, int]] = {
    "hip_flex_deg": ("HipAngles", "X", 1),
    "hip_abd_deg": ("HipAngles", "Y", 1),
    "hip_rot_deg": ("HipAngles", "Z", 1),
    "knee_flex_deg": ("KneeAngles", "X", 1),
    "ankle_flex_deg": ("AnkleAngles", "X", 1),
    "pelvis_flex_deg": ("PelvisAngles", "X", 1),
    "pelvis_abd_deg": ("PelvisAngles", "Y", 1),
    "pelvis_rot_deg": ("PelvisAngles", "Z", 1),
}


def _load_walking_norms() -> dict | None:
    """Load Schwartz2008 normative gait angle bands (Free walking speed).

    Returns compact dict mapping DOF name to {pct, lo, hi} arrays,
    or ``None`` if the file is missing.
    """
    schwartz_path = DATA_NORMATIVE / "Schwartz2008.json"
    if not schwartz_path.exists():
        # Try to download from pyCGM2 GitHub
        url = (
            "https://raw.githubusercontent.com/pyCGM2/pyCGM2/"
            "Master/Data/normativeData/Schwartz2008.json"
        )
        try:
            import urllib.request

            DATA_NORMATIVE.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, schwartz_path)
            print(f"  Downloaded normative data: {schwartz_path}")
        except Exception as e:
            print(f"  WARNING: could not download normative data: {e}")
            return None

    with open(schwartz_path) as f:
        raw = json.load(f)

    free = raw.get("Schwartz2008", {}).get("Free", {})
    if not free:
        return None

    normative: dict[str, dict[str, list[float]]] = {}
    for dof_name, (joint_group, axis, sign) in _SCHWARTZ_MAP.items():
        vals = free.get(joint_group, {}).get(axis)
        if not vals:
            continue
        # Each entry: [gait_cycle_%, lower_band, upper_band]
        has_data = any(abs(v[1]) > 0.01 or abs(v[2]) > 0.01 for v in vals)
        if not has_data:
            continue
        lo = [round(v[1], 1) for v in vals]
        hi = [round(v[2], 1) for v in vals]
        if sign == -1:
            lo, hi = [round(-h, 1) for h in hi], [round(-l, 1) for l in lo]
        normative[dof_name] = {
            "pct": [v[0] for v in vals],
            "lo": lo,
            "hi": hi,
        }

    return normative if normative else None


def _generate_running_norms() -> dict:
    """Generate running normative curves by interpolating phase-based data.

    Sources: Novacheck 1998, Hamner 2010 (hip/knee/ankle sagittal),
    Schache 2002 / Boyer 2018 (pelvis 3-plane, hip frontal/transverse),
    Hinrichs 1987 / Pontzer 2009 (upper limb ROM).
    Phase points are interpolated to 51-point continuous curves using cubic spline.
    """
    from scipy.interpolate import CubicSpline

    pct_out = list(range(0, 101, 2))  # 51 points
    normative: dict[str, dict[str, list[float]]] = {}

    # --- Phase-based curves (cubic spline) ---
    # Each entry: {dof_name: [(pct, mean, std), ...]}
    phase_dofs: dict[str, list[tuple[float, float, float]]] = {
        # Novacheck 1998, Hamner 2010
        "hip_flex_deg": [
            (0, 45, 5), (20, -15, 5), (40, -15, 5), (55, 20, 5),
            (75, 50, 5), (90, 45, 5), (100, 45, 5),
        ],
        "knee_flex_deg": [
            (0, 25, 5), (20, 10, 5), (40, 45, 5), (55, 90, 10),
            (75, 95, 10), (90, 30, 5), (100, 25, 5),
        ],
        "ankle_flex_deg": [
            (0, 5, 3), (20, 15, 5), (40, -30, 5), (55, -10, 5),
            (75, 0, 5), (90, 5, 3), (100, 5, 3),
        ],
        # Novacheck 1998, Schache 2002 — relatively constant anterior tilt
        "pelvis_flex_deg": [
            (0, 12, 3), (20, 10, 3), (40, 14, 3), (55, 13, 3),
            (75, 11, 3), (90, 12, 3), (100, 12, 3),
        ],
        # Schache 2002, Boyer 2018 — contralateral drop pattern
        "pelvis_abd_deg": [
            (0, -2, 3), (15, 4, 3), (30, -2, 3), (50, -5, 3),
            (65, 2, 3), (80, 5, 3), (90, 0, 3), (100, -2, 3),
        ],
        # Schache 2002, Boyer 2018 — counter-rotates with trunk
        "pelvis_rot_deg": [
            (0, 5, 3), (15, 0, 3), (30, -5, 3), (50, -3, 3),
            (65, 3, 3), (80, 5, 3), (90, 3, 3), (100, 5, 3),
        ],
        # Boyer 2018, Ferber 2003 — adduction peak at midstance
        "hip_abd_deg": [
            (0, -5, 3), (15, -8, 4), (30, -3, 3), (50, 2, 3),
            (65, 0, 3), (80, -2, 3), (90, -4, 3), (100, -5, 3),
        ],
        # Boyer 2018, Willson 2008 — internal rotation during stance
        "hip_rot_deg": [
            (0, 0, 4), (15, 5, 5), (30, 3, 5), (50, -5, 5),
            (65, -3, 4), (80, 0, 4), (90, 0, 4), (100, 0, 4),
        ],
    }

    for dof_name, pts in phase_dofs.items():
        pct_pts = [p[0] for p in pts]
        means = [p[1] for p in pts]
        stds = [p[2] for p in pts]

        cs_mean = CubicSpline(pct_pts, means, bc_type="periodic")
        cs_std = CubicSpline(pct_pts, stds, bc_type="periodic")

        mean_curve = cs_mean(pct_out)
        std_curve = cs_std(pct_out)

        normative[dof_name] = {
            "pct": pct_out,
            "lo": [round(float(m - s), 1) for m, s in zip(mean_curve, std_curve)],
            "hi": [round(float(m + s), 1) for m, s in zip(mean_curve, std_curve)],
        }

    # --- ROM bands (constant lo/hi, no well-defined phase pattern) ---
    # Sign convention: SAM3D trunk_flex is negative for forward lean.
    rom_bands = [
        ("trunk_flex_deg", -15, -5),     # Schache 2002, Folland 2017 (negated for SAM3D)
        ("trunk_abd_deg", -7, 7),        # Schache 2002
        ("trunk_rot_deg", -10, 10),      # Schache 2002, Elphinstone 2013
        ("shoulder_flex_deg", -30, 30),  # Hinrichs 1987, Pontzer 2009
        ("elbow_flex_deg", 70, 120),     # Hinrichs 1987
    ]

    for dof_name, lo, hi in rom_bands:
        normative[dof_name] = {
            "pct": pct_out,
            "lo": [lo] * len(pct_out),
            "hi": [hi] * len(pct_out),
        }

    return normative


def _generate_jump_rom_bands() -> dict:
    """Generate jump ROM reference bands from Ford/Hewett 2003/2005.

    Returns horizontal bands (constant lo/hi across 0-100%) for key DOFs.
    These represent acceptable ROM ranges, not cycle-normalized curves.
    """
    pct = list(range(0, 101, 2))  # 51 points
    normative: dict[str, dict[str, list[float]]] = {}

    # (dof_name, lo, hi) — Ford/Hewett 2003-2015 DVJ/CMJ ROM data
    # Sign convention: SAM3D trunk_flex negative for forward lean,
    # hip_abd positive for adduction.
    rom_bands = [
        ("hip_flex_deg", 0, 50),
        ("knee_flex_deg", 0, 105),
        ("ankle_flex_deg", -20, 18),
        ("hip_abd_deg", 0, 15),          # Ford 2003, Hewett 2005 — adduction peaks (SAM3D sign)
        ("hip_rot_deg", -15, 15),        # Ford 2003, Hewett 2005 — valgus collapse
        ("pelvis_flex_deg", 5, 25),      # Hewett 2005, Ford 2003 — anterior tilt
        ("trunk_flex_deg", -30, 0),      # Hewett 2005, Dingenen 2015 (negated for SAM3D)
        ("trunk_abd_deg", -10, 10),      # Hewett 2005 — lateral trunk inclination
    ]

    for dof_name, lo, hi in rom_bands:
        normative[dof_name] = {
            "pct": pct,
            "lo": [lo] * len(pct),
            "hi": [hi] * len(pct),
        }

    return normative


def _walking_supplementary_norms() -> dict:
    """ROM bands for walking DOFs not covered by Schwartz2008.

    Sources: Krebs 2002, Crosbie 1997 (trunk), Pontzer 2009 / Perry 1992
    (shoulder/elbow arm swing).
    """
    pct = list(range(0, 101, 2))  # 51 points
    normative: dict[str, dict[str, list[float]]] = {}

    # Sign convention: SAM3D trunk_flex is negative for forward lean.
    rom_bands = [
        ("trunk_flex_deg", -5, 3),       # Krebs 2002, Crosbie 1997 (negated for SAM3D)
        ("trunk_abd_deg", -7, 7),        # Krebs 2002, Crosbie 1997
        ("trunk_rot_deg", -8, 8),        # Crosbie 1997, Stokes 1989
        ("shoulder_flex_deg", -15, 30),  # Pontzer 2009, Kubo 2006
        ("elbow_flex_deg", 15, 40),      # Pontzer 2009, Perry 1992
    ]

    for dof_name, lo, hi in rom_bands:
        normative[dof_name] = {
            "pct": pct,
            "lo": [lo] * len(pct),
            "hi": [hi] * len(pct),
        }

    return normative


def _generate_general_rom_bands() -> dict:
    """ROM bands for general/ADL movement from Gates 2016 (PMC4690598).

    These are task-aggregated upper-limb and lower-limb ROM envelopes
    across activities of daily living.
    """
    pct = list(range(0, 101, 2))  # 51 points
    normative: dict[str, dict[str, list[float]]] = {}

    rom_bands = [
        ("hip_flex_deg", 0, 101),        # up to crouching
        ("hip_abd_deg", 0, 17),          # shoe donning
        ("hip_rot_deg", -17, 17),        # coupled with flex
        ("knee_flex_deg", 0, 149),       # crouching tasks
        ("ankle_flex_deg", -26, 26),     # stair climbing
        ("shoulder_flex_deg", 0, 108),   # elevation across ADLs
        ("shoulder_abd_deg", -65, 105),  # humeral plane angle
        ("shoulder_rot_deg", -55, 79),   # humeral rotation
        ("elbow_flex_deg", 0, 121),      # ADL range
    ]

    for dof_name, lo, hi in rom_bands:
        normative[dof_name] = {
            "pct": pct,
            "lo": [lo] * len(pct),
            "hi": [hi] * len(pct),
        }

    return normative


def _generate_cycling_norms() -> dict:
    """ROM bands for cycling from Ericson 1988, Park 2021, Du Toit 2022.

    Lower-limb ranges from ergometer cycling studies; upper-body from
    bike fit anthropometric data.
    """
    pct = list(range(0, 101, 2))  # 51 points
    normative: dict[str, dict[str, list[float]]] = {}

    # Sign convention: SAM3D trunk_flex negative for forward lean.
    # Shoulder/elbow: literature reports anatomical angles, converted to
    # SAM3D ISB convention (flexion from neutral).
    rom_bands = [
        ("hip_flex_deg", 32, 70),        # Ericson 1988, Bini 2013, Park 2021
        ("knee_flex_deg", 46, 112),      # Ericson 1988, Bini 2013, Park 2021
        ("ankle_flex_deg", -2, 22),      # Ericson 1988, Bini 2013, Park 2021
        ("pelvis_flex_deg", 10, 20),     # Bini 2020, Park 2021
        ("trunk_flex_deg", -55, -30),    # Wiggins 2021, Jongerius 2022 (negated for SAM3D)
        ("shoulder_flex_deg", 20, 50),   # Du Toit 2022 — arms forward on hoods
        ("elbow_flex_deg", 10, 30),      # Du Toit 2022 — near-extended
    ]

    for dof_name, lo, hi in rom_bands:
        normative[dof_name] = {
            "pct": pct,
            "lo": [lo] * len(pct),
            "hi": [hi] * len(pct),
        }

    return normative


def _generate_pushup_rom_bands() -> dict:
    """ROM bands for pushup from Gouvali 2005, Freeman 2006, Donkers 1993.

    Shoulder and elbow ROM from push-up variant analysis; trunk/pelvis
    from spine loading studies.
    """
    pct = list(range(0, 101, 2))  # 51 points
    normative: dict[str, dict[str, list[float]]] = {}

    # Sign convention: SAM3D trunk_flex negative for forward lean.
    rom_bands = [
        ("shoulder_flex_deg", 60, 90),   # Gouvali 2005, Suprak 2013
        ("shoulder_abd_deg", 20, 70),    # Gouvali 2005, Lunden 2010
        ("elbow_flex_deg", 0, 90),       # Gouvali 2005, Donkers 1993
        ("trunk_flex_deg", -10, 5),      # Freeman 2006, McGill (negated for SAM3D)
        ("pelvis_flex_deg", 0, 10),      # Freeman 2006, Howarth 2008
    ]

    for dof_name, lo, hi in rom_bands:
        normative[dof_name] = {
            "pct": pct,
            "lo": [lo] * len(pct),
            "hi": [hi] * len(pct),
        }

    return normative


def _load_normative_data() -> dict:
    """Load all normative data, keyed by activity type.

    Returns dict: activity -> {dof_name -> {pct, lo, hi}}.
    """
    result: dict[str, dict] = {}

    # Walking: Schwartz 2008 (continuous 51-point curves) + supplementary ROM
    walking = _load_walking_norms()
    supplementary = _walking_supplementary_norms()
    if walking:
        walking.update(supplementary)
        result["walking"] = walking
    else:
        result["walking"] = supplementary

    # Running: interpolated from Novacheck/Hamner/Schache/Boyer/Hinrichs
    result["running"] = _generate_running_norms()

    # General: Gates 2016 ADL ROM bands
    result["general"] = _generate_general_rom_bands()

    # Jumprope: ROM bands from Ford/Hewett jump norms
    result["jumprope"] = _generate_jump_rom_bands()

    # Cycling: Ericson/Park/Du Toit ROM bands
    result["cycling"] = _generate_cycling_norms()

    # Pushup: Gouvali/Freeman/Donkers ROM bands
    result["pushup"] = _generate_pushup_rom_bands()

    return result


def _get_duration_ffprobe(video_path: Path) -> float | None:
    """Get video container duration via ffprobe.

    Returns the format-level duration, which matches what the browser's
    HTML5 video element uses for playback pacing.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _get_fps_ffprobe(video_path: Path) -> float | None:
    """Get video stream fps via ffprobe (r_frame_rate)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            frac = result.stdout.strip().split("\n")[0].rstrip(",")
            if "/" in frac:
                num, den = frac.split("/")
                return float(num) / float(den)
            return float(frac)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _get_video_duration(video_path: Path) -> float | None:
    """Get video duration in seconds, preferring ffprobe over OpenCV.

    ffprobe reads the container duration metadata, which matches what
    the browser's HTML5 video player uses for playback timing. OpenCV's
    frame_count/fps can disagree for VFR or unusual-metadata videos.
    """
    # Try ffprobe first (matches browser playback duration)
    duration = _get_duration_ffprobe(video_path)
    if duration is not None and duration > 0:
        return duration

    # Fallback to OpenCV
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frame_count > 0:
            return frame_count / fps
        return None
    except ImportError:
        return None


def _copy_or_reencode_video(src: Path, dst: Path) -> None:
    """Copy video to demo output, re-encoding to CFR if ffmpeg is available.

    Re-encoding to constant frame rate ensures the browser plays at a
    consistent, predictable rate that matches the skeleton fps.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        fps = _get_fps_ffprobe(src)
        if fps is not None and fps > 0:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(src),
                    "-c:v", "libx264", "-preset", "fast",
                    "-r", str(round(fps, 2)),
                    "-an",  # strip audio (demo page is muted)
                    str(dst),
                ],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                print(f"    re-encoded to CFR {fps:.1f}fps")
                return
            print(f"    WARNING: ffmpeg re-encode failed, falling back to copy")
    except FileNotFoundError:
        pass  # ffmpeg not installed
    shutil.copy2(src, dst)


def _build_video_entry(name: str, output_dir: Path) -> tuple[dict, set[str], Path, Path]:
    """Build the full JSON entry for one video.

    Returns (entry_dict, mesh_names_set, video_dest_path, input_video_path).
    """
    print(f"  Loading {name}...")

    # FK body data
    fk = _load_fk_data(name)
    mesh_names = fk.pop("mesh_names")

    # Angle CSV data (replaces old base64 PNG)
    step = max(1, int(round(fk["native_fps"] / _DEMO_FPS)))
    angle_data = _load_angle_data(name, step)
    if angle_data:
        n_dofs = sum(len(dofs) for dofs in angle_data["joints"].values())
        n_samples = len(angle_data["time_s"])
        print(f"    angle data: {n_dofs} DOFs, {n_samples} samples")
    else:
        print(f"    WARNING: no angle CSV data found")

    # Input video path -- will be COPIED into output_dir/videos/
    input_video = _find_input_video(name)
    assert input_video is not None  # guaranteed by discovery step
    video_dest = output_dir / "videos" / input_video.name
    video_rel = f"videos/{input_video.name}"

    # Compute duration (ffprobe preferred — matches browser playback)
    duration = _get_video_duration(input_video)
    if duration is None:
        # Estimate from frame count and fps
        duration = fk["n_frames"] / fk["fps"]
    elif fk["n_frames"] > 0 and duration > 0:
        # Recompute effective fps from actual container duration so
        # frame_index = floor(video_time * fps) stays synchronized
        # with the browser's HTML5 video playback timing.
        corrected_fps = fk["n_frames"] / duration
        if abs(corrected_fps - fk["fps"]) > 0.5:
            print(f"    fps correction: {fk['fps']:.2f} -> {corrected_fps:.2f} "
                  f"(ffprobe duration={duration:.3f}s, "
                  f"{fk['n_frames']} downsampled frames)")
            fk["fps"] = round(corrected_fps, 4)

    # Pretty label: capitalize name
    label = name.replace("_", " ").title()

    activity = _detect_activity(name)

    entry = {
        "name": name,
        "label": label,
        "activity": activity,
        "fps": fk["fps"],
        "native_fps": fk["native_fps"],
        "n_frames": fk["n_frames"],
        "n_bodies": fk["n_bodies"],
        "bodies": fk["bodies"],
        "edges": fk["edges"],
        "edge_colors": fk["edge_colors"],
        "transforms": fk["transforms"],
        "angle_data": angle_data,
        "video_path": video_rel,
        "duration": round(duration, 3),
    }

    return entry, mesh_names, video_dest, input_video


def _load_meshes(mesh_names: set[str]) -> dict[str, dict]:
    """Load all needed VTP meshes from Geometry directory.

    Returns dict: mesh_name -> {"v": [flat vertices], "f": [flat face indices]}.
    Vertices are rounded to 5 decimal places.
    """
    meshes: dict[str, dict] = {}
    for mname in sorted(mesh_names):
        vtp_path = GEOMETRY_DIR / f"{mname}.vtp"
        if not vtp_path.exists():
            print(f"    WARNING: mesh file not found: {vtp_path}")
            continue
        try:
            verts, faces = parse_vtp(vtp_path)
            # Round vertices to 5 decimal places
            verts = [round(v, 5) for v in verts]
            meshes[mname] = {"v": verts, "f": faces}
            n_verts = len(verts) // 3
            n_tris = len(faces) // 3
            print(f"    mesh {mname}: {n_verts} verts, {n_tris} tris")
        except Exception as e:
            print(f"    WARNING: failed to parse {vtp_path}: {e}")
    return meshes


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D Human Pose Estimation Demo</title>
<style>
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg:        #0d1117;
    --bg-card:   #161b22;
    --bg-card2:  #1c2128;
    --border:    #30363d;
    --accent:    #58a6ff;
    --accent2:   #3fb950;
    --text:      #e6edf3;
    --text-muted:#8b949e;
    --tab-inactive: #21262d;
    --tab-hover: #2d333b;
    --red:       #f85149;
    --radius:    8px;
    --shadow:    0 4px 24px rgba(0,0,0,0.5);
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    line-height: 1.5;
  }

  /* -- Header ----------------------------------------------------------- */
  #header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px 16px;
    display: flex;
    align-items: baseline;
    gap: 20px;
    flex-wrap: wrap;
  }
  #header h1 {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.3px;
  }
  #header h1 span { color: var(--accent); }
  #header p {
    font-size: 0.82rem;
    color: var(--text-muted);
    white-space: nowrap;
  }
  .badge {
    display: inline-block;
    background: #1f2937;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-left: 4px;
    vertical-align: middle;
  }

  /* -- Tab bar ---------------------------------------------------------- */
  #tabs {
    display: flex;
    gap: 4px;
    padding: 12px 32px 0;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    overflow-x: auto;
    scrollbar-width: thin;
  }
  .tab-btn {
    background: var(--tab-inactive);
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: var(--radius) var(--radius) 0 0;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 20px;
    transition: background 0.15s, color 0.15s;
    white-space: nowrap;
  }
  .tab-btn:hover { background: var(--tab-hover); color: var(--text); }
  .tab-btn.active {
    background: var(--bg-card);
    border-color: var(--border);
    color: var(--accent);
    border-bottom: 1px solid var(--bg-card);
    margin-bottom: -1px;
  }

  /* -- Content area ----------------------------------------------------- */
  .tab-pane { display: none; padding: 20px 32px 32px; }
  .tab-pane.active { display: block; }

  /* Top row: video + skeleton side by side */
  .top-row {
    display: grid;
    grid-template-columns: 38% 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
  }
  .card-header {
    padding: 8px 14px;
    background: var(--bg-card2);
    border-bottom: 1px solid var(--border);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-header .dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
  }

  /* Video card */
  .video-card video {
    width: 100%;
    height: auto;
    display: block;
    background: #000;
    max-height: 340px;
    object-fit: contain;
  }

  /* Skeleton card */
  .skeleton-card { position: relative; }
  .skeleton-canvas {
    display: block;
    width: 100%;
    height: 340px;
    background: var(--bg);
  }

  /* -- Unified timeline ------------------------------------------------- */
  .timeline-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
  }
  .play-btn {
    background: var(--accent);
    border: none;
    border-radius: 50%;
    color: #fff;
    cursor: pointer;
    font-size: 0; /* icon only */
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background 0.15s, transform 0.1s;
  }
  .play-btn:hover { background: #79b8ff; transform: scale(1.05); }
  .play-btn svg { width: 14px; height: 14px; fill: #fff; }
  .timeline-slider {
    flex: 1;
    -webkit-appearance: none;
    appearance: none;
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    outline: none;
    cursor: pointer;
  }
  .timeline-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: var(--accent);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.25);
    transition: box-shadow 0.15s;
  }
  .timeline-slider::-webkit-slider-thumb:hover {
    box-shadow: 0 0 0 5px rgba(88,166,255,0.35);
  }
  .timeline-slider::-moz-range-thumb {
    width: 16px; height: 16px;
    background: var(--accent);
    border-radius: 50%;
    border: none;
    cursor: pointer;
  }
  .time-display {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
    min-width: 80px;
    text-align: right;
    flex-shrink: 0;
  }

  /* -- Angle chart grid ------------------------------------------------- */
  .angles-card { position: relative; overflow: visible; }
  .stats-row {
    display: flex;
    gap: 2px;
    padding: 10px 14px;
    background: var(--bg-card2);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .stat-item {
    flex: 1;
    min-width: 120px;
    background: var(--bg-card);
    border-radius: 4px;
    padding: 6px 10px;
    text-align: center;
  }
  .stat-label {
    font-size: 0.62rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .stat-value {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .stat-sub {
    font-size: 0.65rem;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
  }
  .sym-good { color: #3fb950; }
  .sym-warn { color: #d29922; }
  .sym-bad  { color: #f85149; }
  .stat-clickable { cursor: pointer; position: relative; }
  .stat-clickable:hover { background: var(--bg-card2); }
  .sym-chevron {
    font-size: 0.5rem;
    transition: transform 0.15s;
    display: inline-block;
  }
  .stat-clickable.expanded .sym-chevron { transform: rotate(90deg); }
  .sym-details {
    display: none;
    margin-top: 4px;
    border-top: 1px solid var(--border);
    padding-top: 4px;
  }
  .stat-clickable.expanded .sym-details { display: block; }
  .sym-detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.62rem;
    padding: 1px 0;
    gap: 6px;
  }
  .sym-detail-joint {
    color: var(--text-muted);
    min-width: 30px;
  }
  .sym-detail-rom {
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
  }
  .angles-section-title {
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 14px 4px;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
  }
  .angles-section-title:hover { color: var(--text); }
  .angles-section-title::before {
    content: '\u25B8';
    font-size: 0.8rem;
    transition: transform 0.15s;
  }
  .angles-section:not(.collapsed) .angles-section-title::before {
    transform: rotate(90deg);
  }
  .angles-col-headers {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2px;
    padding: 0 2px;
    background: var(--bg-card2);
  }
  .angles-col-header {
    text-align: center;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    padding: 6px 0 4px;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
  }
  .angles-section.collapsed .angles-col-headers { display: none; }
  .angles-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2px;
    padding: 0 2px 2px;
    background: var(--bg-card2);
  }
  .angles-section.collapsed .angles-grid { display: none; }
  .angle-panel {
    background: var(--bg-card);
    padding: 6px 8px 2px;
    min-height: 180px;
  }
  .angle-panel canvas {
    width: 100% !important;
    height: 160px !important;
  }
  .angle-panel-title {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .angles-header-controls {
    margin-left: auto;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .gait-toggle {
    background: var(--tab-inactive);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 0.68rem;
    padding: 3px 10px;
    transition: all 0.15s;
  }
  .gait-toggle:hover { color: var(--text); background: var(--tab-hover); }
  .gait-toggle.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .reset-zoom-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 0.62rem;
    padding: 2px 8px;
    transition: all 0.15s;
    display: none;
  }
  .reset-zoom-btn.visible { display: inline-block; }
  .reset-zoom-btn:hover { color: var(--text); border-color: var(--text-muted); }
  @media (max-width: 900px) {
    .angles-grid, .angles-col-headers { grid-template-columns: 1fr; }
    .angles-col-headers { display: none; }
  }
  @media (min-width: 901px) and (max-width: 1200px) {
    .angles-grid, .angles-col-headers { grid-template-columns: repeat(2, 1fr); }
    .angles-col-headers .angles-col-header:nth-child(3) { display: none; }
  }

  /* -- Legend ------------------------------------------------------------ */
  .legend {
    position: absolute;
    bottom: 10px;
    right: 10px;
    background: rgba(13,17,23,0.88);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.7rem;
    line-height: 1.8;
    backdrop-filter: blur(4px);
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-muted);
  }
  .legend-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* -- Scrollbar -------------------------------------------------------- */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #484f58; }

  /* -- Loading overlay -------------------------------------------------- */
  .loading-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg);
    font-size: 0.85rem;
    color: var(--text-muted);
    border-radius: var(--radius);
    z-index: 10;
    transition: opacity 0.3s;
  }
  .loading-overlay.hidden { opacity: 0; pointer-events: none; }

  .skel-controls-hint {
    position: absolute;
    bottom: 8px;
    left: 8px;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.4);
    background: rgba(0,0,0,0.3);
    padding: 3px 8px;
    border-radius: 4px;
    pointer-events: none;
  }

  /* -- Responsive ------------------------------------------------------- */
  @media (max-width: 900px) {
    .top-row { grid-template-columns: 1fr; }
    #header { padding: 16px 20px 12px; }
    #tabs { padding: 10px 20px 0; }
    .tab-pane { padding: 16px 20px 24px; }
  }
</style>
</head>
<body>

<!-- -- Header ------------------------------------------------------------ -->
<header id="header">
  <div>
    <h1>3D Human Pose Estimation <span>Pipeline</span></h1>
  </div>
</header>

<!-- -- Tab bar ----------------------------------------------------------- -->
<nav id="tabs"></nav>

<!-- -- Per-video panes --------------------------------------------------- -->
<main id="panes"></main>

<!-- -- Full-page loading screen ------------------------------------------- -->
<div id="page-loader" style="position:fixed;inset:0;z-index:9999;background:var(--bg);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;">
  <div id="loader-text" style="color:var(--text-muted);font-size:0.95rem;">Loading data&hellip;</div>
  <div id="loader-err" style="color:#f85149;font-size:0.85rem;max-width:80%;text-align:center;display:none;"></div>
</div>

<!-- -- Three.js (same CDN + importmap pattern as the Django app) ---------- -->
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>

<!-- Chart.js for interactive angle charts -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.2.0/dist/chartjs-plugin-zoom.min.js"></script>

<script>
// Deferred script (not type="module") — matches the proven Django app approach.
// Uses dynamic import() to load Three.js via the importmap above.
(async function main() {
  const loaderEl = document.getElementById('page-loader');
  const loaderText = document.getElementById('loader-text');
  const loaderErr = document.getElementById('loader-err');

  let THREE, OrbitControls, MESHES, VIDEOS, NORMATIVE;

  try {
    loaderText.textContent = 'Loading Three.js\u2026';
    THREE = await import('three');
    ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js'));

    loaderText.textContent = 'Fetching data.json\u2026';
    const resp = await fetch('data.json');
    if (!resp.ok) throw new Error('Failed to load data.json: HTTP ' + resp.status);
    loaderText.textContent = 'Parsing JSON\u2026';
    const d = await resp.json();
    MESHES = d.meshes;
    VIDEOS = d.videos;
    NORMATIVE = d.normative || {};
    loaderText.textContent = 'Building scenes\u2026';
  } catch (e) {
    loaderErr.style.display = 'block';
    loaderErr.textContent = 'Error: ' + e.message;
    console.error('Demo init failed:', e);
    return;
  }

// -- Helpers --------------------------------------------------------------
function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(0).padStart(2, '0');
  return `${m}:${sec}`;
}

// SVG icons for play/pause button
const ICON_PLAY  = `<svg viewBox="0 0 10 12"><polygon points="1,0 9,6 1,12" fill="#fff"/></svg>`;
const ICON_PAUSE = `<svg viewBox="0 0 10 12"><rect x="0" y="0" width="3.5" height="12" fill="#fff"/><rect x="6.5" y="0" width="3.5" height="12" fill="#fff"/></svg>`;

// -- Clinical angle chart configuration ------------------------------------

// Panel definitions: [id, title, rightJoint, rightCol, leftJoint, leftCol, normKey]
// null leftJoint = single trace (pelvis, trunk)
// normKey = key into NORMATIVE data (null = no normative band available)
const ANGLE_PANELS = {
  lowerBody: [
    ['hip_flex',  'Hip Flex/Extension',     'hip_R','hip_flex_deg',   'hip_L','hip_flex_deg',   'hip_flex_deg'],
    ['hip_abd',   'Hip Abd/Adduction',      'hip_R','hip_abd_deg',   'hip_L','hip_abd_deg',   'hip_abd_deg'],
    ['hip_rot',   'Hip Int/Ext Rotation',   'hip_R','hip_rot_deg',   'hip_L','hip_rot_deg',   'hip_rot_deg'],
    ['knee_flex', 'Knee Flex/Extension',    'knee_R','knee_flex_deg','knee_L','knee_flex_deg','knee_flex_deg'],
    ['ankle_flex','Ankle Dorsi/Plantar',    'ankle_R','ankle_flex_deg','ankle_L','ankle_flex_deg','ankle_flex_deg'],
    ['ankle_abd', 'Ankle Inv/Eversion',     'ankle_R','ankle_abd_deg','ankle_L','ankle_abd_deg',null],
  ],
  trunk: [
    ['trunk_flex',  'Trunk Flex/Extension', 'trunk','trunk_flex_deg', null,null, 'trunk_flex_deg'],
    ['trunk_abd',   'Trunk Lat Bending',    'trunk','trunk_abd_deg',  null,null, 'trunk_abd_deg'],
    ['trunk_rot',   'Trunk Rotation',       'trunk','trunk_rot_deg',  null,null, 'trunk_rot_deg'],
    ['pelvis_flex', 'Pelvis Ant/Post Tilt', 'pelvis','pelvis_flex_deg',null,null,'pelvis_flex_deg'],
    ['pelvis_abd',  'Pelvis Lateral List',  'pelvis','pelvis_abd_deg', null,null,'pelvis_abd_deg'],
    ['pelvis_rot',  'Pelvis Rotation',      'pelvis','pelvis_rot_deg', null,null,'pelvis_rot_deg'],
  ],
  upperBody: [
    ['sh_flex',    'Shoulder Flex/Ext',     'shoulder_R','shoulder_flex_deg','shoulder_L','shoulder_flex_deg','shoulder_flex_deg'],
    ['sh_abd',     'Shoulder Abd/Add',      'shoulder_R','shoulder_abd_deg','shoulder_L','shoulder_abd_deg','shoulder_abd_deg'],
    ['sh_rot',     'Shoulder Int/Ext Rot',  'shoulder_R','shoulder_rot_deg','shoulder_L','shoulder_rot_deg','shoulder_rot_deg'],
    ['elbow_flex', 'Elbow Flex/Extension',  'elbow_R','elbow_flex_deg','elbow_L','elbow_flex_deg','elbow_flex_deg'],
  ],
};

// Chart.js playhead plugin (red dashed vertical line at current time)
// Also draws hover crosshair (gray line) synced across charts
const anglePlayheadPlugin = {
  id: 'anglePlayhead',
  afterDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    const xScale = chart.scales.x;
    if (!xScale) return;

    // Hover crosshair (gray, thin)
    const hoverX = chart.options.plugins.anglePlayhead?.hoverX;
    if (Number.isFinite(hoverX)) {
      const px = xScale.getPixelForValue(hoverX);
      if (Number.isFinite(px) && px >= chartArea.left && px <= chartArea.right) {
        ctx.save();
        ctx.strokeStyle = 'rgba(139, 148, 158, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(px, chartArea.top);
        ctx.lineTo(px, chartArea.bottom);
        ctx.stroke();
        ctx.restore();
      }
    }

    // Playhead (red dashed)
    const t = chart.options.plugins.anglePlayhead?.time;
    if (!Number.isFinite(t)) return;
    const x = xScale.getPixelForValue(t);
    if (!Number.isFinite(x) || x < chartArea.left || x > chartArea.right) return;
    ctx.save();
    ctx.strokeStyle = '#f85149';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

// Stance/swing phase background plugin (only in gait cycle mode)
const gaitPhasePlugin = {
  id: 'gaitPhase',
  beforeDatasetsDraw(chart) {
    if (!chart.options.plugins.gaitPhase?.enabled) return;
    const { ctx, chartArea } = chart;
    const xScale = chart.scales.x;
    if (!xScale) return;
    const toeOffPct = 60;
    const toeOffPx = xScale.getPixelForValue(toeOffPct);

    // Stance label
    ctx.save();
    ctx.font = '8px -apple-system, sans-serif';
    ctx.fillStyle = 'rgba(139, 148, 158, 0.4)';
    ctx.textAlign = 'center';
    const stanceMid = (chartArea.left + toeOffPx) / 2;
    ctx.fillText('Stance', stanceMid, chartArea.top + 10);
    // Swing label
    const swingMid = (toeOffPx + chartArea.right) / 2;
    ctx.fillText('Swing', swingMid, chartArea.top + 10);
    // Toe-off vertical line
    ctx.strokeStyle = 'rgba(139, 148, 158, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(toeOffPx, chartArea.top);
    ctx.lineTo(toeOffPx, chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

Chart.register(anglePlayheadPlugin, gaitPhasePlugin);

// -- Summary stats --------------------------------------------------------
// Cross-correlation symmetry: find the max |r| across time lags.
// For gait (alternating L/R), peak r occurs at ~half-stride lag.
// For synchronous motion, peak r is at lag 0.
function waveformSymmetry(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 20) return 0;
  const aa = a.slice(0, n), bb = b.slice(0, n);
  const ma = aa.reduce((s,v)=>s+v,0)/n;
  const mb = bb.reduce((s,v)=>s+v,0)/n;
  const sa = Math.sqrt(aa.reduce((s,v)=>s+(v-ma)**2,0)/n);
  const sb = Math.sqrt(bb.reduce((s,v)=>s+(v-mb)**2,0)/n);
  if (sa < 0.5 || sb < 0.5) return 0;  // near-constant signal
  const maxLag = Math.min(Math.floor(n / 3), 300);
  let best = 0;
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    let num = 0, cnt = 0;
    const i0 = Math.max(0, lag), i1 = Math.min(n, n + lag);
    for (let i = i0; i < i1; i++) {
      num += (aa[i] - ma) * (bb[i - lag] - mb);
      cnt++;
    }
    if (cnt > 0) {
      const r = Math.max(-1, Math.min(1, num / (cnt * sa * sb)));
      if (Math.abs(r) > Math.abs(best)) best = r;
    }
  }
  return Math.abs(best);
}

function computeSummaryStats(ad) {
  if (!ad || !ad.joints) return null;
  const rows = [];

  // Percentile helper
  function pctl(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }

  // Detrend: subtract rolling mean to remove slow drift, keeping oscillations.
  // Without this, ROM captures drift over the whole video instead of actual
  // movement amplitude (e.g. jumprope hip drifts 27° but oscillates only 4°/cycle).
  function detrend(arr) {
    const n = arr.length;
    const ws = Math.max(30, Math.floor(n / 10));  // ~2-3 second window
    const half = Math.floor(ws / 2);
    const dt = new Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0, cnt = 0;
      for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half); j++) {
        sum += arr[j]; cnt++;
      }
      dt[i] = arr[i] - sum / cnt;
    }
    return dt;
  }

  // Bilateral joints: peak flex + ROM
  const bilateral = [
    ['Hip', 'hip_R', 'hip_L', 'hip_flex_deg', true],
    ['Knee', 'knee_R', 'knee_L', 'knee_flex_deg', true],
    ['Ankle', 'ankle_R', 'ankle_L', 'ankle_flex_deg', false],
  ];
  const corrPairs = [];  // only hip+knee (ankle too noisy for symmetry)
  for (const [label, rJoint, lJoint, col, useForSym] of bilateral) {
    const rData = ad.joints[rJoint]?.[col];
    const lData = ad.joints[lJoint]?.[col];
    if (!rData || !lData) continue;
    // Peak flex: p98 of raw signal (absolute position matters)
    const rPeak = pctl(rData, 98), lPeak = pctl(lData, 98);
    // ROM: p2-p98 of detrended signal (captures oscillation, not drift)
    const rDt = detrend(rData), lDt = detrend(lData);
    const rROM = pctl(rDt, 98) - pctl(rDt, 2);
    const lROM = pctl(lDt, 98) - pctl(lDt, 2);
    rows.push({
      type: 'peak',
      label: `Peak ${label} Flex`,
      rVal: Math.round(rPeak),
      lVal: Math.round(lPeak),
    });
    rows.push({
      type: 'rom',
      label: `${label} ROM`,
      rVal: Math.round(rROM),
      lVal: Math.round(lROM),
    });
    if (useForSym) corrPairs.push([rDt, lDt]);
  }

  // Trunk lean (mean ± std of trunk_flex)
  const trunkData = ad.joints.trunk?.trunk_flex_deg;
  if (trunkData) {
    const mean = trunkData.reduce((s,v)=>s+v,0) / trunkData.length;
    const variance = trunkData.reduce((s,v)=>s+(v-mean)**2,0) / trunkData.length;
    const std = Math.sqrt(variance);
    rows.push({
      type: 'trunk',
      label: 'Trunk Lean',
      mean: Math.abs(mean).toFixed(1),
      std: std.toFixed(1),
    });
  }

  // Combined symmetry: max(ROM-SI, cross-correlation) per joint, then
  // ROM-weighted average.  ROM-SI catches amplitude similarity even when
  // cross-correlation fails on low-amplitude oscillations (e.g. jumprope).
  if (corrPairs.length > 0) {
    const symLabels = ['Hip', 'Knee'];
    const details = corrPairs.map(([a, b], i) => {
      const crossCorr = waveformSymmetry(a, b);
      const rROM = Math.round(pctl(a, 98) - pctl(a, 2));
      const lROM = Math.round(pctl(b, 98) - pctl(b, 2));
      const maxROM = Math.max(rROM, lROM);
      const romSI = maxROM > 0 ? 1 - Math.abs(rROM - lROM) / maxROM : 1;
      const r = Math.max(romSI, crossCorr);
      const avgROM = (rROM + lROM) / 2;
      return { joint: symLabels[i], r, rROM, lROM, avgROM };
    });
    const totalROM = details.reduce((s, d) => s + d.avgROM, 0);
    const avgR = totalROM > 0
      ? details.reduce((s, d) => s + d.r * d.avgROM, 0) / totalROM
      : details.reduce((s, d) => s + d.r, 0) / details.length;
    rows.push({
      type: 'symmetry',
      label: 'Waveform Sym',
      value: avgR,
      details,
    });
  }

  return rows;
}

function renderStatsHTML(stats) {
  if (!stats || !stats.length) return '';
  let html = '<div class="stats-row">';
  for (const s of stats) {
    if (s.type === 'peak' || s.type === 'rom') {
      html += `<div class="stat-item">
        <div class="stat-label">${s.label}</div>
        <div class="stat-value">R: ${s.rVal}\u00b0 L: ${s.lVal}\u00b0</div>
      </div>`;
    } else if (s.type === 'trunk') {
      html += `<div class="stat-item">
        <div class="stat-label">${s.label}</div>
        <div class="stat-value">${s.mean}\u00b0 \u00b1 ${s.std}\u00b0</div>
      </div>`;
    } else if (s.type === 'symmetry') {
      const symClass = s.value > 0.90 ? 'sym-good' : s.value > 0.75 ? 'sym-warn' : 'sym-bad';
      const symLabel = s.value > 0.90 ? 'good' : s.value > 0.75 ? 'fair' : 'asymmetric';
      let detailHTML = '';
      if (s.details) {
        detailHTML = '<div class="sym-details">';
        for (const d of s.details) {
          const dc = d.r > 0.90 ? 'sym-good' : d.r > 0.75 ? 'sym-warn' : 'sym-bad';
          detailHTML += `<div class="sym-detail-row">
            <span class="sym-detail-joint">${d.joint}</span>
            <span class="${dc}">${d.r.toFixed(2)}</span>
            <span class="sym-detail-rom">ROM R:${d.rROM}\u00b0 L:${d.lROM}\u00b0</span>
          </div>`;
        }
        detailHTML += '</div>';
      }
      html += `<div class="stat-item stat-clickable" onclick="this.classList.toggle('expanded')">
        <div class="stat-label">${s.label} <span class="sym-chevron">\u25b8</span></div>
        <div class="stat-value ${symClass}">${s.value.toFixed(2)}</div>
        <div class="stat-sub">${symLabel}</div>
        ${detailHTML}
      </div>`;
    }
  }
  html += '</div>';
  return html;
}

// -- Gait cycle detection -------------------------------------------------
function detectGaitCycles(ad) {
  // Find knee flexion troughs (heel strikes) from right knee
  const kneeData = ad.joints.knee_R?.knee_flex_deg;
  if (!kneeData || kneeData.length < 20) return null;

  const N = kneeData.length;
  const troughs = [];
  // Simple trough detection: local minima with minimum separation
  const minSep = Math.max(5, Math.floor(N / 30)); // at least 5 frames between heel strikes
  for (let i = 2; i < N - 2; i++) {
    if (kneeData[i] < kneeData[i-1] && kneeData[i] < kneeData[i+1] &&
        kneeData[i] < kneeData[i-2] && kneeData[i] < kneeData[i+2]) {
      if (troughs.length === 0 || i - troughs[troughs.length - 1] >= minSep) {
        troughs.push(i);
      }
    }
  }

  if (troughs.length < 2) return null;

  // Resample each stride to 101 points (0-100%)
  const nPts = 101;
  const cycles = {};

  for (const jName of Object.keys(ad.joints)) {
    for (const colName of Object.keys(ad.joints[jName])) {
      const data = ad.joints[jName][colName];
      const strides = [];
      for (let s = 0; s < troughs.length - 1; s++) {
        const start = troughs[s], end = troughs[s + 1];
        const strideLen = end - start;
        const resampled = [];
        for (let p = 0; p < nPts; p++) {
          const srcIdx = start + (p / (nPts - 1)) * strideLen;
          const lo = Math.floor(srcIdx), hi = Math.min(lo + 1, N - 1);
          const frac = srcIdx - lo;
          resampled.push(data[lo] * (1 - frac) + data[hi] * frac);
        }
        strides.push(resampled);
      }
      // Compute mean and SD across strides
      const mean = new Array(nPts).fill(0);
      for (const st of strides) {
        for (let p = 0; p < nPts; p++) mean[p] += st[p];
      }
      for (let p = 0; p < nPts; p++) mean[p] /= strides.length;

      const sd = new Array(nPts).fill(0);
      if (strides.length > 1) {
        for (const st of strides) {
          for (let p = 0; p < nPts; p++) sd[p] += (st[p] - mean[p]) ** 2;
        }
        for (let p = 0; p < nPts; p++) sd[p] = Math.sqrt(sd[p] / (strides.length - 1));
      }

      const key = `${jName}|${colName}`;
      cycles[key] = { mean, sd, nStrides: strides.length };
    }
  }

  return { cycles, nStrides: troughs.length - 1 };
}

// -- Build angle charts HTML ----------------------------------------------
function buildAngleChartsHTML(vd) {
  if (!vd.angle_data) {
    return `<div class="card angles-card">
      <div class="card-header">
        <span class="dot" style="background:#f0883e"></span>
        Clinical Joint Angles &mdash; No data
      </div>
    </div>`;
  }

  const n = vd.name;
  const ad = vd.angle_data;
  const nDofs = Object.values(ad.joints).reduce((s, j) => s + Object.keys(j).length, 0);
  const stats = computeSummaryStats(ad);

  function sectionHTML(id, title, panelDefs) {
    let panels = '';
    for (const [panelId, panelTitle] of panelDefs) {
      panels += `<div class="angle-panel">
        <div class="angle-panel-title">${panelTitle} (\u00b0)</div>
        <canvas id="ac-${n}-${panelId}"></canvas>
      </div>`;
    }
    return `<div class="angles-section collapsed" id="asec-${n}-${id}">
      <div class="angles-section-title" onclick="this.closest('.angles-section').classList.toggle('collapsed')">${title}</div>
      <div class="angles-col-headers">
        <div class="angles-col-header">Sagittal Plane</div>
        <div class="angles-col-header">Coronal Plane</div>
        <div class="angles-col-header">Transverse Plane</div>
      </div>
      <div class="angles-grid">${panels}</div>
    </div>`;
  }

  const isGait = ['walking', 'running'].includes(vd.activity);
  const actLabel = (vd.activity || 'unknown').replace(/^./, c => c.toUpperCase());

  return `<div class="card angles-card" id="angles-card-${n}">
    <div class="card-header">
      <span class="dot" style="background:#f0883e"></span>
      Clinical Joint Angles (${nDofs} DOF) &mdash; ${actLabel}
      <div class="angles-header-controls">
        <span style="font-weight:400;color:#484f58;font-size:0.68rem">
          R = solid blue &bull; L = dashed orange
        </span>
        ${isGait ? `<button class="gait-toggle" id="gait-btn-${n}" title="Toggle gait cycle normalization"
                onclick="toggleGaitCycle('${n}')">% Gait Cycle</button>` : ''}
        <button class="reset-zoom-btn" id="reset-zoom-${n}" title="Reset chart zoom"
                onclick="resetAngleZoom('${n}')">Reset Zoom</button>
      </div>
    </div>
    ${renderStatsHTML(stats)}
    ${sectionHTML('lower', 'Lower Body', ANGLE_PANELS.lowerBody)}
    ${sectionHTML('trunk', 'Trunk & Pelvis', ANGLE_PANELS.trunk)}
    ${sectionHTML('upper', 'Upper Body', ANGLE_PANELS.upperBody)}
  </div>`;
}

// -- Initialize Chart.js angle charts -------------------------------------
function initAngleCharts(name, vd) {
  if (!vd.angle_data) return [];

  const ad = vd.angle_data;
  const timeArr = ad.time_s;
  const duration = timeArr[timeArr.length - 1];
  const charts = [];

  const darkGrid = 'rgba(48, 54, 61, 0.7)';
  const darkTick = '#8b949e';

  const allPanels = [
    ...ANGLE_PANELS.lowerBody,
    ...ANGLE_PANELS.trunk,
    ...ANGLE_PANELS.upperBody,
  ];

  for (const [panelId, title, rJoint, rCol, lJoint, lCol, normKey] of allPanels) {
    const canvas = document.getElementById(`ac-${name}-${panelId}`);
    if (!canvas) continue;

    const datasets = [];

    // For non-gait activities, show ROM bands directly on the time-series view
    // (gait-cycle-normalized curves for walking/running are shown via the toggle)
    const actNorms = NORMATIVE?.[vd.activity] || {};
    const normData = normKey ? actNorms[normKey] : null;
    if (normData && !['walking', 'running'].includes(vd.activity)) {
      const lo = normData.lo[0], hi = normData.hi[0];
      datasets.push({
        label: 'Normative ROM',
        data: [{x: 0, y: hi}, {x: duration, y: hi}],
        borderColor: 'rgba(63, 185, 80, 0.25)',
        backgroundColor: 'rgba(63, 185, 80, 0.10)',
        borderWidth: 0.5, pointRadius: 0,
        fill: '+1', order: 10,
      });
      datasets.push({
        label: '_normLo',
        data: [{x: 0, y: lo}, {x: duration, y: lo}],
        borderColor: 'rgba(63, 185, 80, 0.25)',
        backgroundColor: 'transparent',
        borderWidth: 0.5, pointRadius: 0,
        fill: false, order: 11,
      });
    }

    const rData = ad.joints[rJoint]?.[rCol];
    const lData = lJoint ? ad.joints[lJoint]?.[lCol] : null;

    if (rData) {
      datasets.push({
        label: lData ? 'Right' : title,
        data: timeArr.map((t, i) => ({ x: t, y: rData[i] })),
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88, 166, 255, 0.05)',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: 0.2,
        order: 1,
      });
    }
    if (lData) {
      datasets.push({
        label: 'Left',
        data: timeArr.map((t, i) => ({ x: t, y: lData[i] })),
        borderColor: '#f0883e',
        backgroundColor: 'rgba(240, 136, 62, 0.05)',
        borderWidth: 1.5,
        borderDash: [4, 2],
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: 0.2,
        order: 2,
      });
    }

    if (!datasets.length) continue;

    const chart = new Chart(canvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        onHover: (_evt, _els, chart) => {
          // Broadcast hover x-value to all sibling charts
          const xScale = chart.scales.x;
          if (!xScale || !chart._lastEvent) return;
          const rect = chart.canvas.getBoundingClientRect();
          const mouseX = chart._lastEvent.x;
          if (mouseX < chart.chartArea.left || mouseX > chart.chartArea.right) {
            broadcastCrosshair(name, null);
            return;
          }
          const xVal = xScale.getValueForPixel(mouseX);
          broadcastCrosshair(name, xVal);
        },
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: duration,
            grid: { color: darkGrid, lineWidth: 0.5 },
            ticks: { color: darkTick, font: { size: 9 }, maxTicksLimit: 6 },
            border: { color: darkGrid },
          },
          y: {
            grid: {
              color: (ctx) => ctx.tick.value === 0 ? 'rgba(139, 148, 158, 0.5)' : darkGrid,
              lineWidth: (ctx) => ctx.tick.value === 0 ? 1 : 0.5,
            },
            ticks: {
              color: darkTick,
              font: { size: 9 },
              maxTicksLimit: 6,
              callback: (v) => v + '\u00b0',
            },
            border: { color: darkGrid },
          },
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            align: 'end',
            labels: {
              color: darkTick,
              font: { size: 8 },
              boxWidth: 14, boxHeight: 2, padding: 4,
              usePointStyle: false,
              filter: (item) => item.text && !item.text.startsWith('_'),
            },
          },
          tooltip: {
            backgroundColor: 'rgba(13, 17, 23, 0.95)',
            titleColor: '#e6edf3',
            bodyColor: '#8b949e',
            borderColor: '#30363d',
            borderWidth: 1,
            titleFont: { size: 10 },
            bodyFont: { size: 10 },
            callbacks: {
              title: (items) => `t = ${items[0].parsed.x.toFixed(2)}s`,
              label: (item) => {
                if (item.dataset.label.startsWith('_') || item.dataset.label === 'Normative') return null;
                return `${item.dataset.label}: ${item.parsed.y.toFixed(1)}\u00b0`;
              },
            },
          },
          anglePlayhead: { time: 0, hoverX: null },
          gaitPhase: { enabled: false },
          zoom: {
            pan: { enabled: true, mode: 'x' },
            zoom: {
              wheel: { enabled: true, modifierKey: 'ctrl' },
              drag: { enabled: false },
              mode: 'x',
            },
            limits: { x: { min: 0, max: duration } },
          },
        },
      },
    });

    chart._panelId = panelId;
    chart._normKey = normKey;
    chart._rJoint = rJoint;
    chart._rCol = rCol;
    chart._lJoint = lJoint;
    chart._lCol = lCol;
    chart._videoName = name;

    // Clear crosshair on mouse leave
    canvas.addEventListener('mouseleave', () => broadcastCrosshair(name, null));

    // Show Reset Zoom button when user zooms
    canvas.addEventListener('wheel', () => {
      setTimeout(() => {
        const btn = document.getElementById(`reset-zoom-${name}`);
        if (btn) btn.classList.add('visible');
      }, 100);
    });

    charts.push(chart);
  }

  return charts;
}

// -- Broadcast crosshair hover across all charts --------------------------
function broadcastCrosshair(name, xVal) {
  const state = tabStates[name];
  if (!state?.angleCharts) return;
  for (const chart of state.angleCharts) {
    chart.options.plugins.anglePlayhead.hoverX = xVal;
    chart.update('none');
  }
}

// -- Reset zoom on all charts for a video ---------------------------------
function resetAngleZoom(name) {
  const state = tabStates[name];
  if (!state?.angleCharts) return;
  for (const chart of state.angleCharts) {
    chart.resetZoom();
  }
  const btn = document.getElementById(`reset-zoom-${name}`);
  if (btn) btn.classList.remove('visible');
}

// -- Update playhead on all angle charts ----------------------------------
function updateAnglePlayhead(charts, time) {
  for (const chart of charts) {
    chart.options.plugins.anglePlayhead.time = time;
    chart.update('none');
  }
}

// -- Toggle gait cycle view -----------------------------------------------
const gaitCycleStates = {};

function toggleGaitCycle(name) {
  const state = tabStates[name];
  if (!state || !state.angleCharts?.length) return;

  const btn = document.getElementById(`gait-btn-${name}`);
  const isGait = !gaitCycleStates[name];
  gaitCycleStates[name] = isGait;

  if (btn) btn.classList.toggle('active', isGait);

  const vd = state.vd;
  const ad = vd.angle_data;
  const timeArr = ad.time_s;
  const duration = timeArr[timeArr.length - 1];

  // Detect gait cycles if not cached
  if (isGait && !state.gaitCycles) {
    state.gaitCycles = detectGaitCycles(ad);
  }
  const gc = state.gaitCycles;

  for (const chart of state.angleCharts) {
    if (isGait && gc) {
      // Switch to gait cycle view
      const pct = Array.from({length: 101}, (_, i) => i);
      const newDatasets = [];

      // Normative band (green fill, gait-cycle aligned) — activity-aware
      const actNorms = NORMATIVE?.[vd.activity] || {};
      const normData = chart._normKey ? actNorms[chart._normKey] : null;
      if (normData) {
        // Ipsilateral band (right leg reference)
        newDatasets.push({
          label: 'Normative',
          data: normData.pct.map((p, i) => ({ x: p, y: normData.hi[i] })),
          borderColor: 'rgba(63, 185, 80, 0.25)',
          backgroundColor: 'rgba(63, 185, 80, 0.10)',
          borderWidth: 0.5,
          pointRadius: 0,
          fill: '+1',
          order: 10,
        });
        newDatasets.push({
          label: '_normLo',
          data: normData.pct.map((p, i) => ({ x: p, y: normData.lo[i] })),
          borderColor: 'rgba(63, 185, 80, 0.25)',
          backgroundColor: 'transparent',
          borderWidth: 0.5,
          pointRadius: 0,
          fill: false,
          order: 11,
        });

        // Contralateral band (shifted 50% for left leg) — only if bilateral
        if (chart._lJoint) {
          const nPts = normData.pct.length;
          const half = Math.floor(nPts / 2);
          const shiftedHi = normData.pct.map((p, i) => {
            const si = (i + half) % nPts;
            return { x: p, y: normData.hi[si] };
          });
          const shiftedLo = normData.pct.map((p, i) => {
            const si = (i + half) % nPts;
            return { x: p, y: normData.lo[si] };
          });
          newDatasets.push({
            label: '_normContraHi',
            data: shiftedHi,
            borderColor: 'rgba(63, 185, 80, 0.15)',
            backgroundColor: 'rgba(63, 185, 80, 0.06)',
            borderWidth: 0.5,
            pointRadius: 0,
            fill: '+1',
            order: 12,
          });
          newDatasets.push({
            label: '_normContraLo',
            data: shiftedLo,
            borderColor: 'rgba(63, 185, 80, 0.15)',
            backgroundColor: 'transparent',
            borderWidth: 0.5,
            pointRadius: 0,
            fill: false,
            order: 13,
          });
        }
      }

      // Right: mean ± SD band
      const rKey = `${chart._rJoint}|${chart._rCol}`;
      const rCyc = gc.cycles[rKey];
      if (rCyc) {
        // SD band (fill between mean+sd and mean-sd)
        if (rCyc.nStrides > 1) {
          newDatasets.push({
            label: '_rSdHi',
            data: pct.map((p, i) => ({ x: p, y: rCyc.mean[i] + rCyc.sd[i] })),
            borderColor: 'transparent',
            backgroundColor: 'rgba(88, 166, 255, 0.12)',
            borderWidth: 0,
            pointRadius: 0,
            fill: '+1',
            order: 5,
          });
          newDatasets.push({
            label: '_rSdLo',
            data: pct.map((p, i) => ({ x: p, y: rCyc.mean[i] - rCyc.sd[i] })),
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            borderWidth: 0,
            pointRadius: 0,
            fill: false,
            order: 6,
          });
        }
        // Mean (bold)
        const rLabel = chart._lJoint
          ? `Right (N=${rCyc.nStrides})`
          : `Mean (N=${rCyc.nStrides})`;
        newDatasets.push({
          label: rLabel,
          data: pct.map((p, i) => ({ x: p, y: rCyc.mean[i] })),
          borderColor: '#58a6ff',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
          order: 1,
        });
      }

      // Left: mean ± SD band
      if (chart._lJoint) {
        const lKey = `${chart._lJoint}|${chart._lCol}`;
        const lCyc = gc.cycles[lKey];
        if (lCyc) {
          if (lCyc.nStrides > 1) {
            newDatasets.push({
              label: '_lSdHi',
              data: pct.map((p, i) => ({ x: p, y: lCyc.mean[i] + lCyc.sd[i] })),
              borderColor: 'transparent',
              backgroundColor: 'rgba(240, 136, 62, 0.12)',
              borderWidth: 0,
              pointRadius: 0,
              fill: '+1',
              order: 7,
            });
            newDatasets.push({
              label: '_lSdLo',
              data: pct.map((p, i) => ({ x: p, y: lCyc.mean[i] - lCyc.sd[i] })),
              borderColor: 'transparent',
              backgroundColor: 'transparent',
              borderWidth: 0,
              pointRadius: 0,
              fill: false,
              order: 8,
            });
          }
          newDatasets.push({
            label: `Left (N=${lCyc.nStrides})`,
            data: pct.map((p, i) => ({ x: p, y: lCyc.mean[i] })),
            borderColor: '#f0883e',
            borderWidth: 2,
            borderDash: [4, 2],
            pointRadius: 0,
            tension: 0.2,
            order: 2,
          });
        }
      }

      chart.data.datasets = newDatasets;
      chart.options.scales.x.min = 0;
      chart.options.scales.x.max = 100;
      chart.options.scales.x.ticks.callback = (v) => v + '%';
      chart.options.plugins.tooltip.callbacks.title = (items) => `${items[0].parsed.x.toFixed(0)}% gait cycle`;
      chart.options.plugins.gaitPhase.enabled = true;
      chart.options.plugins.zoom.limits = { x: { min: 0, max: 100 } };
    } else {
      // Switch back to time view (no normative bands — they are gait-cycle-normalized)
      const datasets = [];
      const rData = ad.joints[chart._rJoint]?.[chart._rCol];
      const lData = chart._lJoint ? ad.joints[chart._lJoint]?.[chart._lCol] : null;
      if (rData) {
        datasets.push({
          label: lData ? 'Right' : chart._panelId,
          data: timeArr.map((t, i) => ({ x: t, y: rData[i] })),
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.05)',
          borderWidth: 1.5,
          pointRadius: 0, pointHoverRadius: 3,
          tension: 0.2, order: 1,
        });
      }
      if (lData) {
        datasets.push({
          label: 'Left',
          data: timeArr.map((t, i) => ({ x: t, y: lData[i] })),
          borderColor: '#f0883e',
          backgroundColor: 'rgba(240, 136, 62, 0.05)',
          borderWidth: 1.5,
          borderDash: [4, 2],
          pointRadius: 0, pointHoverRadius: 3,
          tension: 0.2, order: 2,
        });
      }
      chart.data.datasets = datasets;
      chart.options.scales.x.min = 0;
      chart.options.scales.x.max = duration;
      chart.options.scales.x.ticks.callback = undefined;
      chart.options.plugins.tooltip.callbacks.title = (items) => `t = ${items[0].parsed.x.toFixed(2)}s`;
      chart.options.plugins.gaitPhase.enabled = false;
      chart.options.plugins.zoom.limits = { x: { min: 0, max: duration } };
    }
    chart.update();
  }
}

// Expose to global scope for inline onclick handlers (they run outside the IIFE)
window.toggleGaitCycle = toggleGaitCycle;
window.resetAngleZoom = resetAngleZoom;

// -- Three.js renderer (single, reused across tabs) -----------------------
let renderer = null;
let currentRenderer = null; // canvas element the renderer is attached to

function getRenderer(canvas) {
  if (!renderer) {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentRenderer = canvas;
  } else if (currentRenderer !== canvas) {
    // Re-attach renderer to new canvas by replacing its dom element reference
    // Three.js doesn't support changing canvas after creation, so we recreate.
    renderer.dispose();
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentRenderer = canvas;
  }
  return renderer;
}

// -- Per-tab state --------------------------------------------------------
const tabStates = {};  // name -> { scene, camera, bodyGroups, boneMeshes, transforms, ... }
let activeTab = null;
let animFrameId = null;

// -- Build tab DOM --------------------------------------------------------
const tabNav  = document.getElementById('tabs');
const panesEl = document.getElementById('panes');

VIDEOS.forEach((vd, vi) => {
  // Tab button
  const btn = document.createElement('button');
  btn.className = 'tab-btn' + (vi === 0 ? ' active' : '');
  btn.textContent = vd.label;
  btn.dataset.name = vd.name;
  btn.addEventListener('click', () => switchTab(vd.name));
  tabNav.appendChild(btn);

  // Pane
  const pane = document.createElement('div');
  pane.className = 'tab-pane' + (vi === 0 ? ' active' : '');
  pane.id = `pane-${vd.name}`;
  pane.innerHTML = `
    <div class="top-row">
      <!-- Video card -->
      <div class="card video-card">
        <div class="card-header">
          <span class="dot" style="background:#3fb950"></span>
          Input Video &mdash; ${vd.label}
        </div>
        <video id="vid-${vd.name}" muted loop playsinline preload="none"
               data-src="${vd.video_path}">
          Your browser does not support video.
        </video>
      </div>
      <!-- Skeleton card -->
      <div class="card skeleton-card">
        <div class="card-header">
          <span class="dot" style="background:#58a6ff"></span>
          OpenSim IK Skeleton &mdash; ${vd.n_frames} frames @ ${vd.fps.toFixed(1)} fps
          <span style="margin-left:auto;font-weight:400;color:#484f58">3D viewer</span>
        </div>
        <div id="skel-wrap-${vd.name}" style="position:relative;">
          <canvas class="skeleton-canvas" id="skel-${vd.name}"></canvas>
          <div class="loading-overlay" id="skel-load-${vd.name}">Initializing&hellip;</div>
          <div class="skel-controls-hint">Drag to orbit &middot; Scroll to zoom &middot; Right-drag to pan &middot; Double-click to auto-orbit</div>
          <div class="legend">
            <div class="legend-item"><span class="legend-dot" style="background:#4CAF50"></span>Pelvis / Spine</div>
            <div class="legend-item"><span class="legend-dot" style="background:#2196F3"></span>Left Leg</div>
            <div class="legend-item"><span class="legend-dot" style="background:#F44336"></span>Right Leg</div>
            <div class="legend-item"><span class="legend-dot" style="background:#00BCD4"></span>Left Arm</div>
            <div class="legend-item"><span class="legend-dot" style="background:#9C27B0"></span>Right Arm</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Unified timeline bar -->
    <div class="timeline-bar">
      <button class="play-btn" id="play-${vd.name}" title="Play / Pause (Space)">${ICON_PLAY}</button>
      <input type="range" class="timeline-slider" id="slider-${vd.name}"
             min="0" max="${vd.duration}" step="0.033" value="0">
      <span class="time-display" id="time-${vd.name}">0:00 / ${fmtTime(vd.duration)}</span>
    </div>

    <!-- Interactive clinical angle charts -->
    ${buildAngleChartsHTML(vd)}
  `;
  panesEl.appendChild(pane);
});

// -- Three.js scene builder -----------------------------------------------
function buildScene(vd, canvas) {
  const W = canvas.clientWidth  || 600;
  const H = canvas.clientHeight || 340;

  const r = getRenderer(canvas);
  r.setSize(W, H);
  r.setClearColor(0x0d1117, 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(2, 4, 3);
  scene.add(dirLight);
  const fillLight = new THREE.DirectionalLight(0x8899ff, 0.3);
  fillLight.position.set(-3, 1, -2);
  scene.add(fillLight);

  // Ground grid
  const grid = new THREE.GridHelper(6, 24, 0x1c2128, 0x161b22);
  scene.add(grid);

  // Camera
  const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 100);
  camera.position.set(2.5, 1.2, 0);

  // OrbitControls
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.9, 0);
  controls.enableDamping = true;
  controls.enableZoom = false;  // disable built-in (too aggressive), use manual below
  controls.minDistance = 1.0;
  controls.maxDistance = 8.0;
  controls.autoRotate = true;
  // ~1 revolution per 2× clip duration (gentle orbit)
  controls.autoRotateSpeed = 30 / (vd.duration || 10);
  controls.addEventListener('start', () => { controls.autoRotate = false; });
  // Double-click to re-enable auto-orbit
  canvas.addEventListener('dblclick', () => { controls.autoRotate = true; });
  // Manual scroll zoom with reduced sensitivity
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
    const dist = camera.position.distanceTo(controls.target);
    const delta = e.deltaY * 0.002 * dist;  // proportional to distance
    const newDist = Math.max(1.0, Math.min(8.0, dist + delta));
    camera.position.copy(controls.target).addScaledVector(dir, newDist);
  }, { passive: false });
  controls.update();

  // Create body mesh groups
  const bodyGroups = [];
  for (const bodyDef of vd.bodies) {
    const group = new THREE.Group();
    for (const meshDef of bodyDef.meshes) {
      const meshData = MESHES[meshDef.name];
      if (!meshData) continue;

      const geometry = new THREE.BufferGeometry();
      const verts = new Float32Array(meshData.v);
      const indices = meshData.f.length > 65535
        ? new Uint32Array(meshData.f)
        : new Uint16Array(meshData.f);
      geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
      geometry.computeVertexNormals();

      const material = new THREE.MeshPhongMaterial({
        color: bodyDef.color,
        shininess: 40,
        transparent: true,
        opacity: 0.9,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.scale.set(meshDef.s[0], meshDef.s[1], meshDef.s[2]);
      group.add(mesh);
    }
    scene.add(group);
    bodyGroups.push(group);
  }

  // Edge bones (thin cylinders for skeleton wire overlay)
  const boneMeshes = [];
  for (let ei = 0; ei < vd.edges.length; ei++) {
    const geo = new THREE.CylinderGeometry(0.003, 0.003, 1, 6);
    geo.translate(0, 0.5, 0);
    geo.rotateX(Math.PI / 2);
    const mat = new THREE.MeshPhongMaterial({
      color: vd.edge_colors[ei],
      emissive: new THREE.Color(vd.edge_colors[ei]).multiplyScalar(0.08),
      shininess: 30,
      transparent: true,
      opacity: 0.5,
    });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    boneMeshes.push(mesh);
  }

  // Pre-parse transforms into Float32Arrays
  const NF = vd.n_frames;
  const NB = vd.n_bodies;
  const transforms = new Array(NF);
  for (let f = 0; f < NF; f++) {
    transforms[f] = new Float32Array(vd.transforms[f]);
  }

  // Compute average pelvis Y for camera target (body 0 = pelvis typically)
  let sumPelvisY = 0;
  for (let f = 0; f < NF; f++) {
    sumPelvisY += transforms[f][1];  // Y of first body
  }
  const avgPelvisY = sumPelvisY / NF;

  // Update controls target with actual pelvis height
  controls.target.set(0, avgPelvisY * 0.85, 0);
  controls.update();

  return { scene, camera, controls, bodyGroups, boneMeshes, transforms, avgPelvisY, NF, NB };
}

function setFrame(state, fi) {
  const t = state.transforms[fi];
  const NB = state.NB;
  const vd = state.vd;

  // Get pelvis position for XZ centering (first body)
  const px = t[0], pz = t[2];

  for (let bi = 0; bi < NB; bi++) {
    const off = bi * 7;
    const group = state.bodyGroups[bi];
    group.position.set(t[off] - px, t[off + 1], t[off + 2] - pz);
    // NPZ quaternion is [w,x,y,z], Three.js Quaternion constructor is (x,y,z,w)
    group.quaternion.set(t[off + 4], t[off + 5], t[off + 6], t[off + 3]);
  }

  // Update bone edges
  const _tmp = new THREE.Vector3();
  for (let ei = 0; ei < vd.edges.length; ei++) {
    const [pi, ci] = vd.edges[ei];
    const pOff = pi * 7, cOff = ci * 7;
    const bx = t[pOff] - px, by = t[pOff + 1], bz = t[pOff + 2] - pz;
    const tx = t[cOff] - px, ty = t[cOff + 1], tz = t[cOff + 2] - pz;
    const dx = tx - bx, dy = ty - by, dz = tz - bz;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const bone = state.boneMeshes[ei];
    bone.position.set(bx, by, bz);
    bone.scale.set(1, 1, len);
    _tmp.set(tx, ty, tz);
    bone.lookAt(_tmp);
  }
}

// -- Tab switching --------------------------------------------------------
function switchTab(name) {
  if (activeTab === name) return;

  // Pause currently active video
  if (activeTab) {
    const prevVid = document.getElementById(`vid-${activeTab}`);
    if (prevVid) prevVid.pause();
  }

  // Update tab button styles
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.name === name);
  });
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `pane-${name}`);
  });

  activeTab = name;

  // Lazy-load video source
  const vid = document.getElementById(`vid-${name}`);
  if (vid && !vid.src && vid.dataset.src) {
    vid.src = vid.dataset.src;
  }

  // Initialize or re-attach Three.js scene
  if (!tabStates[name]) {
    initTabScene(name);
  } else {
    // Re-attach renderer to this tab's canvas
    const canvas = document.getElementById(`skel-${name}`);
    const state = tabStates[name];
    const r = getRenderer(canvas);
    const W = canvas.clientWidth  || 600;
    const H = canvas.clientHeight || 340;
    r.setSize(W, H);
    state.camera.aspect = W / H;
    state.camera.updateProjectionMatrix();
    // Show first frame
    setFrame(state, 0);
    r.render(state.scene, state.camera);
    const overlay = document.getElementById(`skel-load-${name}`);
    if (overlay) overlay.classList.add('hidden');
  }

  // Reset timeline to 0
  const slider = document.getElementById(`slider-${name}`);
  const timeEl = document.getElementById(`time-${name}`);
  const vd = VIDEOS.find(v => v.name === name);
  if (slider) slider.value = 0;
  if (timeEl && vd) timeEl.textContent = `0:00 / ${fmtTime(vd.duration)}`;

  // Start render loop
  if (animFrameId) cancelAnimationFrame(animFrameId);
  renderLoop();
}

function initTabScene(name) {
  const vd = VIDEOS.find(v => v.name === name);
  if (!vd) return;

  const canvas = document.getElementById(`skel-${name}`);
  if (!canvas) return;

  const state = buildScene(vd, canvas);
  state.vd = vd;
  tabStates[name] = state;

  // Wire up video + timeline controls for this tab
  const vid    = document.getElementById(`vid-${name}`);
  const slider = document.getElementById(`slider-${name}`);
  const playBtn = document.getElementById(`play-${name}`);
  const timeEl  = document.getElementById(`time-${name}`);
  // Initialize angle charts for this tab
  state.angleCharts = initAngleCharts(name, vd);

  let userSeeking = false;

  function updateFromTime(t) {
    const clamp = Math.max(0, Math.min(t, vd.duration));
    if (slider && !userSeeking) slider.value = clamp;
    if (timeEl) timeEl.textContent = `${fmtTime(clamp)} / ${fmtTime(vd.duration)}`;

    // Update angle chart playheads
    if (state.angleCharts?.length && !gaitCycleStates[name]) {
      updateAnglePlayhead(state.angleCharts, clamp);
    }

    // Skeleton frame
    const fi = Math.min(
      Math.floor(clamp * vd.fps),
      vd.n_frames - 1
    );
    const curState = tabStates[name];
    if (curState && fi >= 0) {
      setFrame(curState, fi);
      if (curState.controls) curState.controls.update();
    }
  }

  // Video timeupdate -> sync skeleton
  if (vid) {
    vid.addEventListener('timeupdate', () => {
      if (!vid.paused && !userSeeking) updateFromTime(vid.currentTime);
    });
    vid.addEventListener('seeked',  () => updateFromTime(vid.currentTime));
    // Loop reset
    vid.addEventListener('ended', () => {
      if (playBtn) { playBtn.innerHTML = ICON_PLAY; }
    });
  }

  // Slider -> seek video + sync skeleton
  if (slider) {
    slider.addEventListener('mousedown',  () => { userSeeking = true; });
    slider.addEventListener('touchstart', () => { userSeeking = true; });
    slider.addEventListener('input', () => {
      const t = parseFloat(slider.value);
      if (vid) vid.currentTime = t;
      updateFromTime(t);
    });
    const endSeek = () => { userSeeking = false; };
    slider.addEventListener('mouseup',  endSeek);
    slider.addEventListener('touchend', endSeek);
    slider.addEventListener('change',   endSeek);
  }

  // Play button
  if (playBtn) {
    playBtn.innerHTML = ICON_PLAY;
    playBtn.addEventListener('click', () => {
      if (!vid) return;
      if (vid.paused) {
        // Lazy-load
        if (!vid.src && vid.dataset.src) vid.src = vid.dataset.src;
        vid.play().catch(() => {});
        playBtn.innerHTML = ICON_PAUSE;
      } else {
        vid.pause();
        playBtn.innerHTML = ICON_PLAY;
      }
    });
    vid && vid.addEventListener('play',  () => { playBtn.innerHTML = ICON_PAUSE; });
    vid && vid.addEventListener('pause', () => { playBtn.innerHTML = ICON_PLAY; });
  }

  // Keyboard shortcuts (only active for active tab)
  document.addEventListener('keydown', (e) => {
    if (activeTab !== name) return;
    if (e.code === 'Space') {
      e.preventDefault();
      if (playBtn) playBtn.click();
    } else if (e.code === 'ArrowRight' && vid) {
      vid.currentTime = Math.min(vid.currentTime + 1/30, vd.duration);
    } else if (e.code === 'ArrowLeft' && vid) {
      vid.currentTime = Math.max(vid.currentTime - 1/30, 0);
    }
  });

  // Remove loading overlay
  setTimeout(() => {
    setFrame(state, 0);
    const r = getRenderer(canvas);
    r.render(state.scene, state.camera);
    const overlay = document.getElementById(`skel-load-${name}`);
    if (overlay) overlay.classList.add('hidden');
  }, 50);
}

// -- Main render loop -----------------------------------------------------
let lastRenderFrame = -1;

function renderLoop() {
  animFrameId = requestAnimationFrame(renderLoop);
  if (!activeTab) return;

  const state = tabStates[activeTab];
  if (!state) return;

  const canvas = document.getElementById(`skel-${activeTab}`);
  if (!canvas) return;

  const r = getRenderer(canvas);
  const vid = document.getElementById(`vid-${activeTab}`);

  // Sync skeleton from video time each rAF when playing
  if (vid && !vid.paused) {
    const vd = state.vd;
    const t  = vid.currentTime;
    const fi = Math.min(Math.floor(t * vd.fps), vd.n_frames - 1);

    if (fi !== lastRenderFrame) {
      setFrame(state, fi);
      lastRenderFrame = fi;
    }
  }

  // Update OrbitControls (handles auto-rotate + damping)
  if (state.controls) state.controls.update();

  r.render(state.scene, state.camera);
}

// -- Window resize --------------------------------------------------------
window.addEventListener('resize', () => {
  if (!activeTab) return;
  const canvas = document.getElementById(`skel-${activeTab}`);
  if (!canvas) return;
  const state = tabStates[activeTab];
  const r = getRenderer(canvas);
  const W = canvas.clientWidth  || 600;
  const H = canvas.clientHeight || 340;
  r.setSize(W, H);
  if (state) {
    state.camera.aspect = W / H;
    state.camera.updateProjectionMatrix();
    r.render(state.scene, state.camera);
  }
});

// -- Hide loader and bootstrap --------------------------------------------
loaderEl.style.display = 'none';

if (VIDEOS.length > 0) {
  switchTab(VIDEOS[0].name);
} else {
  panesEl.innerHTML = `
    <div style="padding:48px;text-align:center;color:var(--text-muted);">
      <p style="font-size:1.1rem;">No processed videos found.</p>
      <p style="margin-top:8px;font-size:0.85rem;">
        Run the pipeline on a video first:<br>
        <code style="color:var(--accent)">uv run python main.py --video data/input/joey.mp4 --height 1.78 --plot-joint-angles</code>
      </p>
    </div>`;
}

})(); // end async main
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML demo page for the OpenSim FK pipeline."
    )
    parser.add_argument(
        "--output-dir",
        default="data/demo",
        help="Directory to write demo.html and copied videos (default: data/demo)",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        metavar="NAME",
        help="Video base names to include (default: auto-discover all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    # Discover videos
    names = _discover_videos(args.videos)
    if not names:
        print("No processed videos found. Run the pipeline first.", file=sys.stderr)
        print("Required files per video:", file=sys.stderr)
        print("  data/output/<name>/<name>_fk_bodies.npz", file=sys.stderr)
        print("  data/output/<name>/joint_angles/ (CSV files)", file=sys.stderr)
        print("  data/input/<name>.mp4 (or .avi/.mov)", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(names)} video(s): {', '.join(names)}")

    # Build per-video data and collect all needed mesh names
    videos_data: list[dict] = []
    video_copies: list[tuple[Path, Path]] = []  # (src, dst)
    all_mesh_names: set[str] = set()

    for name in names:
        entry, mesh_names, video_dest, input_video = _build_video_entry(
            name, output_dir
        )
        all_mesh_names |= mesh_names

        entry_size_kb = sum(len(str(v)) for v in entry.values()) // 1024
        print(f"    body data: ~{entry_size_kb} KB")

        videos_data.append(entry)
        video_copies.append((input_video.resolve(), video_dest))

    # Load all needed VTP meshes
    print(f"\nLoading {len(all_mesh_names)} mesh file(s)...")
    meshes_data = _load_meshes(all_mesh_names)
    mesh_total_verts = sum(len(m["v"]) // 3 for m in meshes_data.values())
    mesh_total_tris = sum(len(m["f"]) // 3 for m in meshes_data.values())
    print(f"  Total: {mesh_total_verts} vertices, {mesh_total_tris} triangles")

    # Copy/re-encode videos (CFR re-encode ensures consistent browser playback)
    print("\nProcessing video files...")
    for src, dst in video_copies:
        if dst.exists():
            dst.unlink()
        _copy_or_reencode_video(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  {dst.name} ({size_mb:.1f} MB)")

    # Load normative reference data (walking, running, jump)
    print("\nLoading normative reference data...")
    normative_data = _load_normative_data()
    for activity, norms in normative_data.items():
        print(f"  {activity}: {len(norms)} DOFs")
    if not normative_data:
        print("  WARNING: no normative data available")

    # Write data as external JSON file (keeps HTML small and parseable)
    data_payload = {
        "meshes": meshes_data,
        "videos": videos_data,
        "normative": normative_data,
    }
    data_json = json.dumps(data_payload, separators=(",", ":"))
    data_file = output_dir / "data.json"
    data_file.write_text(data_json, encoding="utf-8")
    data_mb = data_file.stat().st_size / 1024 / 1024
    print(f"\nData file written: {data_file} ({data_mb:.1f} MB)")

    # Generate HTML (data loaded via fetch, not inline)
    html = _HTML_TEMPLATE.replace(
        "const MESHES = __MESHES_DATA__;\nconst VIDEOS = __VIDEOS_DATA__;\nconst NORMATIVE = __NORMATIVE_DATA__;",
        "let MESHES, VIDEOS, NORMATIVE;",
    )
    out_html = output_dir / "demo.html"
    out_html.write_text(html, encoding="utf-8")

    size_kb = out_html.stat().st_size / 1024
    print(f"Demo page written: {out_html} ({size_kb:.0f} KB)")
    print(f"\nTo view: python -m http.server 8001 --directory {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"  then open http://localhost:8001/demo.html")


if __name__ == "__main__":
    main()
