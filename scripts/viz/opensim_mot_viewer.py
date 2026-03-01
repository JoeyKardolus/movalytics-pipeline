#!/usr/bin/env python3
"""OpenSim motion viewer — plays .mot IK results on .osim model.

Runs in the opensim conda env (Python 3.12 + OpenSim 4.5.2).
Uses Simbody's native 3D visualizer with built-in playback controls.

Usage:
    conda run -n opensim python scripts/viz/opensim_mot_viewer.py \
        --mot data/output/joey/joey_sam3d_markers_ik.mot \
        --model data/output/joey/joey_sam3d_markers.osim
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import opensim as osim


def main():
    parser = argparse.ArgumentParser(description="OpenSim .mot viewer")
    parser.add_argument("--mot", required=True, help="Path to .mot motion file")
    parser.add_argument("--model", default=None, help="Path to .osim model (auto-discovered if omitted)")
    parser.add_argument("--target-fps", type=float, default=30.0,
                        help="Downsample to this fps for smooth playback (default 30)")
    args = parser.parse_args()

    mot_path = Path(args.mot).resolve()
    if not mot_path.exists():
        print(f"ERROR: .mot file not found: {mot_path}", file=sys.stderr)
        sys.exit(1)

    # Auto-discover .osim model if not specified
    if args.model:
        model_path = Path(args.model).resolve()
    else:
        mot_dir = mot_path.parent
        osim_files = list(mot_dir.glob("*.osim"))
        if not osim_files:
            print(f"ERROR: No .osim model found in {mot_dir}. Use --model.", file=sys.stderr)
            sys.exit(1)
        # Prefer model matching the .mot stem (e.g. joey_sam3d_markers.osim for joey_sam3d_markers_ik.mot)
        mot_stem = mot_path.stem.replace("_ik", "")
        model_path = None
        for f in osim_files:
            if f.stem == mot_stem:
                model_path = f
                break
        if model_path is None:
            model_path = osim_files[0]

    if not model_path.exists():
        print(f"ERROR: .osim model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Add geometry search paths
    # 1. Project's own Geometry dir (LaiUhlrich2022)
    project_root = Path(__file__).resolve().parent.parent.parent
    for geo_dir in [
        project_root / "models" / "opensim" / "Geometry",
        model_path.parent / "Geometry",
    ]:
        if geo_dir.exists():
            osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geo_dir))
    # 2. Pose2Sim fallback
    try:
        import Pose2Sim
        geometry_dir = Path(Pose2Sim.__file__).parent / "OpenSim_Setup" / "Geometry"
        if geometry_dir.exists():
            osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geometry_dir))
    except ImportError:
        pass

    print(f"[viewer] Model: {model_path.name}")
    print(f"[viewer] Motion: {mot_path.name}")

    # Strip prescribed coordinates from XML before loading — the viewer
    # drives all DOFs from .mot values; prescribed functions (from IK pelvis
    # fix) conflict with showMotion and can crash the Simbody visualizer.
    osim_xml = model_path.read_text()
    osim_xml = re.sub(r"<prescribed>true</prescribed>",
                      "<prescribed>false</prescribed>", osim_xml)
    tmp_osim = tempfile.NamedTemporaryFile(suffix=".osim", delete=False, dir=model_path.parent)
    tmp_osim.write(osim_xml.encode())
    tmp_osim.close()
    tmp_osim_path = Path(tmp_osim.name)

    try:
        model = osim.Model(str(tmp_osim_path))
    finally:
        tmp_osim_path.unlink(missing_ok=True)

    # Remove muscles entirely — no muscle analysis, and their GeometryPath
    # appearance flag doesn't reliably hide them in the Simbody visualizer.
    fs = model.getForceSet()
    muscle_names = []
    for i in range(fs.getSize()):
        if osim.Muscle.safeDownCast(fs.get(i)) is not None:
            muscle_names.append(fs.get(i).getName())
    for name in muscle_names:
        idx = fs.getIndex(name)
        if idx >= 0:
            fs.remove(idx)
    # Disable remaining forces (ligaments, etc.)
    for i in range(fs.getSize()):
        fs.get(i).set_appliesForce(False)

    # Hide hands — no hand detection, just visual noise
    body_set = model.updBodySet()
    for name in ("hand_r", "hand_l"):
        try:
            body = body_set.get(name)
            for j in range(body.getPropertyByName("attached_geometry").size()):
                body.upd_attached_geometry(j).upd_Appearance().set_visible(False)
        except Exception:
            pass

    model.initSystem()

    table = osim.TimeSeriesTable(str(mot_path))
    n_rows = table.getNumRows()
    times = table.getIndependentColumn()
    last_time = times[-1] if times else 0.0

    # Downsample high-fps MOTs for smooth Simbody playback.
    # The visualizer can't reliably render >30fps in real-time, causing
    # frame skipping that makes motion appear faster than real-time.
    if n_rows > 1 and last_time > 0:
        source_fps = (n_rows - 1) / last_time
        target_fps = args.target_fps
        if source_fps > target_fps * 1.5:
            step = max(1, round(source_fps / target_fps))
            keep_indices = list(range(0, n_rows, step))
            if keep_indices[-1] != n_rows - 1:
                keep_indices.append(n_rows - 1)  # always keep last frame
            labels = list(table.getColumnLabels())
            downsampled = osim.TimeSeriesTable()
            downsampled.setColumnLabels(labels)
            downsampled.addTableMetaDataString("inDegrees", "yes")
            for i in keep_indices:
                downsampled.appendRow(times[i], table.getRowAtIndex(i))
            new_n = len(keep_indices)
            print(f"[viewer] Downsampled {source_fps:.0f}fps → {new_n / last_time:.0f}fps "
                  f"({n_rows} → {new_n} frames) for smooth playback")
            table = downsampled
            n_rows = new_n

    print(f"[viewer] {n_rows} frames, {last_time:.1f}s")
    print(f"[viewer] Opening Simbody visualizer... (close window to exit)")

    try:
        osim.VisualizerUtilities.showMotion(model, table)
    except RuntimeError as e:
        if "Broken pipe" in str(e):
            pass  # normal — user closed visualizer window
        else:
            raise


if __name__ == "__main__":
    main()
