# Visualization Scripts

Standalone viewers for pipeline output files. Each script reads one or more output files and renders an interactive visualization.

## Viewers

| Script | Env | Input | Description |
|--------|-----|-------|-------------|
| `sam3d_viewer.py` | uv | `_sam3d.npz` | Matplotlib 3D skeleton animation (127 MHR joints) |
| `sam3d_clinical_viewer.py` | uv | `_sam3d.npz` | HTML viewer with clinical 30-DOF skeleton + live angle readouts |
| `sam3d_viewer_html.py` | uv | `_sam3d.npz` | Standalone HTML Three.js skeleton viewer (orbit, timeline, play/pause) |
| `sam3d_mesh_viewer.py` | sam3d | `_sam3d.npz` + video | Mesh overlay on video — renders MHR body mesh per frame |
| `opensim_mot_viewer.py` | opensim | `_ik.mot` + `.osim` | Plays OpenSim IK results on the scaled model (Simbody visualizer) |
| `trc_3d_viewer.py` | uv | `.trc` | Matplotlib animated 3D marker skeleton from TRC data |
| `demo_page.py` | uv | `data/output/` | Generates shareable HTML demo page with bone mesh animation + video sync |

## Quick Start

```bash
# SAM 3D skeleton (matplotlib, quick check)
uv run python scripts/viz/sam3d_viewer.py data/output/joey/joey_sam3d.npz

# SAM 3D skeleton (HTML, shareable)
uv run python scripts/viz/sam3d_viewer_html.py data/output/joey/joey_sam3d.npz

# Clinical angles (HTML with live readouts)
uv run python scripts/viz/sam3d_clinical_viewer.py data/output/joey/joey_sam3d.npz

# Mesh overlay on video (requires sam3d conda env)
conda run -n sam3d python scripts/viz/sam3d_mesh_viewer.py \
    --npz data/output/joey/joey_sam3d.npz --video data/input/joey.mp4

# OpenSim IK playback (requires opensim conda env)
conda run -n opensim python scripts/viz/opensim_mot_viewer.py \
    --mot data/output/joey/joey_mhr_markers_ik.mot

# TRC marker skeleton
uv run python scripts/viz/trc_3d_viewer.py data/output/joey/joey_mhr_markers.trc

# Demo page (auto-discovers all processed videos)
uv run python scripts/viz/demo_page.py
```

## Subdirectories

- `mhr_marker_picker/` — HTML/JS interactive app for manually siting anatomical markers on the MHR mesh. Used during atlas development; not part of the pipeline.
