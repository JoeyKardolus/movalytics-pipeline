# Scripts

Overview of all scripts in the repository.

---

## Tools (`scripts/tools/`)

Utility scripts for model setup, atlas building, and data export.

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_models.py` | Download SAM 3D Body weights (~2.7 GB) | `uv run python scripts/tools/download_models.py` |
| `build_mhr_atlas.py` | Build MHR surface marker atlas from mesh | `uv run python scripts/tools/build_mhr_atlas.py` |
| `validate_marker_atlas.py` | Validate marker atlas consistency | `uv run python scripts/tools/validate_marker_atlas.py` |
| `auto_site_markers.py` | Auto-position MuJoCo site markers | `uv run python scripts/tools/auto_site_markers.py` |
| `export_mhr_mesh_json.py` | Export MHR mesh for WebGL viewers | `uv run python scripts/tools/export_mhr_mesh_json.py <npz>` |

---

## Visualization (`scripts/viz/`)

Viewers and visualization generators for pipeline outputs.

| Script | Input | Purpose | Prerequisites |
|--------|-------|---------|---------------|
| `demo_page.py` | `data/output/` | Generate shareable HTML demo with 3D skeleton + charts | Processed videos with `_fk_bodies.npz` |
| `opensim_mot_viewer.py` | `_ik.mot` | OpenSim MOT file viewer | `opensim` conda env |
| `sam3d_mesh_viewer.py` | `_sam3d.npz` | 3D mesh visualization | matplotlib |
| `trc_3d_viewer.py` | `.trc` | TRC marker skeleton viewer | matplotlib |
| `mhr_marker_picker/` | -- | Interactive HTML/JS marker annotation tool | Browser |

---

## Demo Page Workflow

The demo page generator auto-discovers all processed videos in `data/output/` and produces a self-contained HTML page with an interactive 3D skeleton viewer and joint angle charts.

```bash
# 1. Process videos
uv run python main.py --video data/input/walking.mp4 --height 1.75
uv run python main.py --video data/input/running.mp4 --height 1.80

# 2. Generate demo (auto-discovers all processed videos)
uv run python scripts/viz/demo_page.py

# 3. View
open data/demo/demo.html
```

Each processed video must have a `_fk_bodies.npz` file in its output directory for the 3D skeleton to render. If `_fk_bodies.npz` is missing, ensure `lifting.opensim.skip_fk` is set to `false` (the default) and re-run the pipeline.
