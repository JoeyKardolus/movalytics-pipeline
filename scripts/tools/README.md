# Build & Setup Tools

Utilities for building marker atlases, exporting mesh data, and validating pipeline artifacts. These are run manually during setup or after model changes — not called by the pipeline at runtime.

## Tools

| Script | Description |
|--------|-------------|
| `build_mhr_atlas.py` | Build MHR surface marker atlas from rest-pose mesh (auto-siting or manual JSON input) |
| `auto_site_markers.py` | Auto-detect 41 anatomical landmarks on MHR mesh geometry via surface extrema |
| `validate_marker_atlas.py` | Check marker placement against biomechanical norms and LaiUhlrich2022 reference |
| `export_mhr_mesh_json.py` | Export MHR rest-pose mesh as JSON for the interactive marker picker |
| `download_models.py` | Check availability of pretrained model weights |

## Typical Workflow

After processing a new video (to get the MHR rest-pose mesh):

```bash
# 1. Build the marker atlas (auto-siting from mesh geometry)
uv run python scripts/tools/build_mhr_atlas.py --auto data/output/joey/joey_sam3d.npz

# 2. Validate placement against anatomical norms
uv run python scripts/tools/validate_marker_atlas.py data/output/joey/

# 3. Export mesh for manual refinement (if needed)
uv run python scripts/tools/export_mhr_mesh_json.py data/output/joey/
# Then open scripts/viz/mhr_marker_picker/ to refine interactively

# 4. Check model weights are available
uv run python scripts/tools/download_models.py --check
```
