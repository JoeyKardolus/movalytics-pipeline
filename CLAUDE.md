# CLAUDE.md

## Rules
- **NEVER revert code or make major design/architectural changes without asking the user first.** Fine-tune and fix issues incrementally.
- **Clinical joint angles come from the OpenSim .mot file** (via `mot_to_clinical.py`). Do NOT switch to SAM3D angles or any other source without explicit user approval.
- **Skeleton visualization comes from NimblePhysics FK** (via `opensim_ik_worker.py`). The FK bodies NPZ powers the demo page skeleton.
- When sign conventions are wrong, fix the sign mapping — don't replace the upstream data source.

## Project
3D human pose estimation pipeline: monocular video → YOLOX detection → SAM 3D Body (MHR) → TRC markers → OpenSim IK → clinical joint angles + skeleton visualization.

## Environments
- Main: `uv run python main.py --video <path>`
- SAM 3D: `conda run -n sam3d` (Python 3.11)
- OpenSim IK + NimblePhysics FK: `conda run -n opensim`
- NimblePhysics requires numpy<2 (segfaults with numpy 2.x)

## Key Paths
- Entry: `main.py`
- Config: `src/core/config.py`
- OpenSim IK worker: `src/workers/opensim_ik_worker.py`
- TRC generation: `src/core/conversion/mhr_markers_to_trc.py`
- .mot → clinical angles: `src/core/kinematics/mot_to_clinical.py`
- SAM3D clinical angles: `src/core/conversion/sam3d_clinical_angles.py`
- Angle plot: `src/core/conversion/sam3d_visualization.py`
- Demo page: `scripts/viz/demo_page.py`
