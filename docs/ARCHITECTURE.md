# Architecture

System architecture for the movalytics-pipeline: video-to-biomechanics via 3D pose estimation.

---

## Module Map

All core modules live under `src/core/`.

| Module | Purpose |
|--------|---------|
| `detection/` | YOLOX bounding box detection, IOU tracking + OneEuro smoothing |
| `lifting/` | SAM 3D Body lifter, OpenSim IK + FK export |
| `conversion/` | TRC I/O, SAM 3D clinical angles, MHR marker atlas, marker maps |
| `kinematics/` | Joint angle export and visualization |
| `video/` | Video I/O via OpenCV |
| `pipeline/` | Output cleanup and organization |
| `config.py` | Centralized config, YAML load/dump, CLI override |

---

## Pipeline Flow

```
 +-------+     +------------+     +-----------------+     +---------------+
 | INPUT | --> | LOAD VIDEO | --> | YOLOX DETECTION | --> | SAM 3D BODY   |
 +-------+     +------------+     +-----------------+     +---------------+
                                                                 |
                                                                 v
                                                        +-----------------+
                                                        | CLINICAL ANGLES |
                                                        +-----------------+
                                                                 |
                                          +----------------------+----------------------+
                                          |                      |                      |
                                          v                      v                      v
                                 +-----------------+    +--------------+    +------------+
                                 | SURFACE MARKERS |    | OPENSIM IK   |    | FK EXPORT  |
                                 +-----------------+    +--------------+    +------------+
                                          |                      |                      |
                                          +----------------------+----------------------+
                                                                 |
                                                                 v
                                                           +----------+
                                                           |  EXPORT  |
                                                           +----------+
                                                                 |
                                                                 v
                                                           +----------+
                                                           |  OUTPUT  |
                                                           +----------+
```

---

## Output Structure

```
data/output/<video>/
+-- <video>_sam3d.npz             # Raw MHR body model (127 joints, rotations, shape, mesh)
+-- <video>_mhr_markers.trc       # 65 anatomical surface markers
+-- <video>.osim                  # Scaled OpenSim model
+-- <video>_ik.mot                # OpenSim IK joint angles
+-- <video>_fk_bodies.npz         # FK body positions/orientations for demo viewer
+-- <video>_sam3d_angles.png      # Clinical angle plots (if --plot-joint-angles)
+-- joint_angles/                 # CSV per joint group (30 DOFs)
```

---

## Entry Points

| File | Purpose |
|------|---------|
| `main.py` | CLI pipeline orchestrator |
| `scripts/viz/demo_page.py` | Demo page generator |
| `src/workers/sam3d_worker.py` | SAM 3D Body subprocess (conda sam3d) |
| `src/workers/opensim_ik_worker.py` | OpenSim IK + FK subprocess (conda opensim) |

---

## Tools

| Script | Purpose |
|--------|---------|
| `scripts/tools/download_models.py` | Download SAM 3D Body weights |
| `scripts/tools/build_mhr_atlas.py` | Build MHR surface marker atlas from mesh |
| `scripts/tools/validate_marker_atlas.py` | Validate marker atlas consistency |
| `scripts/tools/auto_site_markers.py` | Auto-position MuJoCo site markers |
| `scripts/tools/export_mhr_mesh_json.py` | Export MHR mesh for WebGL viewers |

## Viewers

| Script | Input | Purpose |
|--------|-------|---------|
| `scripts/viz/demo_page.py` | `data/output/` | Generate shareable HTML demo with 3D skeleton + charts |
| `scripts/viz/opensim_mot_viewer.py` | `_ik.mot` | OpenSim MOT file viewer |
| `scripts/viz/sam3d_mesh_viewer.py` | `_sam3d.npz` | 3D mesh visualization |
| `scripts/viz/trc_3d_viewer.py` | `.trc` | TRC marker skeleton viewer |
| `scripts/viz/mhr_marker_picker/` | -- | Interactive HTML/JS marker annotation tool |

---

## Module Communication

| From | To | Mechanism |
|------|----|-----------|
| `main.py` | `detection/` | Direct import, function call |
| `main.py` | `lifting/` | Direct import, function call |
| `lifting/` | `sam3d_worker.py` | Subprocess via `conda run -n sam3d` |
| `lifting/` | `opensim_ik_worker.py` | Subprocess via `conda run -n opensim` |
| `lifting/` | `conversion/` | Direct import for TRC/angle output |
| `main.py` | `kinematics/` | Direct import for angle export |
| `main.py` | `pipeline/` | Direct import for output cleanup |

---

## Data Flow

```
Video frames (OpenCV)
    |
    v
Bounding boxes (YOLOX detection + IOU tracking + OneEuro smoothing)
    |
    v
MHR body model per frame (SAM 3D Body subprocess)
    |-- 127 MHR joints
    |-- Global rotations
    |-- Shape/scale parameters
    |-- Mesh vertices
    |
    +---> Clinical joint angles (30 DOFs from MHR rotation matrices)
    |         |
    |         v
    |     Joint angle CSVs + optional PNGs
    |
    +---> Surface markers (65 anatomical markers from MHR mesh via atlas)
    |         |
    |         v
    |     TRC file for OpenSim
    |         |
    |         v
    |     OpenSim IK (.mot) + FK bodies (.npz)
    |
    +---> FK body export (body positions/orientations for demo viewer)
```

---

## Coordinate Systems

Three coordinate frames are used across the pipeline. Transformations between them are handled in `src/core/conversion/` and `src/shared/`.

### MHR body-centric

The native coordinate system of the MHR body model used by SAM 3D Body.

- **X** = right
- **Y** = up
- **Z** = backward
- Right-handed

### Camera (SAM 3D output)

The camera-space coordinate system in which SAM 3D Body produces its output.

- **X** = right
- **Y** = down
- **Z** = forward
- Right-handed
- Units: meters

### Pipeline / TRC (OpenSim)

The coordinate system used in TRC marker files and OpenSim models.

- **X** = forward
- **Y** = up
- **Z** = right
- Right-handed
- Units: meters

When writing transformation code between these systems, always document which convention applies at each step. Never assume frames match between systems.
