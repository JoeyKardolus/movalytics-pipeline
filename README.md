# Movalytics Pipeline

3D human pose estimation from monocular video. Clinical joint angles and a scaled musculoskeletal model -- the same measurements a motion capture lab would give you, from a single camera.

```
Video --> YOLOX Detection --> SAM 3D Body (127 joints) --> Clinical Angles (30 DOFs) --> OpenSim IK
```

---

## What You Get

```
data/output/<video>/
├── <video>_sam3d.npz             # Raw MHR body model (127 joints, rotations, shape, mesh)
├── <video>_mhr_markers.trc       # Anatomical surface markers
├── <video>.osim                  # Scaled OpenSim model
├── <video>_ik.mot                # OpenSim IK joint angles
├── <video>_fk_bodies.npz         # FK body positions (for demo page)
├── <video>_sam3d_angles.png      # Clinical angle plots
└── joint_angles/                 # CSV per joint group (30 DOFs)
```

---

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | 3.12 | Managed by uv |
| [uv](https://docs.astral.sh/uv/) | Latest | Python package manager |
| [Conda](https://docs.conda.io/) | Any | Required for `sam3d` and `opensim` environments |
| NVIDIA GPU | Recommended | 8 GB+ VRAM for real-time inference |

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/<user>/movalytics-pipeline.git
cd movalytics-pipeline
```

### 2. Install Python dependencies

```bash
uv sync
```

### 3. Download model weights

SAM 3D Body weights (~2.7 GB) are not included in the repository.

```bash
uv run python scripts/tools/download_models.py
```

### 4. Create the sam3d conda environment

SAM 3D Body requires Python 3.11 with detectron2 and PyTorch nightly.

```bash
conda create -n sam3d python=3.11 -y
conda activate sam3d
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install scipy opencv-python-headless chumpy smplx trimesh pyrender
pip install 'git+https://github.com/EasternJournalist/utils3d.git'
conda deactivate
```

Verify:

```bash
conda run -n sam3d python -c "import torch; import detectron2; print('OK')"
```

### 5. Create the opensim conda environment

```bash
conda create -n opensim python=3.12 -y
conda activate opensim
conda install -c opensim-org opensim=4.5.2
pip install scipy
conda deactivate
```

Verify:

```bash
conda run -n opensim python -c "import opensim; print('OK')"
```

### 6. Run the pipeline

```bash
uv run python main.py --video data/input/walking.mp4 --height 1.75
```

---

## Demo Page

After processing one or more videos, generate a shareable HTML demo:

```bash
uv run python scripts/viz/demo_page.py
open data/demo/demo.html
```

The demo page auto-discovers all processed videos in `data/output/` and generates a standalone HTML file with:

- Three.js 3D bone mesh animations synced with video playback
- Interactive Chart.js clinical angle charts with L/R overlay
- Normative ROM bands for activity comparison

To set the activity type for a video (used for normative data comparison), create `data/output/<video>/metadata.json`:

```json
{"activity": "walking"}
```

Supported activities: `walking`, `running`, `cycling`, `pushup`, `squat`, `jumprope`, `general`.

---

## CLI Reference

| Flag | Purpose |
|------|---------|
| `--video VIDEO` | Input video file (required) |
| `--height N` / `--mass N` | Subject height (m) and mass (kg) |
| `--visibility-min N` | Confidence threshold (default 0.3) |
| `--plot-joint-angles` | Generate joint angle visualization PNGs |
| `--save-angle-comparison` | Save L/R comparison plot |
| `--movement-analysis` | Analyze against normative data |
| `--temporal-smoothing N` | Temporal smoothing window (0=disabled) |
| `--config path.yaml` | Load YAML config (CLI overrides) |
| `--dump-config` | Print full default config as YAML |

---

## Configuration

All ~60 parameters are exposed via YAML. See [docs/CONFIG.md](docs/CONFIG.md) for the full reference.

```bash
uv run python main.py --dump-config > config.yaml
# edit config.yaml...
uv run python main.py --video input.mp4 --config config.yaml
```

CLI flags always override values from the config file.

---

## Pipeline Details

### 1. YOLOX Detection

Detects person bounding boxes in each frame. IOU-based tracking associates detections across frames, and OneEuro smoothing reduces jitter in the bounding box coordinates before they are passed downstream.

### 2. SAM 3D Body

Meta's 840M-parameter model (DINOv3 backbone) estimates a 127-joint MHR body model with per-joint rotations, body shape, and mesh vertices. Runs as a subprocess via `conda run -n sam3d`. Post-processing applies shape stabilization (median PCA), Kalman filtering on body pose, and EMA smoothing on global rotation.

### 3. Clinical Angles

Extracts 30 degrees of freedom from the MHR global rotation matrices: hip, knee, ankle, trunk, shoulder, elbow, and wrist -- each decomposed into flexion/extension, abduction/adduction, and internal/external rotation.

### 4. Surface Markers

65 anatomical surface markers are sampled from MHR mesh vertices using a pre-built atlas (`build_mhr_atlas.py`). The markers are exported in TRC format for consumption by OpenSim.

### 5. OpenSim IK

A scaled LaiUhlrich2022 full-body musculoskeletal model is generated from the subject's height and marker positions. Inverse kinematics solves joint angles that best fit the TRC markers, and a forward kinematics pass exports body segment positions for the demo page.

---

## Visualization Scripts

| Script | Input | Purpose |
|--------|-------|---------|
| `scripts/viz/demo_page.py` | `data/output/` | Generate shareable HTML demo |
| `scripts/viz/opensim_mot_viewer.py` | `_ik.mot` | OpenSim MOT file viewer |
| `scripts/viz/sam3d_mesh_viewer.py` | `_sam3d.npz` | 3D mesh viewer |
| `scripts/viz/trc_3d_viewer.py` | `.trc` | TRC marker skeleton |

See [scripts/README.md](scripts/README.md) for full documentation.

---

## Tools

| Script | Purpose |
|--------|---------|
| `scripts/tools/download_models.py` | Download SAM 3D Body weights |
| `scripts/tools/build_mhr_atlas.py` | Build MHR surface marker atlas |
| `scripts/tools/validate_marker_atlas.py` | Validate marker atlas |
| `scripts/tools/auto_site_markers.py` | Auto-position site markers |
| `scripts/tools/export_mhr_mesh_json.py` | Export MHR mesh for WebGL |

---

## Project Structure

```
movalytics-pipeline/
├── main.py                        # CLI entry point
├── pyproject.toml                 # Project metadata and dependencies
├── data/
│   ├── input/                     # Input videos
│   ├── output/                    # Per-video output directories
│   └── demo/                      # Generated demo HTML
├── docs/                          # Documentation
├── lib/
│   └── sam-3d-body/               # Vendored SAM 3D Body (Meta)
├── models/                        # Model weights (not in repo)
├── scripts/
│   ├── tools/                     # Utility scripts (download, atlas, export)
│   └── viz/                       # Visualization scripts
└── src/
    ├── api/                       # Django REST API
    ├── application/               # Django project settings
    ├── core/
    │   ├── config.py              # Pipeline configuration
    │   ├── conversion/            # TRC I/O, clinical angles, marker atlas
    │   ├── detection/             # YOLOX + IOU tracker + OneEuro
    │   ├── evaluation/            # Angle metrics (MAE, MJAE, correlation)
    │   ├── kinematics/            # Joint angle export and plots
    │   ├── lifting/               # SAM 3D Body + OpenSim IK subprocesses
    │   ├── pipeline/              # Output cleanup and organization
    │   └── video/                 # Video I/O (OpenCV)
    ├── infra/                     # Infrastructure and deployment
    ├── shared/                    # Constants, coordinate transforms
    └── workers/                   # Background workers
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `model.ckpt` missing or too small | Run `uv run python scripts/tools/download_models.py` |
| `sam3d` conda env missing | Follow step 4 in Getting Started |
| `opensim` conda env missing | Follow step 5 in Getting Started |
| No person detected | Lower the threshold: `--visibility-min 0.1` |
| OpenSim IK fails | Verify: `conda run -n opensim python -c "import opensim"` |
| SAM 3D subprocess hangs | Check: `conda run -n sam3d python --version` returns 3.11 |
| PyTorch CUDA error | RTX 5080 needs sm_120 -- requires PyTorch nightly or >=2.7 |

---

## Citation

- **SAM 3D Body**: Choutas et al. (Meta) -- [github.com/facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
- **OpenSim**: Seth et al. 2018 -- [doi.org/10.1371/journal.pcbi.1006223](https://doi.org/10.1371/journal.pcbi.1006223)
- **LaiUhlrich2022**: Lai, Uhlrich et al. -- [simtk.org/projects/full_body](https://simtk.org/projects/full_body)

---

## License

CC BY-NC-SA 4.0 -- Research use only. See [LICENSE](LICENSE).

Vendored `lib/sam-3d-body/` is under Meta's SAM License.
