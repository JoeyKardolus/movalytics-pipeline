# Configuration Reference

Full configuration reference for the movalytics-pipeline. The config system uses YAML files with CLI overrides.

## Overview

Generate the default configuration:

```bash
uv run python main.py --dump-config > config.yaml
```

**Priority order** (highest to lowest):

1. CLI flags
2. Config file values
3. Hardcoded defaults

Load a config file:

```bash
uv run python main.py --video data/input/video.mp4 --config config.yaml
```

CLI flags override any values set in the config file.

---

## Config Sections

### subject

Subject physical parameters used for model scaling and metric output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `height` | float | `1.78` | Subject height in meters; enables metric scale recovery |
| `mass` | float | `75.0` | Subject mass in kg |

### detection

YOLOX person detection settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | str | `"m"` | YOLOX model size: `s`, `m`, or `l` |
| `visibility_min` | float | `0.3` | Minimum landmark confidence threshold |

### detection.tracking

IOU-based bounding box tracker parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_age` | int | `30` | Frames before a track is dropped |
| `min_iou` | float | `0.3` | Minimum IOU for bounding box matching |
| `confirm_hits` | int | `3` | Consecutive frames to confirm a new track |

### detection.smoothing

OneEuro filter parameters for detection smoothing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_cutoff` | float | `10.0` | Frequency cutoff (lower = more smoothing) |
| `beta` | float | `2.0` | Speed coefficient (higher = less lag) |
| `d_cutoff` | float | `1.0` | Derivative cutoff |

### lifting.sam3d

SAM 3D Body model and post-processing configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conda_env` | str | `"sam3d"` | Conda environment with SAM 3D Body |
| `checkpoint_path` | str | `"models/sam-3d-body-dinov3/model.ckpt"` | Path to SAM 3D Body checkpoint |
| `mhr_path` | str | `"models/sam-3d-body-dinov3/assets/mhr_model.pt"` | Path to MHR body model |
| `calibration_frames` | int | `0` | Static calibration: median of first N frames as zero-pose |
| `shape_stabilize` | bool | `true` | Freeze shape/scale to median across all frames |
| `use_mask` | bool | `false` | Mask conditioning via SAM3 segmentor |
| `temporal_smooth` | bool | `true` | Kalman + EMA post-processing on pose/rotation |
| `kalman_q_pos` | float | `0.01` | Kalman process noise (position) |
| `kalman_q_vel` | float | `0.001` | Kalman process noise (velocity) |
| `ema_alpha_static` | float | `0.10` | EMA alpha for static global rotation segments |

### lifting.opensim

OpenSim inverse kinematics and forward kinematics configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conda_env` | str | `"opensim"` | Conda environment with OpenSim |
| `skip_fk` | bool | `false` | Skip FK body export (disables demo page skeleton) |

### movement_analysis

Normative data comparison settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Compare against normative data |

### post_processing

Final output post-processing options.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temporal_smoothing` | int | `0` | Final TRC smoothing window (0 = disabled) |

### visualization

Plot and visualization output options.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot_joint_angles` | bool | `false` | Generate joint angle PNGs |
| `save_angle_comparison` | bool | `false` | Save L/R comparison plot |

---

## CLI-to-Config Mapping

| CLI Flag | Config Path |
|----------|-------------|
| `--height` | `subject.height` |
| `--mass` | `subject.mass` |
| `--visibility-min` | `detection.visibility_min` |
| `--movement-analysis` | `movement_analysis.enabled` |
| `--temporal-smoothing` | `post_processing.temporal_smoothing` |
| `--plot-joint-angles` | `visualization.plot_joint_angles` |
| `--save-angle-comparison` | `visualization.save_angle_comparison` |

---

## Example Configs

### 1. Minimal -- subject parameters only

```yaml
subject:
  height: 1.82
  mass: 80.0
```

All other settings use hardcoded defaults. This is the simplest useful config: subject height enables metric scale recovery, and mass is used for OpenSim model scaling.

### 2. Speed-optimized -- disable FK, no plots

```yaml
subject:
  height: 1.78
  mass: 75.0

detection:
  model_size: "s"

lifting:
  sam3d:
    shape_stabilize: false
    temporal_smooth: false
  opensim:
    skip_fk: true

post_processing:
  temporal_smoothing: 0

visualization:
  plot_joint_angles: false
  save_angle_comparison: false
```

Uses the smallest YOLOX model, disables SAM 3D post-processing (shape stabilization, Kalman/EMA), skips FK body export (which means demo page skeleton will not be available), and turns off all visualization output. Suitable for batch processing where only IK joint angle CSVs are needed.

### 3. Full quality -- all post-processing enabled

```yaml
subject:
  height: 1.78
  mass: 75.0

detection:
  model_size: "l"
  visibility_min: 0.1
  tracking:
    max_age: 45
    min_iou: 0.25
    confirm_hits: 5
  smoothing:
    min_cutoff: 5.0
    beta: 1.5
    d_cutoff: 1.0

lifting:
  sam3d:
    shape_stabilize: true
    temporal_smooth: true
    kalman_q_pos: 0.005
    kalman_q_vel: 0.0005
    ema_alpha_static: 0.05
  opensim:
    skip_fk: false

movement_analysis:
  enabled: true

post_processing:
  temporal_smoothing: 5

visualization:
  plot_joint_angles: true
  save_angle_comparison: true
```

Uses the largest YOLOX model with a lower visibility threshold to retain more landmarks. Tighter Kalman noise values and a lower EMA alpha produce smoother output at the cost of slightly more temporal lag. Final TRC smoothing window of 5 frames applied. Movement analysis compares against normative data. All visualization outputs enabled.
