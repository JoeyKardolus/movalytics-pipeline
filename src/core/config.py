"""Centralized pipeline configuration.

Single source of truth for every tunable parameter across all pipeline
stages.  Use ``--config path/to/config.yaml`` to load a config file, or
``--dump-config`` to print the full default config with comments.

Priority: CLI flags > config file > hardcoded defaults.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from dataclasses import dataclass, field, fields, asdict
from io import StringIO
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------

@dataclass
class SubjectConfig:
    height: float = 1.78
    mass: float = 75.0


@dataclass
class TrackingConfig:
    max_age: int = 30
    min_iou: float = 0.3
    confirm_hits: int = 3


@dataclass
class SmoothingConfig:
    min_cutoff: float = 10.0
    beta: float = 2.0
    d_cutoff: float = 1.0


@dataclass
class DetectionConfig:
    model_size: str = "m"
    visibility_min: float = 0.3
    detect_stride: int = 3
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)


@dataclass
class SAM3DConfig:
    conda_env: str = "sam3d"
    checkpoint_path: str = "models/sam-3d-body-dinov3/model.ckpt"
    mhr_path: str = "models/sam-3d-body-dinov3/assets/mhr_model.pt"
    calibration_frames: int = 0  # Static calibration: median of first N frames as zero
    shape_stabilize: bool = True        # Freeze shape/scale to median across all frames
    use_mask: bool = False              # Mask conditioning via SAM3 segmentor
    temporal_smooth: bool = True        # Kalman + EMA post-processing on pose/rotation
    kalman_q_pos: float = 0.01          # Kalman process noise (position)
    kalman_q_vel: float = 0.001         # Kalman process noise (velocity)
    ema_alpha_static: float = 0.10      # EMA alpha for static global rotation segments


@dataclass
class OpenSimConfig:
    conda_env: str = "opensim"
    skip_fk: bool = False  # FK body export needed for demo page skeleton visualization


@dataclass
class LiftingConfig:
    sam3d: SAM3DConfig = field(default_factory=SAM3DConfig)
    opensim: OpenSimConfig = field(default_factory=OpenSimConfig)


@dataclass
class MovementAnalysisConfig:
    enabled: bool = False


@dataclass
class PostProcessingConfig:
    temporal_smoothing: int = 0


@dataclass
class VisualizationConfig:
    plot_joint_angles: bool = False
    save_angle_comparison: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Complete pipeline configuration.

    Every tunable parameter in the pipeline is exposed here.
    Generate a default YAML with ``dump_default_config()``.
    """

    subject: SubjectConfig = field(default_factory=SubjectConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    lifting: LiftingConfig = field(default_factory=LiftingConfig)
    movement_analysis: MovementAnalysisConfig = field(default_factory=MovementAnalysisConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

# Comments keyed by dotted path (section or field)
_COMMENTS: dict[str, str] = {
    # Section headers
    "subject": "Subject",
    "detection": "YOLOX Detection",
    "detection.tracking": "IOU tracker",
    "detection.smoothing": "OneEuro filter",
    "lifting": "3D Lifting",
    "lifting.sam3d": "SAM 3D Body (Meta) — raw MHR body model output",
    "lifting.sam3d.conda_env": "conda environment with SAM 3D Body installed",
    "lifting.sam3d.checkpoint_path": "path to SAM 3D Body model.ckpt",
    "lifting.sam3d.mhr_path": "path to MHR body model (mhr_model.pt)",
    "lifting.opensim": "OpenSim IK (LaiUhlrich2022)",
    "lifting.opensim.conda_env": "conda environment with OpenSim installed",
    "lifting.opensim.skip_fk": "skip FK body export (disables demo page skeleton rendering)",
    "movement_analysis": "Movement Analysis",
    "post_processing": "Post-Processing",
    "visualization": "Visualization",
    # Field descriptions
    "subject.height": "meters, enables metric scale recovery",
    "subject.mass": "kg",
    "detection.model_size": "{s, m, l} — YOLOX model size",
    "detection.visibility_min": "minimum landmark confidence",
    "detection.tracking.max_age": "frames before track is dropped",
    "detection.tracking.min_iou": "minimum IOU for bbox matching",
    "detection.tracking.confirm_hits": "frames to confirm a new track",
    "detection.smoothing.min_cutoff": "frequency cutoff (lower = more smoothing)",
    "detection.smoothing.beta": "speed coefficient (higher = less lag)",
    "detection.smoothing.d_cutoff": "derivative cutoff",
    "movement_analysis.enabled": "compare against normative data",
    "post_processing.temporal_smoothing": "final TRC smoothing window (0 = disabled)",
    "visualization.plot_joint_angles": "generate joint angle PNGs",
    "visualization.save_angle_comparison": "L/R comparison plot",
}


def _yaml_value(v: Any) -> str:
    """Format a Python value as YAML."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.1e}"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.4f}"
        return f"{v:.4g}" if v != int(v) else f"{v:.1f}"
    return str(v)


def _dump_section(obj: Any, indent: int, prefix: str, out: StringIO) -> None:
    """Recursively dump a dataclass as commented YAML."""
    pad = "  " * indent
    for f in fields(obj):
        val = getattr(obj, f.name)
        key = f"{prefix}.{f.name}" if prefix else f.name
        comment = _COMMENTS.get(key, "")

        if hasattr(val, "__dataclass_fields__"):
            # Sub-section header
            if indent > 0:
                out.write("\n")
            section_label = _COMMENTS.get(key, f.name)
            out.write(f"\n{pad}# {'─' * 2} {section_label} {'─' * max(1, 50 - len(section_label) - indent * 2)}\n")
            out.write(f"{pad}{f.name}:\n")
            _dump_section(val, indent + 1, key, out)
        else:
            formatted = _yaml_value(val)
            if comment:
                out.write(f"{pad}{f.name}: {formatted:<20s} # {comment}\n")
            else:
                out.write(f"{pad}{f.name}: {formatted}\n")


def dump_default_config() -> str:
    """Generate the full default config as commented YAML.

    Usage::

        uv run python main.py --dump-config > pipeline_config.yaml
    """
    cfg = PipelineConfig()
    out = StringIO()
    out.write("# Pipeline Configuration\n")
    out.write("# Generated with: uv run python main.py --dump-config\n")
    out.write("# CLI flags override values set here.\n")
    _dump_section(cfg, 0, "", out)
    return out.getvalue()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    """Recursively build a dataclass from a dict."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        # Check if the field type is itself a dataclass
        field_type = f.type
        if isinstance(field_type, str):
            # Resolve forward refs from the local namespace
            field_type = globals().get(field_type, field_type)
        if isinstance(field_type, type) and hasattr(field_type, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[f.name] = _dict_to_dataclass(field_type, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: str | Path | None) -> PipelineConfig:
    """Load a YAML config file and merge with defaults.

    Missing keys use the default values from ``PipelineConfig``.
    """
    if path is None:
        return PipelineConfig()

    path = Path(path)
    if not path.exists():
        print(f"[config] WARNING: Config file not found: {path}", file=sys.stderr)
        return PipelineConfig()

    try:
        import yaml
    except ImportError:
        print("[config] WARNING: PyYAML not installed. Install with: uv add pyyaml", file=sys.stderr)
        return PipelineConfig()

    with open(path) as f:
        user_data = yaml.safe_load(f) or {}

    # Start from defaults, merge user overrides
    defaults = asdict(PipelineConfig())
    merged = _deep_merge(defaults, user_data)
    return _dict_to_dataclass(PipelineConfig, merged)


# ---------------------------------------------------------------------------
# CLI ↔ Config bridge
# ---------------------------------------------------------------------------

# Map CLI arg name -> dotted config path
_CLI_TO_CONFIG: dict[str, str] = {
    # Subject
    "height": "subject.height",
    "mass": "subject.mass",
    # Detection
    "visibility_min": "detection.visibility_min",
    # Movement Analysis
    "movement_analysis": "movement_analysis.enabled",
    # Post-Processing
    "temporal_smoothing": "post_processing.temporal_smoothing",
    # Visualization
    "plot_joint_angles": "visualization.plot_joint_angles",
    "save_angle_comparison": "visualization.save_angle_comparison",
}


def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a nested attribute on a dataclass by dotted path."""
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _get_nested(obj: Any, dotted_path: str) -> Any:
    """Get a nested attribute from a dataclass by dotted path."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def apply_cli_overrides(config: PipelineConfig, args: Namespace) -> PipelineConfig:
    """Apply CLI argument overrides to a config.

    Only overrides values that the user explicitly set on the command line
    (i.e. differ from argparse defaults).  Boolean store_true flags are
    only applied when ``True`` (user passed the flag).
    """
    for cli_name, config_path in _CLI_TO_CONFIG.items():
        if not hasattr(args, cli_name):
            continue
        cli_val = getattr(args, cli_name)

        # For store_true flags: only override if user actually passed the flag
        if isinstance(cli_val, bool):
            if cli_val:
                _set_nested(config, config_path, True)
            # Don't override config=True with CLI default False
            continue

        # For valued args: check if user set it (compare to argparse default)
        # We detect this by checking _explicitly_set (populated by our parse_args)
        explicitly_set = getattr(args, "_explicitly_set", set())
        if cli_name in explicitly_set:
            _set_nested(config, config_path, cli_val)
        elif cli_val is not None and not isinstance(cli_val, bool):
            # If no explicit tracking, still apply non-default non-None values
            current = _get_nested(config, config_path)
            if cli_val != current:
                _set_nested(config, config_path, cli_val)

    return config
