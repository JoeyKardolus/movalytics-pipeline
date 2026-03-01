"""SAM 3D Body lifter — orchestrates SAM 3D inference subprocess.

Runs SAM 3D Body in a conda subprocess (Python 3.11) and returns raw
MHR body model output: 127-joint positions, rotations, model parameters.
No retargeting — downstream consumers use the MHR skeleton directly.

Coordinate convention:
  SAM 3D output positions: camera space (X=right, Y=down, Z=forward), meters
  SAM 3D output rotations: MHR body-centric (X=right, Y=up, Z=backward)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SAM3DResult:
    """Raw output from SAM 3D Body estimation.

    All arrays use the MHR body model's native coordinate conventions.
    Positions are in camera space; rotations are in MHR body-centric frame.

    Attributes:
        joint_coords: (N, 127, 3) MHR joint positions, body-model-relative meters.
            Camera-space = joint_coords + cam_t[:, None, :]
        global_rots: (N, 127, 3, 3) MHR global rotation matrices.
        cam_t: (N, 3) camera translation, camera space meters.
        model_params: (N, 204) raw MHR model parameters.
        shape_params: (N, 45) identity/shape coefficients.
        keypoints_3d: (N, 70, 3) surface keypoints, body-model-relative meters.
            Axes: X=right, Y=down, Z=forward (camera convention after Y/Z flip).
            Camera-space = keypoints_3d + cam_t[:, None, :]
        focal_length: (N,) per-frame focal length in pixels.
        rest_global_rots: (127, 3, 3) rest-pose global rotations (zero body pose).
        rest_joint_coords: (127, 3) rest-pose joint positions (zero body pose).
        rest_vertices: (V, 3) rest-pose mesh vertices, or empty if unavailable.
        rest_faces: (F, 3) MHR face connectivity (triangle indices), or empty.
        fps: Video frame rate.
        n_frames: Number of frames.
        success: Whether inference completed successfully.
        error: Error message if success is False.
    """
    joint_coords: np.ndarray    # (N, 127, 3)
    global_rots: np.ndarray     # (N, 127, 3, 3)
    cam_t: np.ndarray           # (N, 3)
    model_params: np.ndarray    # (N, 204)
    shape_params: np.ndarray    # (N, 45)
    keypoints_3d: np.ndarray    # (N, 70, 3)
    focal_length: np.ndarray    # (N,)
    rest_global_rots: np.ndarray  # (127, 3, 3)
    rest_joint_coords: np.ndarray  # (127, 3)
    rest_vertices: np.ndarray   # (V, 3)
    rest_faces: np.ndarray      # (F, 3)
    marker_positions: np.ndarray  # (N, M, 3) surface marker positions, or empty
    marker_names: list[str]     # M marker names matching marker_positions dim 1
    fps: float
    n_frames: int
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# SAM 3D Body lifter
# ---------------------------------------------------------------------------

class SAM3DLifter:
    """SAM 3D Body inference via conda subprocess.

    Runs SAM 3D Body in a conda subprocess (sam3d env), loads all raw
    outputs, and returns them as SAM3DResult.
    """

    def __init__(
        self,
        conda_env: str = "sam3d",
        checkpoint_path: str = "models/sam-3d-body-dinov3/model.ckpt",
        mhr_path: str = "models/sam-3d-body-dinov3/assets/mhr_model.pt",
        shape_stabilize: bool = True,
        temporal_smooth: bool = True,
        use_mask: bool = False,
        kalman_q_pos: float = 0.01,
        kalman_q_vel: float = 0.001,
        ema_alpha_static: float = 0.10,
    ):
        self.conda_env = conda_env
        self.shape_stabilize = shape_stabilize
        self.temporal_smooth = temporal_smooth
        self.use_mask = use_mask
        self.kalman_q_pos = kalman_q_pos
        self.kalman_q_vel = kalman_q_vel
        self.ema_alpha_static = ema_alpha_static
        self.worker_script = Path(__file__).parent.parent.parent / "workers" / "sam3d_worker.py"
        # Resolve relative paths to absolute (relative to project root)
        project_root = Path(__file__).parent.parent.parent.parent
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = project_root / self.checkpoint_path
        self.mhr_path = Path(mhr_path)
        if not self.mhr_path.is_absolute():
            self.mhr_path = project_root / self.mhr_path

    def _make_error_result(self, error: str, fps: float) -> SAM3DResult:
        """Create a failed SAM3DResult."""
        empty = np.array([])
        return SAM3DResult(
            joint_coords=empty, global_rots=empty, cam_t=empty,
            model_params=empty, shape_params=empty, keypoints_3d=empty,
            focal_length=empty, rest_global_rots=empty,
            rest_joint_coords=empty, rest_vertices=empty, rest_faces=empty,
            marker_positions=empty, marker_names=[],
            fps=fps, n_frames=0,
            success=False, error=error,
        )

    def estimate(
        self,
        video_path: str | Path,
        boxes: np.ndarray | None,
        fps: float,
        frames_shm_path: "Path | None" = None,
        frames_shape: tuple | None = None,
    ) -> SAM3DResult:
        """Run SAM 3D Body inference.

        Args:
            video_path: Path to input video.
            boxes: (N, 4) bounding boxes in xyxy pixel coordinates, or None
                to use vitdet per-frame detection (recommended).
            fps: Video frame rate.
            frames_shm_path: Path to shared memory memmap with pre-decoded
                RGB frames. Avoids re-reading video in subprocess.
            frames_shape: Shape of the frames array (N, H, W, 3).

        Returns:
            SAM3DResult with raw MHR body model output.
        """
        video_path = Path(video_path)
        t0 = time.perf_counter()

        # Run SAM 3D Body worker subprocess
        with tempfile.TemporaryDirectory(prefix="sam3d_") as tmpdir:
            tmpdir = Path(tmpdir)

            output_dir = tmpdir / "sam3d_output"

            # Use full path to conda env's Python to avoid PATH conflicts
            # (uv run puts .venv/bin first, which conda run inherits)
            import shutil
            conda_prefix = Path(shutil.which("conda")).parent.parent
            conda_python = conda_prefix / "envs" / self.conda_env / "bin" / "python"
            if not conda_python.exists():
                conda_python = "python"  # fallback

            print(f"[sam3d] Running SAM 3D Body worker ({self.conda_env} env)...")
            cmd = [
                "conda", "run", "-n", self.conda_env,
                str(conda_python), str(self.worker_script),
                "--video", str(video_path),
                "--output-dir", str(output_dir),
                "--checkpoint-path", str(self.checkpoint_path),
                "--mhr-path", str(self.mhr_path),
                "--kalman-q-pos", str(self.kalman_q_pos),
                "--kalman-q-vel", str(self.kalman_q_vel),
                "--ema-alpha-static", str(self.ema_alpha_static),
            ]
            # Pass pre-decoded frames via shared memory (avoids re-reading video)
            if frames_shm_path is not None and frames_shape is not None:
                cmd.extend([
                    "--frames-memmap", str(frames_shm_path),
                    "--frames-shape", ",".join(str(d) for d in frames_shape),
                    "--fps", str(fps),
                ])
            # Pass bounding boxes only if provided (otherwise vitdet detects)
            if boxes is not None:
                boxes_path = tmpdir / "boxes.npy"
                np.save(boxes_path, boxes.astype(np.float32))
                cmd.extend(["--boxes", str(boxes_path)])
            # Pass marker indices for surface marker extraction (if atlas exists)
            # __file__ = src/core/lifting/sam3d_lifter.py
            marker_indices_path = (
                Path(__file__).parent.parent / "conversion" / "mhr_marker_indices.npy"
            )
            if marker_indices_path.exists():
                cmd.extend(["--marker-indices", str(marker_indices_path)])

            if not self.shape_stabilize:
                cmd.append("--no-shape-stabilize")
            if not self.temporal_smooth:
                cmd.append("--no-temporal-smooth")
            if self.use_mask:
                cmd.append("--use-mask")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour max
                )
            except subprocess.TimeoutExpired:
                return self._make_error_result(
                    "SAM 3D Body worker timed out after 1 hour", fps,
                )

            # Print worker output, filtering spam from SAM 3D internals
            _SPAM_PREFIXES = (
                "Running on",
                "Predicting",
                "Loading",
                "Using cache",
                "Downloading",
                "100%",
                " ",  # progress bars
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    if any(line.strip().startswith(p) for p in _SPAM_PREFIXES):
                        continue
                    print(f"  {line}")
            if result.returncode != 0:
                error_msg = result.stderr[-2000:] if result.stderr else "unknown error"
                print(f"[sam3d] Worker failed:\n{error_msg}", file=sys.stderr)
                return self._make_error_result(
                    f"SAM 3D Body worker failed: {error_msg}", fps,
                )

            # Load ALL worker outputs
            joint_coords = np.load(output_dir / "joint_coords.npy")    # (N, 127, 3)
            global_rots = np.load(output_dir / "global_rots.npy")      # (N, 127, 3, 3)
            cam_t = np.load(output_dir / "cam_t.npy")                  # (N, 3)
            model_params = np.load(output_dir / "model_params.npy")    # (N, 204)
            shape_params = np.load(output_dir / "shape_params.npy")    # (N, 45)
            keypoints_3d = np.load(output_dir / "keypoints_3d.npy")    # (N, 70, 3)
            focal_length = np.load(output_dir / "focal_length.npy")    # (N,)

            # Rest-pose rotations (computed by worker with zero body pose)
            rest_rots_path = output_dir / "rest_global_rots.npy"
            if rest_rots_path.exists():
                rest_global_rots = np.load(rest_rots_path)  # (127, 3, 3)
            else:
                print("[sam3d] WARNING: rest_global_rots.npy not found, using identity")
                rest_global_rots = np.tile(np.eye(3, dtype=np.float32), (127, 1, 1))

            # Rest-pose joint coords (for atlas building)
            rest_jcoords_path = output_dir / "rest_joint_coords.npy"
            rest_joint_coords = np.load(rest_jcoords_path) if rest_jcoords_path.exists() else np.array([])

            # Rest-pose mesh (for atlas building)
            rest_verts_path = output_dir / "rest_vertices.npy"
            rest_vertices = np.load(rest_verts_path) if rest_verts_path.exists() else np.array([])

            rest_faces_path = output_dir / "rest_faces.npy"
            rest_faces = np.load(rest_faces_path) if rest_faces_path.exists() else np.array([])

            # Per-frame surface marker positions (from atlas vertex extraction)
            marker_pos_path = output_dir / "marker_positions.npy"
            if marker_pos_path.exists():
                marker_positions = np.load(marker_pos_path)  # (N, M, 3)
                # Load marker names from atlas JSON
                import json
                names_path = (
                    Path(__file__).parent.parent / "conversion" / "mhr_marker_names.json"
                )
                if names_path.exists():
                    marker_names = json.loads(names_path.read_text())
                else:
                    # Fallback: load from atlas module
                    from src.core.conversion.mhr_marker_atlas import MHR_SURFACE_MARKER_NAMES
                    marker_names = MHR_SURFACE_MARKER_NAMES
            else:
                marker_positions = np.array([])
                marker_names = []

        n_frames = joint_coords.shape[0]
        total_time = time.perf_counter() - t0
        print(f"[sam3d] {n_frames} frames in {total_time:.1f}s "
              f"({n_frames / total_time:.1f} fps)")

        if rest_vertices.size > 0:
            n_verts = rest_vertices.shape[0]
            n_faces = rest_faces.shape[0] if rest_faces.size > 0 else 0
            print(f"[sam3d] Rest-pose mesh: {n_verts} vertices, {n_faces} faces")

        if marker_positions.size > 0:
            print(f"[sam3d] Surface markers: {marker_positions.shape[1]} markers, "
                  f"{marker_positions.shape[0]} frames")

        return SAM3DResult(
            joint_coords=joint_coords,
            global_rots=global_rots,
            cam_t=cam_t,
            model_params=model_params,
            shape_params=shape_params,
            keypoints_3d=keypoints_3d,
            focal_length=focal_length,
            rest_global_rots=rest_global_rots,
            rest_joint_coords=rest_joint_coords,
            rest_vertices=rest_vertices,
            rest_faces=rest_faces,
            marker_positions=marker_positions,
            marker_names=marker_names,
            fps=fps,
            n_frames=n_frames,
            success=True,
        )
