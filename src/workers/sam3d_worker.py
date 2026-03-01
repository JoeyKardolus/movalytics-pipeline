"""SAM 3D Body subprocess worker.

Runs in `conda run -n sam3d` (Python 3.11) environment.
Reads video frames, runs SAM 3D Body inference per-frame, saves results as .npy.

Post-processing pipeline (SAM-Body4D inspired):
  1. Shape stabilization — median shape/scale across all frames, MHR forward re-run
  2. Temporal smoothing — Kalman filter on body pose, EMA on global rotation
  3. Mask conditioning — optional SAM3 segmentor for background removal

Usage:
    conda run -n sam3d python src/workers/sam3d_worker.py \
        --video /path/to/video.mp4 \
        --boxes /path/to/boxes.npy \
        --output-dir /path/to/output/ \
        --checkpoint-path /path/to/model.ckpt \
        --mhr-path /path/to/mhr_model.pt \
        [--no-shape-stabilize] [--no-temporal-smooth] [--use-mask]
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

# Add vendored sam-3d-body library to sys.path — enables both `sam_3d_body`
# package and `tools.*` imports (build_fov_estimator, etc.)
_project_root = Path(__file__).resolve().parent.parent.parent
_sam3d_root = _project_root / "lib" / "sam-3d-body"
if str(_sam3d_root) not in sys.path:
    sys.path.insert(0, str(_sam3d_root))

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Kalman smoother (constant-velocity, forward-backward)
# Adapted from SAM-Body4D (gaomingqi/sam-body4d)
# ---------------------------------------------------------------------------

def _kalman_smooth(data: np.ndarray, q_pos: float = 0.01,
                   q_vel: float = 0.001) -> np.ndarray:
    """Constant-velocity Kalman smoother (forward-backward).

    Operates independently on each dimension of a (T, D) array.
    Uses factored scalar covariance per dimension for efficiency.

    Args:
        data: (T, D) time series.
        q_pos: Process noise for position.
        q_vel: Process noise for velocity.

    Returns:
        (T, D) smoothed data.
    """
    T, D = data.shape
    if T < 3:
        return data.copy()

    # Adaptive observation noise: proportional to local motion magnitude
    diffs = np.diff(data, axis=0)  # (T-1, D)
    motion_mag = np.sqrt((diffs ** 2).sum(axis=1))  # (T-1,)
    median_motion = max(np.median(motion_mag), 1e-6)

    # Forward pass
    x_fwd = np.zeros_like(data)
    v_fwd = np.zeros((T, D))
    Pxx_fwd = np.zeros((T, D))
    Pxv_fwd = np.zeros((T, D))
    Pvv_fwd = np.zeros((T, D))

    # Initialize from first two frames
    x_fwd[0] = data[0]
    v_fwd[0] = data[1] - data[0] if T > 1 else 0.0
    Pxx_fwd[0] = q_pos * 10
    Pxv_fwd[0] = 0.0
    Pvv_fwd[0] = q_vel * 10

    for t in range(1, T):
        # Predict
        x_pred = x_fwd[t - 1] + v_fwd[t - 1]
        v_pred = v_fwd[t - 1]
        Pxx_pred = Pxx_fwd[t - 1] + 2 * Pxv_fwd[t - 1] + Pvv_fwd[t - 1] + q_pos
        Pxv_pred = Pxv_fwd[t - 1] + Pvv_fwd[t - 1]
        Pvv_pred = Pvv_fwd[t - 1] + q_vel

        # Adaptive observation noise
        r_obs = max(motion_mag[min(t - 1, T - 2)] / median_motion, 0.1) * q_pos * 5

        # Update
        S = Pxx_pred + r_obs
        S = np.maximum(S, 1e-10)
        K_pos = Pxx_pred / S
        K_vel = Pxv_pred / S
        innov = data[t] - x_pred

        x_fwd[t] = x_pred + K_pos * innov
        v_fwd[t] = v_pred + K_vel * innov
        Pxx_fwd[t] = (1 - K_pos) * Pxx_pred
        Pxv_fwd[t] = (1 - K_pos) * Pxv_pred
        Pvv_fwd[t] = Pvv_pred - K_vel * Pxv_pred

        # Guard against numerical blow-up
        Pxx_fwd[t] = np.clip(np.nan_to_num(Pxx_fwd[t]), 0, 100)
        Pxv_fwd[t] = np.clip(np.nan_to_num(Pxv_fwd[t]), -100, 100)
        Pvv_fwd[t] = np.clip(np.nan_to_num(Pvv_fwd[t]), 0, 100)

    # Backward pass
    x_bwd = np.zeros_like(data)
    x_bwd[-1] = x_fwd[-1]

    for t in range(T - 2, -1, -1):
        Pxx_pred = Pxx_fwd[t] + 2 * Pxv_fwd[t] + Pvv_fwd[t] + q_pos
        Pxx_pred = np.maximum(Pxx_pred, 1e-10)
        # Backward gain
        L = (Pxx_fwd[t] + Pxv_fwd[t]) / Pxx_pred
        L = np.clip(np.nan_to_num(L), -1, 1)
        x_bwd[t] = x_fwd[t] + L * (x_bwd[t + 1] - (x_fwd[t] + v_fwd[t]))

    return x_bwd


# ---------------------------------------------------------------------------
# EMA smoother for global rotation (static/dynamic adaptive)
# ---------------------------------------------------------------------------

def _smooth_global_rot_quat(global_rot: np.ndarray,
                            outlier_threshold_deg: float = 60.0,
                            alpha: float = 0.3) -> np.ndarray:
    """Quaternion-space global rotation smoother with outlier rejection.

    Operates in quaternion space to handle 180° orientation flips that
    Euler-space averaging cannot. Steps:
    1. ZYX Euler → quaternion
    2. Sign continuity (q ≡ -q)
    3. Outlier detection via geodesic distance
    4. SLERP interpolation of invalid/outlier frames
    5. Quaternion EMA
    6. Quaternion → ZYX Euler

    Args:
        global_rot: (T, 3) global rotation ZYX Euler angles (radians).
        outlier_threshold_deg: Geodesic distance threshold for outlier detection.
        alpha: EMA smoothing factor (lower = more smoothing).

    Returns:
        (T, 3) smoothed global rotation ZYX Euler angles (radians).
    """
    from scipy.spatial.transform import Rotation, Slerp

    T = global_rot.shape[0]
    if T < 3:
        return global_rot.copy()

    # Mark invalid frames early: NaN or all-zero euler
    valid = np.ones(T, dtype=bool)
    for t in range(T):
        if np.any(np.isnan(global_rot[t])) or np.allclose(global_rot[t], 0, atol=1e-6):
            valid[t] = False

    # Replace invalid frames with zeros before Euler→quat (avoids NaN quaternions)
    clean_rot = global_rot.copy()
    clean_rot[~valid] = 0.0

    # Euler → quaternion
    rots = Rotation.from_euler("ZYX", clean_rot, degrees=False)
    quats = rots.as_quat()  # (T, 4) [x,y,z,w]

    # Sign continuity (only valid frames matter, but apply to all for consistency)
    for t in range(1, T):
        if np.dot(quats[t], quats[t - 1]) < 0:
            quats[t] = -quats[t]

    # Orientation outliers: geodesic distance to previous valid frame > threshold
    threshold_rad = np.radians(outlier_threshold_deg)
    n_outliers = 0
    for t in range(1, T):
        if not valid[t]:
            continue
        # Find previous valid frame
        prev = t - 1
        while prev >= 0 and not valid[prev]:
            prev -= 1
        if prev < 0:
            continue
        dot_val = np.clip(np.abs(np.dot(quats[t], quats[prev])), 0, 1)
        angle = 2 * np.arccos(dot_val)
        if angle > threshold_rad:
            valid[t] = False
            n_outliers += 1

    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        print(f"  Global rot: {n_invalid} invalid frames "
              f"({n_outliers} outliers, {n_invalid - n_outliers} zero-detection)")

    # SLERP interpolation for invalid frames from valid neighbors
    valid_indices = np.where(valid)[0]
    if len(valid_indices) < 2:
        return global_rot.copy()

    invalid_indices = np.where(~valid)[0]
    if len(invalid_indices) > 0:
        valid_rots = Rotation.from_quat(quats[valid_indices])
        slerp = Slerp(valid_indices.astype(float), valid_rots)
        # Clamp to valid range (no extrapolation)
        interp_times = np.clip(invalid_indices.astype(float),
                               valid_indices[0], valid_indices[-1])
        interp_rots = slerp(interp_times)
        quats[invalid_indices] = interp_rots.as_quat()
        # Re-apply sign continuity after interpolation
        for t in range(1, T):
            if np.dot(quats[t], quats[t - 1]) < 0:
                quats[t] = -quats[t]

    # Quaternion EMA (component-wise + re-normalize)
    smoothed = quats.copy()
    for t in range(1, T):
        smoothed[t] = alpha * quats[t] + (1 - alpha) * smoothed[t - 1]
        norm = np.linalg.norm(smoothed[t])
        smoothed[t] /= max(norm, 1e-8)

    # Back to ZYX Euler
    result = Rotation.from_quat(smoothed).as_euler("ZYX", degrees=False)
    return result


# ---------------------------------------------------------------------------
# MHR forward re-run (shape stabilization + temporal smoothing)
# ---------------------------------------------------------------------------

def _rerun_mhr_forward(
    estimator,
    body_pose: np.ndarray,
    global_rot: np.ndarray,
    scale_pca: np.ndarray,
    hand_pca: np.ndarray,
    shape_override: np.ndarray | None,
    return_vertices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-run MHR forward kinematics with stabilized/smoothed parameters.

    Uses raw PCA coefficients (NOT decoded values from mhr_model_params).
    Replicates MHRHead.forward post-processing: Y/Z flip + keypoint truncation.

    Args:
        estimator: SAM3DBodyEstimator with head_pose.
        body_pose: (N, 130) body pose euler angles.
        global_rot: (N, 3) global rotation ZYX euler.
        scale_pca: (N, 28) raw PCA scale coefficients.
        hand_pca: (N, H) raw PCA hand coefficients.
        shape_override: (45,) fixed shape params, or None for zeros.
        return_vertices: If True, also return per-frame mesh vertices.

    Returns:
        (joint_coords, global_rots, keypoints_3d) — in camera convention
        (Y/Z flipped to match process_one_image output).
        If return_vertices=True, also returns vertices (N, V, 3) in MHR
        body-centric coords (NO Y/Z flip — matches rest_vertices convention).
    """
    head = estimator.model.head_pose
    N = body_pose.shape[0]

    # Process in batches to avoid GPU OOM
    batch_size = 64
    all_jcoords = []
    all_grots = []
    all_kp3d = []
    all_verts = [] if return_vertices else None

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        b = end - start

        with torch.no_grad():
            # global_trans is always zero (matches MHRHead.forward line 296)
            gt = torch.zeros(b, 3).float().cuda()
            gr = torch.from_numpy(global_rot[start:end]).float().cuda()
            bp = torch.from_numpy(body_pose[start:end]).float().cuda()
            sp = torch.from_numpy(scale_pca[start:end]).float().cuda()
            hp = torch.from_numpy(hand_pca[start:end]).float().cuda()

            if shape_override is not None:
                sh = torch.from_numpy(shape_override).float().unsqueeze(0).expand(b, -1).cuda()
            else:
                sh = torch.zeros(b, 45).float().cuda()

            ep = torch.zeros(b, head.num_face_comps).float().cuda()

            # return_keypoints=True to get proper 308 surface keypoints
            # (Bug fix: J_regressor_mhr_70 doesn't exist on MHRHead)
            output = head.mhr_forward(
                global_trans=gt,
                global_rot=gr,
                body_pose_params=bp,
                hand_pose_params=hp,
                scale_params=sp,
                shape_params=sh,
                expr_params=ep,
                return_keypoints=True,
                return_joint_coords=True,
                return_joint_rotations=True,
            )

            # With return_keypoints + return_joint_coords + return_joint_rotations:
            # output = (verts, kp308, jcoords, joint_rots)
            verts, kp308, jcoords, grots = output

            # Save vertices BEFORE Y/Z flip (MHR body-centric convention,
            # matches rest_vertices.npy)
            if return_vertices:
                all_verts.append(verts.cpu().numpy())

            # Bug fix: apply Y/Z flip (MHR body → camera convention)
            # Matches MHRHead.forward lines 340-343
            jcoords[..., [1, 2]] *= -1
            kp308[..., [1, 2]] *= -1

            # Truncate 308 → 70 keypoints (matches MHRHead.forward line 337)
            kp70 = kp308[:, :70]

            all_jcoords.append(jcoords.cpu().numpy())
            all_grots.append(grots.cpu().numpy())
            all_kp3d.append(kp70.cpu().numpy())

    joint_coords = np.concatenate(all_jcoords, axis=0)
    global_rots = np.concatenate(all_grots, axis=0)
    keypoints_3d = np.concatenate(all_kp3d, axis=0)

    if return_vertices:
        vertices = np.concatenate(all_verts, axis=0)
        return joint_coords, global_rots, keypoints_3d, vertices

    return joint_coords, global_rots, keypoints_3d


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_sam3d(checkpoint_path: str, mhr_path: str, use_mask: bool = False):
    """Load SAM 3D Body model + MoGe2 FOV estimator.

    Args:
        checkpoint_path: Path to model.ckpt.
        mhr_path: Path to mhr_model.pt.
        use_mask: If True, try to load SAM3 segmentor for mask conditioning.

    Returns:
        SAM3DBodyEstimator instance.
    """
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from tools.build_fov_estimator import FOVEstimator

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SAM 3D Body model
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=checkpoint_path,
        device=device,
        mhr_path=mhr_path,
    )

    # Load MoGe2 FOV estimator
    print("[sam3d-worker] Loading MoGe2 FOV estimator...")
    fov_estimator = FOVEstimator(name="moge2", device=device)

    # NOTE: vitdet human detector is lazy-loaded in process_video() only when
    # no bounding boxes are provided (fallback mode). This saves ~500MB VRAM
    # and ~10s startup time in the normal YOLOX-boxes path.

    # Optionally load SAM3 segmentor for mask conditioning
    human_segmentor = None
    if use_mask:
        try:
            from tools.build_sam import build_sam
            print("[sam3d-worker] Loading SAM3 segmentor for mask conditioning...")
            human_segmentor = build_sam(name="sam3", path=None)
            print("[sam3d-worker] SAM3 segmentor loaded")
        except Exception as e:
            print(f"[sam3d-worker] WARNING: Could not load SAM3 segmentor: {e}")
            print("[sam3d-worker] Continuing without mask conditioning")

    # Build estimator
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,  # lazy-loaded in process_video() when needed
        fov_estimator=fov_estimator,
        human_segmentor=human_segmentor,
    )

    return estimator


def _extract_result(out: dict, person_idx: int = 0) -> dict:
    """Extract per-person result from SAM 3D output dict (numpy).

    Args:
        out: output["mhr"] after recursive_to(cpu/numpy).
        person_idx: Which person to extract.

    Returns:
        Dict with pred_joint_coords, pred_global_rots, etc.
    """
    return {
        "pred_joint_coords": out["pred_joint_coords"][person_idx],
        "pred_global_rots": out["joint_global_rots"][person_idx],
        "pred_cam_t": out["pred_cam_t"][person_idx],
        "focal_length": out["focal_length"][person_idx],
        "mhr_model_params": out["mhr_model_params"][person_idx],
        "pred_keypoints_3d": out["pred_keypoints_3d"][person_idx],
        "shape_params": out["shape"][person_idx],
        # Raw PCA params for correct MHR forward re-run
        "scale_params": out["scale"][person_idx],        # (28,) raw PCA coefficients
        "hand_pose_params": out["hand"][person_idx],     # (H,) raw PCA hand
        "body_pose_params": out["body_pose"][person_idx],    # (130,) euler angles
        "global_rot": out["global_rot"][person_idx],     # (3,) ZYX euler
    }


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(
    estimator,
    video_path: str,
    boxes: np.ndarray | None,
    output_dir: Path,
    shape_stabilize: bool = True,
    temporal_smooth: bool = True,
    use_mask: bool = False,
    kalman_q_pos: float = 0.01,
    kalman_q_vel: float = 0.001,
    ema_alpha_static: float = 0.10,
    marker_indices: np.ndarray | None = None,
):
    """Process video frames through SAM 3D Body with post-processing.

    Pipeline:
      1. Per-frame inference (body-only)
      2. Shape stabilization (median shape/scale → MHR forward re-run)
      3. Temporal smoothing (Kalman on body pose, quaternion EMA on global rot)
      4. Optional: extract surface marker positions from mesh vertices

    Args:
        estimator: SAM3DBodyEstimator instance.
        video_path: Path to input video.
        boxes: (N, 4) bounding boxes in xyxy pixel coords, or None to use
            vitdet per-frame detection.
        output_dir: Directory to save .npy outputs.
        shape_stabilize: Freeze shape/scale to median across all frames.
        temporal_smooth: Apply Kalman + quaternion smoother post-processing.
        use_mask: Use mask conditioning (requires SAM3 segmentor loaded).
        kalman_q_pos: Kalman process noise for position.
        kalman_q_vel: Kalman process noise for velocity.
        ema_alpha_static: EMA alpha for static global rotation segments.
        marker_indices: (M,) int array of MHR mesh vertex indices for marker
            extraction. If provided, saves marker_positions.npy (N, M, 3).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Per-frame output arrays
    all_joint_coords = []     # (N, 127, 3) camera space, meters
    all_global_rots = []      # (N, 127, 3, 3) MHR body-centric
    all_cam_t = []            # (N, 3) camera translation
    all_focal_length = []     # (N,) focal length
    all_model_params = []     # (N, 204) raw MHR model parameters
    all_keypoints_3d = []     # (N, 70, 3) surface keypoints
    all_shape_params = []     # (N, 45) identity coefficients
    # Raw PCA params (for correct MHR forward re-run)
    all_scale_pca = []        # (N, 28) raw PCA scale coefficients
    all_hand_pca = []         # (N, H) raw PCA hand coefficients
    all_body_pose = []        # (N, 130) euler body pose
    all_global_rot = []       # (N, 3) ZYX euler global rotation

    # Compute camera intrinsics once from first frame (MoGe2 FOV estimation)
    # Camera FOV doesn't change between video frames, so compute once and reuse.
    ret, first_frame = cap.read()
    if not ret:
        print("[sam3d-worker] ERROR: Cannot read first frame", file=sys.stderr)
        return
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    if estimator.fov_estimator is not None:
        print("[sam3d-worker] Computing FOV from first frame (MoGe2)...")
        cam_int = estimator.fov_estimator.get_cam_intrinsics(first_frame_rgb)
        cam_int = cam_int.to(torch.float32).cuda()
        print(f"[sam3d-worker] Camera intrinsics cached (focal={cam_int[0, 0, 0]:.1f})")
    else:
        cam_int = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to start

    # Lazy-load vitdet when no boxes provided (fallback mode)
    if boxes is None and estimator.human_detector is None:
        try:
            from tools.build_detector import HumanDetector
            device = next(estimator.sam_3d_body_model.parameters()).device
            print("[sam3d-worker] Loading vitdet human detector (fallback)...")
            estimator.human_detector = HumanDetector(name="vitdet", device=device)
            print("[sam3d-worker] vitdet detector loaded")
        except Exception as e:
            print(f"[sam3d-worker] WARNING: Could not load vitdet: {e}")
            print("[sam3d-worker] Using full-frame bounding boxes instead")

    print(f"[sam3d-worker] Processing {n_frames} frames at {fps:.1f} fps")
    if use_mask and estimator.human_segmentor is not None:
        print("[sam3d-worker] Mask conditioning ENABLED")
    t0 = time.perf_counter()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB (SAM 3D expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get bbox for this frame
        # If boxes provided (YOLOX), use them. Otherwise let vitdet detect.
        if boxes is not None:
            if frame_idx < len(boxes):
                bbox = boxes[frame_idx:frame_idx + 1]  # (1, 4)
            else:
                h, w = frame.shape[:2]
                bbox = np.array([[0, 0, w, h]])
        else:
            bbox = None  # vitdet will detect per-frame

        # Run SAM 3D Body inference
        # bboxes=None triggers vitdet detection inside process_one_image
        # Suppress verbose per-frame prints from SAM 3D internals
        with open(os.devnull, "w") as _devnull, \
             contextlib.redirect_stdout(_devnull):
            results = estimator.process_one_image(
                frame_rgb,
                bboxes=bbox,
                cam_int=cam_int,
                inference_type="body",
                use_mask=use_mask and estimator.human_segmentor is not None,
            )
        result = results[0] if len(results) > 0 else None

        if result is None:
            # No detection — fill with zeros
            print(f"  [frame {frame_idx}] No detection, using zeros")
            all_joint_coords.append(np.zeros((127, 3), dtype=np.float32))
            all_global_rots.append(np.eye(3, dtype=np.float32)[None].repeat(127, axis=0))
            all_cam_t.append(np.zeros(3, dtype=np.float32))
            all_focal_length.append(500.0)
            all_model_params.append(np.zeros(204, dtype=np.float32))
            all_keypoints_3d.append(np.zeros((70, 3), dtype=np.float32))
            all_shape_params.append(np.zeros(45, dtype=np.float32))
            all_scale_pca.append(np.zeros(28, dtype=np.float32))
            all_hand_pca.append(np.zeros(108, dtype=np.float32))  # 54*2 hand PCA
            all_body_pose.append(np.zeros(130, dtype=np.float32))
            all_global_rot.append(np.zeros(3, dtype=np.float32))
        else:
            all_joint_coords.append(result["pred_joint_coords"].astype(np.float32))
            all_global_rots.append(result["pred_global_rots"].astype(np.float32))
            all_cam_t.append(result["pred_cam_t"].astype(np.float32))
            all_focal_length.append(float(result["focal_length"]))
            all_model_params.append(result["mhr_model_params"].astype(np.float32))
            all_keypoints_3d.append(result["pred_keypoints_3d"].astype(np.float32))
            all_shape_params.append(result["shape_params"].astype(np.float32))
            all_scale_pca.append(result["scale_params"].astype(np.float32))
            all_hand_pca.append(result["hand_pose_params"].astype(np.float32))
            all_body_pose.append(result["body_pose_params"].astype(np.float32))
            all_global_rot.append(result["global_rot"].astype(np.float32))

        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = frame_idx / elapsed
            print(f"  [frame {frame_idx}/{n_frames}] {fps_proc:.1f} fps")

    cap.release()
    elapsed = time.perf_counter() - t0
    print(f"[sam3d-worker] Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps)")

    # Stack into arrays
    model_params = np.stack(all_model_params, axis=0)      # (N, 204)
    shape_params_arr = np.stack(all_shape_params, axis=0)   # (N, 45)
    joint_coords = np.stack(all_joint_coords, axis=0)       # (N, 127, 3)
    global_rots = np.stack(all_global_rots, axis=0)         # (N, 127, 3, 3)
    keypoints_3d = np.stack(all_keypoints_3d, axis=0)       # (N, 70, 3)
    # Raw PCA arrays for correct MHR forward re-run
    scale_pca_arr = np.stack(all_scale_pca, axis=0)         # (N, 28)
    body_pose_arr = np.stack(all_body_pose, axis=0)         # (N, 130)
    global_rot_arr = np.stack(all_global_rot, axis=0)       # (N, 3)
    # Hand PCA: may vary in dim if zero-detection frames have placeholder size
    hand_dim = max(h.shape[0] for h in all_hand_pca)
    hand_pca_arr = np.zeros((frame_idx, hand_dim), dtype=np.float32)
    for i, h in enumerate(all_hand_pca):
        hand_pca_arr[i, :h.shape[0]] = h

    # ── Post-processing pipeline ──────────────────────────────────────
    needs_rerun = False
    shape_override = None

    # Detect zero-detection frames (all-zero body pose = no detection)
    zero_mask = np.all(np.abs(body_pose_arr) < 1e-6, axis=1)
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        print(f"[sam3d-worker] {n_zero} zero-detection frames detected")

    # 1. Shape/scale stabilization: median across VALID frames only
    if shape_stabilize and frame_idx > 1:
        print("[sam3d-worker] Shape stabilization: computing median shape/scale...")
        valid_mask = ~zero_mask
        if valid_mask.any():
            shape_override = np.median(shape_params_arr[valid_mask], axis=0).astype(np.float32)
            median_scale_pca = np.median(scale_pca_arr[valid_mask], axis=0)
        else:
            shape_override = np.median(shape_params_arr, axis=0).astype(np.float32)
            median_scale_pca = np.median(scale_pca_arr, axis=0)
        shape_var = np.std(shape_params_arr[valid_mask] if valid_mask.any()
                          else shape_params_arr, axis=0).mean()
        print(f"  Shape param std (mean across dims): {shape_var:.4f}")
        scale_pca_arr[:] = median_scale_pca
        print(f"  Scale PCA stabilized ({scale_pca_arr.shape[1]} dims)")
        needs_rerun = True

    # 2. Temporal smoothing on raw PCA params
    if temporal_smooth and frame_idx > 3:
        print("[sam3d-worker] Temporal smoothing...")
        t_smooth = time.perf_counter()

        # Interpolate zero-detection frames before Kalman
        # (zeros drag valid poses toward neutral standing)
        if n_zero > 0:
            valid_idx = np.where(~zero_mask)[0]
            invalid_idx = np.where(zero_mask)[0]
            if len(valid_idx) >= 2:
                for d in range(body_pose_arr.shape[1]):
                    body_pose_arr[invalid_idx, d] = np.interp(
                        invalid_idx, valid_idx, body_pose_arr[valid_idx, d]
                    )
                print(f"  Interpolated {len(invalid_idx)} zero-detection body pose frames")

        # 2a. Kalman on body pose (130-dim euler angles)
        body_pose_raw = body_pose_arr.copy()
        body_pose_arr = _kalman_smooth(body_pose_raw, q_pos=kalman_q_pos, q_vel=kalman_q_vel)
        pose_diff = np.sqrt(((body_pose_arr - body_pose_raw) ** 2).sum(axis=1)).mean()
        print(f"  Kalman body pose: mean change {pose_diff:.4f} rad")

        # 2b. Quaternion smoother on global rotation (3-dim ZYX euler)
        # Operates in quaternion space with outlier rejection + SLERP interpolation
        global_rot_raw = global_rot_arr.copy()
        global_rot_arr = _smooth_global_rot_quat(global_rot_raw,
                                                  outlier_threshold_deg=60.0,
                                                  alpha=0.5)
        rot_diff = np.sqrt(((global_rot_arr - global_rot_raw) ** 2).sum(axis=1)).mean()
        print(f"  Quaternion global rot: mean change {rot_diff:.4f} rad")
        print(f"  Smoothing took {time.perf_counter() - t_smooth:.2f}s")
        needs_rerun = True

    # 3. Re-run MHR forward if anything changed
    # Also request vertices when marker extraction is needed
    need_vertices = marker_indices is not None
    all_vertices = None

    if needs_rerun:
        print("[sam3d-worker] Re-running MHR forward with stabilized/smoothed params...")
        t_rerun = time.perf_counter()
        rerun_result = _rerun_mhr_forward(
            estimator, body_pose_arr, global_rot_arr,
            scale_pca_arr, hand_pca_arr, shape_override,
            return_vertices=need_vertices,
        )
        if need_vertices:
            joint_coords, global_rots, keypoints_3d, all_vertices = rerun_result
        else:
            joint_coords, global_rots, keypoints_3d = rerun_result
        print(f"  MHR forward re-run: {time.perf_counter() - t_rerun:.1f}s "
              f"({frame_idx / max(time.perf_counter() - t_rerun, 0.01):.0f} fps)")

        # Update shape_params_arr to reflect stabilized shape
        if shape_override is not None:
            shape_params_arr[:] = shape_override

    elif need_vertices:
        # No post-processing rerun, but we still need vertices for markers.
        # Do a forward pass with original params just to get mesh vertices.
        print("[sam3d-worker] Running MHR forward for vertex extraction...")
        t_verts = time.perf_counter()
        _, _, _, all_vertices = _rerun_mhr_forward(
            estimator, body_pose_arr, global_rot_arr,
            scale_pca_arr, hand_pca_arr,
            shape_override if shape_override is not None else None,
            return_vertices=True,
        )
        print(f"  Vertex extraction: {time.perf_counter() - t_verts:.1f}s")

    # ── Save outputs ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "joint_coords.npy", joint_coords)
    np.save(output_dir / "global_rots.npy", global_rots)
    np.save(output_dir / "cam_t.npy", np.stack(all_cam_t, axis=0))
    np.save(output_dir / "focal_length.npy",
            np.array(all_focal_length, dtype=np.float32))
    np.save(output_dir / "model_params.npy", model_params)
    np.save(output_dir / "keypoints_3d.npy", keypoints_3d)
    np.save(output_dir / "shape_params.npy", shape_params_arr)

    # Save metadata
    np.savez(output_dir / "metadata.npz",
             fps=fps,
             n_frames=frame_idx,
             video_path=video_path)

    # Extract and save surface marker positions from mesh vertices
    # Vertices are in MHR body-centric coords (X=right, Y=up, Z=backward),
    # matching rest_vertices.npy convention. NO Y/Z flip applied.
    if marker_indices is not None and all_vertices is not None:
        marker_positions = all_vertices[:, marker_indices]  # (N, M, 3)
        np.save(output_dir / "marker_positions.npy",
                marker_positions.astype(np.float32))
        print(f"[sam3d-worker] Marker positions: {marker_positions.shape} "
              f"({len(marker_indices)} markers × {frame_idx} frames)")

    print(f"[sam3d-worker] Saved outputs to {output_dir}")

    # Compute and save rest pose (uses stabilized shape/scale if available)
    rest_shape = shape_override if shape_override is not None else all_shape_params[0]
    # scale_pca_arr is already stabilized (filled with median) if shape_stabilize was on
    rest_scale = scale_pca_arr[0] if frame_idx > 0 else None
    _compute_rest_pose(estimator, output_dir, rest_shape, rest_scale)


def _compute_rest_pose(
    estimator,
    output_dir: Path,
    shape_params: np.ndarray,
    scale_pca: np.ndarray | None = None,
):
    """Compute rest-pose geometry with zero pose parameters.

    Uses the provided shape params (stabilized median or first frame)
    and scale PCA (for per-subject proportions).

    Saves:
      - rest_global_rots.npy: (127, 3, 3) rest-pose global rotations
      - rest_joint_coords.npy: (127, 3) rest-pose joint positions
      - rest_vertices.npy: (V, 3) rest-pose mesh vertices
      - rest_faces.npy: (F, 3) MHR face connectivity (triangle indices)
    """
    print("[sam3d-worker] Computing rest pose...")

    try:
        head = estimator.model.head_pose

        with torch.no_grad():
            shape = torch.from_numpy(shape_params).float().unsqueeze(0).cuda()
            # Zero global translation/rotation + zero body pose
            global_trans = torch.zeros(1, 3).cuda()
            global_rot = torch.zeros(1, 3).cuda()
            body_pose = torch.zeros(1, 130).cuda()
            if scale_pca is not None:
                sp = torch.from_numpy(scale_pca).float().unsqueeze(0).cuda()
            else:
                sp = torch.zeros(1, head.num_scale_comps).float().cuda()
            expr_params = torch.zeros(1, head.num_face_comps).cuda()

            output = head.mhr_forward(
                global_trans=global_trans,
                global_rot=global_rot,
                body_pose_params=body_pose,
                hand_pose_params=None,
                scale_params=sp,
                shape_params=shape,
                expr_params=expr_params,
                return_keypoints=True,
                return_joint_coords=True,
                return_joint_rotations=True,
            )

            # With return_keypoints + return_joint_coords + return_joint_rotations:
            # output = (verts, kp308, joint_coords, joint_rots)
            rest_verts, _kp308, rest_jcoords, rest_rots = output

            rest_rots_np = rest_rots.cpu().numpy()[0]      # (127, 3, 3)
            rest_jcoords_np = rest_jcoords.cpu().numpy()[0] # (127, 3)
            rest_verts_np = rest_verts.cpu().numpy()[0]     # (V, 3)

            np.save(output_dir / "rest_global_rots.npy", rest_rots_np.astype(np.float32))
            np.save(output_dir / "rest_joint_coords.npy", rest_jcoords_np.astype(np.float32))
            np.save(output_dir / "rest_vertices.npy", rest_verts_np.astype(np.float32))
            print(f"[sam3d-worker] Rest pose saved: rots {rest_rots_np.shape}, "
                  f"joints {rest_jcoords_np.shape}, verts {rest_verts_np.shape}")

            # Extract and save face connectivity from MHR model
            _save_mhr_faces(head, output_dir)

    except Exception as e:
        print(f"[sam3d-worker] Warning: could not compute rest pose: {e}")
        import traceback; traceback.print_exc()
        print("[sam3d-worker] Retargeting will work without rest pose (may have offset)")


def _save_mhr_faces(head, output_dir: Path):
    """Extract and save MHR face connectivity (triangle indices).

    MHR faces live at head.mhr.character_torch.mesh.faces (TorchScript module).
    """
    try:
        faces = head.mhr.character_torch.mesh.faces  # (F, 3) int32
        if torch.is_tensor(faces):
            faces_np = faces.cpu().numpy()
        else:
            faces_np = np.asarray(faces)

        np.save(output_dir / "rest_faces.npy", faces_np.astype(np.int32))
        print(f"[sam3d-worker] MHR faces saved: {faces_np.shape}")
    except Exception as e:
        print(f"[sam3d-worker] Warning: could not save MHR faces: {e}")


def main():
    parser = argparse.ArgumentParser(description="SAM 3D Body worker")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--boxes", default=None,
                        help="Path to bounding boxes .npy (N,4). "
                             "If omitted, vitdet detects per-frame.")
    parser.add_argument("--output-dir", required=True, help="Directory for output .npy files")
    parser.add_argument("--checkpoint-path", required=True,
                        help="Path to SAM 3D Body model.ckpt")
    parser.add_argument("--mhr-path", required=True,
                        help="Path to MHR body model (mhr_model.pt)")
    parser.add_argument("--no-shape-stabilize", action="store_true",
                        help="Disable shape parameter stabilization")
    parser.add_argument("--no-temporal-smooth", action="store_true",
                        help="Disable temporal smoothing (Kalman + EMA)")
    parser.add_argument("--use-mask", action="store_true",
                        help="Enable mask conditioning via SAM3 segmentor")
    parser.add_argument("--kalman-q-pos", type=float, default=0.01,
                        help="Kalman process noise for position")
    parser.add_argument("--kalman-q-vel", type=float, default=0.001,
                        help="Kalman process noise for velocity")
    parser.add_argument("--ema-alpha-static", type=float, default=0.10,
                        help="EMA alpha for static global rotation segments")
    parser.add_argument("--marker-indices", type=str, default=None,
                        help="Path to .npy file with MHR vertex indices (M,) "
                             "for per-frame surface marker extraction")
    args = parser.parse_args()

    # Load bounding boxes (optional — vitdet detects if not provided)
    boxes = None
    if args.boxes is not None:
        boxes = np.load(args.boxes)
        print(f"[sam3d-worker] Loaded {len(boxes)} bounding boxes from {args.boxes}")
    else:
        print("[sam3d-worker] No bounding boxes provided — vitdet will detect per-frame")

    # Load marker indices for surface marker extraction (optional)
    marker_indices = None
    if args.marker_indices is not None:
        marker_indices = np.load(args.marker_indices)
        print(f"[sam3d-worker] Loaded {len(marker_indices)} marker indices "
              f"from {args.marker_indices}")

    # Setup model + FOV estimator (+ optional SAM3 segmentor)
    print("[sam3d-worker] Loading SAM 3D Body model...")
    estimator = setup_sam3d(args.checkpoint_path, args.mhr_path,
                            use_mask=args.use_mask)
    print("[sam3d-worker] Model loaded")

    # Process video with post-processing
    process_video(
        estimator,
        args.video,
        boxes,
        Path(args.output_dir),
        shape_stabilize=not args.no_shape_stabilize,
        temporal_smooth=not args.no_temporal_smooth,
        use_mask=args.use_mask,
        kalman_q_pos=args.kalman_q_pos,
        kalman_q_vel=args.kalman_q_vel,
        ema_alpha_static=args.ema_alpha_static,
        marker_indices=marker_indices,
    )


if __name__ == "__main__":
    main()
