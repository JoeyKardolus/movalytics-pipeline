#!/usr/bin/env python3
"""Render SAM 3D mesh overlay on video using the existing worker's MHR forward.

Uses the sam3d_worker's _rerun_mhr_forward to reconstruct per-frame vertices,
then renders using SAM 3D's built-in pyrender Renderer.

Usage:
    conda run -n sam3d python scripts/viz/sam3d_mesh_viewer.py \
        --npz data/output/joey/joey_sam3d.npz \
        --video data/input/joey.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add vendored sam-3d-body library to path
SAM3D_DIR = Path(__file__).resolve().parents[2] / "lib" / "sam-3d-body"
sys.path.insert(0, str(SAM3D_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="SAM 3D output NPZ")
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--output", default=None, help="Output MP4 (default: auto)")
    parser.add_argument("--max-frames", type=int, default=0, help="0=all")
    args = parser.parse_args()

    import os
    os.chdir(str(SAM3D_DIR))

    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from sam_3d_body.visualization.renderer import Renderer

    # Load NPZ
    d = np.load(args.npz, allow_pickle=True)
    model_params = d["model_params"]  # (N, 204)
    cam_t = d["cam_t"]  # (N, 3)
    fl = d["focal_length"]
    focal_length = float(fl[0]) if fl.ndim > 0 else float(fl)
    rest_faces = d["rest_faces"]  # (F, 3)
    shape_params = d["shape_params"] if "shape_params" in d else None
    n_frames = int(d["n_frames"])

    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)

    # Load SAM 3D model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(__file__).resolve().parents[2]  # humanpose3d_backend root
    checkpoint = str(base / "models" / "sam-3d-body-dinov3" / "model.ckpt")
    mhr_path = str(base / "models" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt")

    print("[mesh-viewer] Loading SAM 3D model...")
    model, model_cfg = load_sam_3d_body(checkpoint, device=device, mhr_path=mhr_path)

    # Use the worker's proven MHR forward function
    from src.workers.sam3d_worker import _rerun_mhr_forward

    # Parse params for _rerun_mhr_forward
    body_pose = model_params[:n_frames, :126]
    global_rot = model_params[:n_frames, 126:132]
    # scale_pca from shape_params[:, 17:45] (28 dims)
    if shape_params is not None and shape_params.shape[1] >= 45:
        scale_pca = shape_params[:n_frames, 17:45]
    else:
        scale_pca = np.zeros((n_frames, 28), dtype=np.float32)
    hand_pca = np.zeros((n_frames, 24), dtype=np.float32)
    # shape override (betas 4-dim)
    shape_override = model_params[0, 132:136]

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
    )

    print(f"[mesh-viewer] Reconstructing {n_frames} frames...")
    t0 = time.perf_counter()
    _, _, _, all_vertices = _rerun_mhr_forward(
        estimator, body_pose, global_rot,
        scale_pca, hand_pca, shape_override,
        return_vertices=True,
    )
    dt = time.perf_counter() - t0
    print(f"[mesh-viewer] MHR forward: {n_frames} frames in {dt:.1f}s ({n_frames/dt:.0f} fps)")
    print(f"[mesh-viewer] Vertices shape: {all_vertices.shape}")

    # Render mesh overlay
    renderer = Renderer(focal_length=focal_length, faces=rest_faces)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    output_path = args.output or str(Path(args.npz).with_suffix("")) + "_mesh.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    print(f"[mesh-viewer] Rendering {n_frames} frames → {output_path}")
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        mesh_img = (
            renderer(
                all_vertices[i],
                cam_t[i],
                frame.copy(),
                mesh_base_color=(0.65, 0.74, 0.86),
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        combined = np.concatenate([frame, mesh_img], axis=1)
        out.write(combined)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_frames}]")

    cap.release()
    out.release()
    print(f"[mesh-viewer] Done → {output_path}")
    print(f"[mesh-viewer] Play: ffplay -autoexit {output_path}")


if __name__ == "__main__":
    main()
