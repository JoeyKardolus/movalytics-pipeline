from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

OUTPUT_ROOT = Path("data/output")


class _TrackingAction(argparse.Action):
    """Custom action that tracks which args the user explicitly set."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        explicitly_set = getattr(namespace, "_explicitly_set", set())
        explicitly_set.add(self.dest)
        namespace._explicitly_set = explicitly_set


class _TrackingStoreTrueAction(argparse.Action):
    """store_true that tracks explicit usage."""

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(option_strings, dest, nargs=0, const=True,
                         default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        explicitly_set = getattr(namespace, "_explicitly_set", set())
        explicitly_set.add(self.dest)
        namespace._explicitly_set = explicitly_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D human pose estimation: YOLOX detection → SAM 3D Body → joint angles."
    )
    # Required
    parser.add_argument(
        "--video",
        required="--dump-config" not in sys.argv,
        help="Input video file",
    )

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (all params; CLI flags override config values)")
    parser.add_argument("--dump-config", action="store_true",
                        help="Print full default config as YAML and exit")

    # Subject parameters
    parser.add_argument("--height", type=float, default=1.78, action=_TrackingAction,
                        help="Subject height in meters")
    parser.add_argument("--mass", type=float, default=75.0, action=_TrackingAction,
                        help="Subject mass in kg")

    # Detection
    parser.add_argument("--visibility-min", type=float, default=0.3,
                        action=_TrackingAction,
                        help="Minimum landmark visibility threshold (default 0.3)")

    # Visualization
    parser.add_argument("--plot-joint-angles", action=_TrackingStoreTrueAction,
                        help="Generate joint angle visualization PNGs")
    parser.add_argument("--save-angle-comparison", action=_TrackingStoreTrueAction,
                        help="Save side-by-side right vs left comparison plot")

    # Movement analysis
    parser.add_argument("--movement-analysis", action=_TrackingStoreTrueAction,
                        help="Analyze movement patterns against normative data")

    # Post-processing
    parser.add_argument("--temporal-smoothing", type=int, default=0, metavar="WINDOW",
                        action=_TrackingAction,
                        help="Apply temporal smoothing with given window size (0=disabled)")

    args = parser.parse_args()

    # Handle --dump-config before validation
    if args.dump_config:
        from src.core.config import dump_default_config
        print(dump_default_config())
        sys.exit(0)

    # Initialize _explicitly_set if no tracking actions fired
    if not hasattr(args, "_explicitly_set"):
        args._explicitly_set = set()

    return args


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[main] video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Load config (YAML file if provided, then CLI overrides)
    from src.core.config import load_config, apply_cli_overrides
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Import heavy dependencies only when actually running
    from src.core.video.media_stream import apply_rotation, probe_video_rotation, read_video_rgb
    from src.core.pipeline.cleanup import cleanup_output_directory

    run_dir = OUTPUT_ROOT / video_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import time as _time
        _timings = {}

        # Step 1: Load video
        _t0 = _time.perf_counter()
        frames, fps = read_video_rgb(video_path)
        video_rotation = probe_video_rotation(video_path, decoded_shape=frames.shape[1:])
        if video_rotation != 0:
            print(f"[main] Correcting video rotation: {video_rotation}°")
            frames = apply_rotation(frames, video_rotation)

        video_width, video_height = frames.shape[2], frames.shape[1]
        _timings["1_load_video"] = _time.perf_counter() - _t0

        # Step 2: YOLOX bounding box detection
        _t0 = _time.perf_counter()
        from src.core.detection.bbox_detector import BBoxDetector
        import cv2

        det_frames = frames
        det_scale = 1.0
        max_det_dim = 1920
        if max(video_height, video_width) > max_det_dim:
            det_scale = max_det_dim / max(video_height, video_width)
            det_h = int(video_height * det_scale)
            det_w = int(video_width * det_scale)
            det_frames = np.stack([
                cv2.resize(f, (det_w, det_h), interpolation=cv2.INTER_AREA)
                for f in frames
            ])
            print(f"[main] Downscaled {video_width}x{video_height} → {det_w}x{det_h} for detection")

        print(f"[main] Using YOLOX bbox detection (model={cfg.detection.model_size})")
        detector = BBoxDetector(
            model_size=cfg.detection.model_size,
            smoothing_min_cutoff=cfg.detection.smoothing.min_cutoff,
            smoothing_beta=cfg.detection.smoothing.beta,
            smoothing_d_cutoff=cfg.detection.smoothing.d_cutoff,
            tracking_max_age=cfg.detection.tracking.max_age,
            tracking_min_iou=cfg.detection.tracking.min_iou,
            tracking_confirm_hits=cfg.detection.tracking.confirm_hits,
        )
        result = detector.detect(det_frames, fps, cfg.detection.visibility_min)
        del det_frames  # free downscaled copy
        boxes_xyxy = result.metadata.get("bounding_boxes")

        # Scale boxes back to original resolution
        if det_scale != 1.0 and boxes_xyxy is not None:
            boxes_xyxy = boxes_xyxy / det_scale
        if boxes_xyxy is None:
            print("[main] ERROR: No bounding boxes in detection result", file=sys.stderr)
            sys.exit(1)

        # Free detector GPU memory
        del detector
        import torch; torch.cuda.empty_cache()
        _timings["2_yolox_detection"] = _time.perf_counter() - _t0

        # Step 3: SAM 3D Body lifting
        sam3d_result = _run_sam3d(cfg, video_path, boxes_xyxy, fps, _timings)
        del frames

        if not sam3d_result.success:
            print(f"[main] ERROR: sam3d failed - {sam3d_result.error}",
                  file=sys.stderr)
            sys.exit(1)

        # Save raw SAM 3D output (MHR body model data)
        from src.core.conversion.sam3d_joint_map import MHR, MHR_SKELETON_EDGES
        sam3d_output = run_dir / f"{video_path.stem}_sam3d.npz"
        save_kwargs = dict(
            joint_coords=sam3d_result.joint_coords,
            global_rots=sam3d_result.global_rots,
            cam_t=sam3d_result.cam_t,
            model_params=sam3d_result.model_params,
            shape_params=sam3d_result.shape_params,
            keypoints_3d=sam3d_result.keypoints_3d,
            focal_length=sam3d_result.focal_length,
            rest_global_rots=sam3d_result.rest_global_rots,
            fps=sam3d_result.fps,
            n_frames=sam3d_result.n_frames,
            joint_names=list(MHR.keys()),
            joint_indices=list(MHR.values()),
            skeleton_edges=np.array(MHR_SKELETON_EDGES),
        )
        # Include rest-pose mesh data if available (for atlas building)
        if sam3d_result.rest_joint_coords.size > 0:
            save_kwargs["rest_joint_coords"] = sam3d_result.rest_joint_coords
        if sam3d_result.rest_vertices.size > 0:
            save_kwargs["rest_vertices"] = sam3d_result.rest_vertices
        if sam3d_result.rest_faces.size > 0:
            save_kwargs["rest_faces"] = sam3d_result.rest_faces
        np.savez(sam3d_output, **save_kwargs)
        print(f"[main] SAM 3D raw output -> {sam3d_output}")

        # Step 4: Extract 30 clinical DOFs from MHR rotations
        from src.core.conversion.sam3d_clinical_angles import (
            extract_sam3d_clinical_angles,
        )
        from src.core.kinematics.angle_export import (
            save_comprehensive_angles_csv,
        )
        from src.core.conversion.sam3d_visualization import (
            plot_sam3d_clinical_angles,
        )

        sam3d_angles = extract_sam3d_clinical_angles(
            sam3d_result.global_rots,
            sam3d_result.rest_global_rots,
            fps,
            calibration_frames=cfg.lifting.sam3d.calibration_frames,
        )

        # Step 5: OpenSim IK from MHR 43-marker clinical set
        from src.core.lifting.opensim_ik import run_opensim_ik

        if sam3d_result.marker_positions.size > 0:
            # 43-marker clinical set (41 vertex + 2 HJC from MHR joints)
            from src.core.conversion.mhr_markers_to_trc import mhr_markers_to_trc

            trc_path = mhr_markers_to_trc(
                marker_positions=sam3d_result.marker_positions,
                marker_names=sam3d_result.marker_names,
                subject_height=cfg.subject.height,
                fps=fps,
                output_path=run_dir / f"{video_path.stem}_mhr_markers.trc",
                rest_vertices=sam3d_result.rest_vertices
                if sam3d_result.rest_vertices.size > 0 else None,
                joint_coords=sam3d_result.joint_coords
                if sam3d_result.joint_coords.size > 0 else None,
                keypoints_3d=sam3d_result.keypoints_3d
                if sam3d_result.keypoints_3d.size > 0 else None,
                cam_t=sam3d_result.cam_t
                if sam3d_result.cam_t.size > 0 else None,
            )
            print(f"[main] MHR 76-marker TRC -> {trc_path}")
        else:
            print("[main] WARNING: No marker atlas found. Run the marker picker "
                  "and symmetry enforcer first (see scripts/build_mhr_atlas.py).",
                  file=sys.stderr)
            # Skip OpenSim IK — no markers available
            trc_path = None

        if trc_path is not None:
            mot_path = run_opensim_ik(
                trc_path, run_dir,
                subject_height=cfg.subject.height,
                subject_mass=cfg.subject.mass,
                sam3d_npz=sam3d_output,
                skip_fk=cfg.lifting.opensim.skip_fk,
            )
            print(f"[main] OpenSim IK (LaiUhlrich2022) -> {mot_path}")
            print(f"[main] View with: conda run -n opensim python "
                  f"scripts/viz/opensim_mot_viewer.py --mot {mot_path}")

        # Step 6: Save joint angles
        angles_dir = run_dir / "joint_angles"
        save_comprehensive_angles_csv(
            sam3d_angles,
            output_dir=angles_dir,
            basename=video_path.stem,
        )

        # Always generate clinical angle visualization
        plot_sam3d_clinical_angles(
            sam3d_angles,
            output_path=run_dir / f"{video_path.stem}_sam3d_angles.png",
            title_prefix=video_path.stem,
        )

        # Cleanup and organize
        cleanup_output_directory(run_dir, video_path.stem)

        # Pipeline timing summary
        _total = sum(_timings.values())
        print(f"\n{'─' * 50}")
        print(f"  Pipeline Timing ({_total:.1f}s total)")
        print(f"{'─' * 50}")
        for label, secs in _timings.items():
            pct = 100 * secs / _total if _total > 0 else 0
            bar = "█" * int(pct / 2.5)
            print(f"  {label:25s} {secs:6.1f}s  {pct:4.1f}%  {bar}")
        print(f"{'─' * 50}")
        print(f"[main] finished pipeline. Output: {run_dir}")

    except Exception as exc:
        print(f"[main] error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# SAM 3D Body helper
# ---------------------------------------------------------------------------

def _run_sam3d(cfg, video_path, boxes_xyxy, fps, _timings):
    """SAM 3D Body → raw MHR body model output."""
    import time as _time

    _t0 = _time.perf_counter()
    from src.core.lifting.sam3d_lifter import SAM3DLifter

    print("\n" + "=" * 60)
    print("Running SAM 3D Body")
    print("=" * 60)

    lifter = SAM3DLifter(
        conda_env=cfg.lifting.sam3d.conda_env,
        checkpoint_path=cfg.lifting.sam3d.checkpoint_path,
        mhr_path=cfg.lifting.sam3d.mhr_path,
        shape_stabilize=cfg.lifting.sam3d.shape_stabilize,
        temporal_smooth=cfg.lifting.sam3d.temporal_smooth,
        use_mask=cfg.lifting.sam3d.use_mask,
        kalman_q_pos=cfg.lifting.sam3d.kalman_q_pos,
        kalman_q_vel=cfg.lifting.sam3d.kalman_q_vel,
        ema_alpha_static=cfg.lifting.sam3d.ema_alpha_static,
    )
    # Pass YOLOX boxes for fast detection (~6fps). Quaternion-space smoother
    # in sam3d_worker handles occasional orientation flips. vitdet lazy-loads
    # as fallback only when no boxes are provided.
    result = lifter.estimate(video_path, boxes_xyxy, fps)
    _timings["3_sam3d_lifting"] = _time.perf_counter() - _t0

    return result


if __name__ == "__main__":
    main()
