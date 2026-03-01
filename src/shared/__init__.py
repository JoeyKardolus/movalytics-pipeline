"""Shared utilities for HumanPose3D pipeline.

This module contains reusable components:
- constants: COCO skeleton definitions and visualization colors
- coordinate_transforms: Pipeline/Camera coordinate conversions
- filtering: Temporal signal filtering (Butterworth, median, moving average)
"""

from .constants import (
    COCO_SKELETON_CONNECTIONS,
    COCO_JOINT_NAMES,
    JOINT_COLORS,
    LIMB_COLORS,
    # MHR body model
    MHR_JOINT_INDICES,
    MHR_JOINT_NAMES,
    MHR_SKELETON_EDGES,
    MHR_JOINT_CENTERS,
    MHR70_KEYPOINTS,
)

from .coordinate_transforms import (
    camera_to_pipeline,
    pipeline_to_camera,
    MHR_BODY_TO_PIPELINE,
    CAMERA_TO_PIPELINE,
)

from .filtering import (
    butterworth_lowpass,
    median_filter_1d,
    moving_average,
)

__all__ = [
    # Constants — COCO
    "COCO_SKELETON_CONNECTIONS",
    "COCO_JOINT_NAMES",
    "JOINT_COLORS",
    "LIMB_COLORS",
    # Constants — MHR
    "MHR_JOINT_INDICES",
    "MHR_JOINT_NAMES",
    "MHR_SKELETON_EDGES",
    "MHR_JOINT_CENTERS",
    "MHR70_KEYPOINTS",
    # Coordinate transforms
    "camera_to_pipeline",
    "pipeline_to_camera",
    "MHR_BODY_TO_PIPELINE",
    "CAMERA_TO_PIPELINE",
    # Filtering
    "butterworth_lowpass",
    "median_filter_1d",
    "moving_average",
]
