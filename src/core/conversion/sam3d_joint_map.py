"""MHR body model constants and skeleton topology.

Re-exports from src.shared.constants for backward compatibility.
Canonical definitions live in src/shared/constants.py.
"""

from __future__ import annotations

from src.shared.constants import (
    MHR_JOINT_INDICES as MHR,
    MHR_JOINT_NAMES as MHR_NAMES,
    MHR_SKELETON_EDGES,
)

__all__ = ["MHR", "MHR_NAMES", "MHR_SKELETON_EDGES"]
