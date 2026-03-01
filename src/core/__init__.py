"""Core pipeline modules for 3D human pose estimation.

Pipeline: BBoxDetector (YOLOX) -> SAM 3D Body -> Clinical Angles + OpenSim IK
"""

from . import detection
from . import lifting
from . import conversion
from . import video
from . import kinematics
from . import pipeline

__all__ = [
    "detection",
    "lifting",
    "conversion",
    "video",
    "kinematics",
    "pipeline",
]
