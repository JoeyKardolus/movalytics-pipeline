"""Shared constants for COCO-17 and Halpe-26 skeleton formats.

This is the canonical source for keypoint definitions.
All modules should import from here to maintain consistency.

Contains:
- Keypoint names and indices (COCO-17, Halpe-26)
- Skeleton connections
- Marker name mappings
- Visualization colors
"""

from typing import Dict, List, Tuple, Union

# =============================================================================
# COCO-17 Keypoint Names
# =============================================================================

# Canonical COCO-17 keypoint names (lowercase with underscore)
# This is the standard format used by RTMPose, OpenPose, etc.
COCO_KEYPOINT_NAMES: List[str] = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Legacy/visualization format (L_/R_ prefix)
COCO_JOINT_NAMES: List[str] = [
    "nose",       # 0
    "L_eye",      # 1
    "R_eye",      # 2
    "L_ear",      # 3
    "R_ear",      # 4
    "L_shoulder", # 5
    "R_shoulder", # 6
    "L_elbow",    # 7
    "R_elbow",    # 8
    "L_wrist",    # 9
    "R_wrist",    # 10
    "L_hip",      # 11
    "R_hip",      # 12
    "L_knee",     # 13
    "R_knee",     # 14
    "L_ankle",    # 15
    "R_ankle",    # 16
]

# Short names for compact display
COCO_JOINT_NAMES_SHORT: List[str] = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr",
    "L_hip", "R_hip", "L_kn", "R_kn", "L_an", "R_an",
]

# Index lookup by name
COCO_KEYPOINT_INDICES: Dict[str, int] = {
    name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)
}

NUM_COCO_KEYPOINTS: int = 17


# =============================================================================
# Skeleton Connections
# =============================================================================

# COCO-17 skeleton connections (parent_idx, child_idx)
COCO_SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    # Head
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    # Shoulders
    (5, 6),   # left_shoulder -> right_shoulder
    # Left arm
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    # Right arm
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    # Torso
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12), # left_hip -> right_hip
    # Left leg
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    # Right leg
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle
]

# Alias for backwards compatibility
COCO_SKELETON = COCO_SKELETON_CONNECTIONS


# =============================================================================
# TRC Marker Sets
# =============================================================================

# Standard 21-marker set (COCO-17 + derived, without Head)
MARKER_NAMES_21: List[str] = [
    "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
    "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
    "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist",
    "Hip", "Nose"
]


# =============================================================================
# Marker Name Mappings
# =============================================================================

# COCO-17 index to TRC marker name (used by datastream)
COCO_TO_MARKER_NAME: Dict[int, str] = {
    0: "Nose",
    5: "LShoulder",
    6: "RShoulder",
    7: "LElbow",
    8: "RElbow",
    9: "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "RKnee",
    15: "LAnkle",
    16: "RAnkle",
}

# Reverse mapping: marker name to COCO index
MARKER_NAME_TO_COCO: Dict[str, int] = {v: k for k, v in COCO_TO_MARKER_NAME.items()}

# COCO-WholeBody foot keypoints (indices 17-22, from rtmlib Wholebody output)
COCO_WHOLEBODY_FOOT_INDICES: Dict[int, str] = {
    17: "LBigToe",
    18: "LSmallToe",
    19: "LHeel",
    20: "RBigToe",
    21: "RSmallToe",
    22: "RHeel",
}

# Foot index within 6-element array → marker name (COCO WholeBody grouped ordering)
FOOT_IDX_TO_MARKER: Dict[int, str] = {
    i: name for i, (_, name) in enumerate(sorted(COCO_WHOLEBODY_FOOT_INDICES.items()))
}

# Halpe-26 foot index within 6-element slice [20:26] → marker name (interleaved L/R)
HALPE_FOOT_IDX_TO_MARKER: Dict[int, str] = {
    0: "LBigToe",     # Halpe index 20
    1: "RBigToe",     # Halpe index 21
    2: "LSmallToe",   # Halpe index 22
    3: "RSmallToe",   # Halpe index 23
    4: "LHeel",       # Halpe index 24
    5: "RHeel",       # Halpe index 25
}


# =============================================================================
# Halpe-26 Keypoint Definitions
# =============================================================================

# Halpe-26 = COCO-17 body + head_top + neck + hip_center + 6 feet
# Output format from rtmlib.BodyWithFeet (verified empirically)
HALPE_KEYPOINT_NAMES: List[str] = [
    *COCO_KEYPOINT_NAMES,   # 0-16: same as COCO-17
    "head_top",             # 17
    "neck",                 # 18
    "hip_center",           # 19
    "left_big_toe",         # 20
    "right_big_toe",        # 21
    "left_small_toe",       # 22
    "right_small_toe",      # 23
    "left_heel",            # 24
    "right_heel",           # 25
]

NUM_HALPE_KEYPOINTS: int = 26

# Halpe-26 index → TRC marker name (superset of COCO_TO_MARKER_NAME)
# HeadTop (17) is not mapped — not in the 21-marker TRC set
HALPE_TO_MARKER_NAME: Dict[int, str] = {
    **COCO_TO_MARKER_NAME,
    # 17: HeadTop — no corresponding TRC marker
    18: "Neck",
    19: "Hip",
    20: "LBigToe",
    21: "RBigToe",
    22: "LSmallToe",
    23: "RSmallToe",
    24: "LHeel",
    25: "RHeel",
}

# Halpe-26 skeleton connections (extends COCO-17)
HALPE_SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    *COCO_SKELETON_CONNECTIONS,
    # Head/neck/hip
    (0, 17),   # nose -> head_top
    (0, 18),   # nose -> neck
    (18, 5),   # neck -> left_shoulder
    (18, 6),   # neck -> right_shoulder
    (19, 11),  # hip_center -> left_hip
    (19, 12),  # hip_center -> right_hip
    # Left foot: ankle(15) → big_toe(20), small_toe(22), heel(24)
    (15, 20),  # left_ankle -> left_big_toe
    (15, 22),  # left_ankle -> left_small_toe
    (15, 24),  # left_ankle -> left_heel
    # Right foot: ankle(16) → big_toe(21), small_toe(23), heel(25)
    (16, 21),  # right_ankle -> right_big_toe
    (16, 23),  # right_ankle -> right_small_toe
    (16, 25),  # right_ankle -> right_heel
]


# =============================================================================
# Utility Functions
# =============================================================================

def get_marker_name(coco_index: int) -> str:
    """Get TRC marker name for a COCO keypoint index.

    Args:
        coco_index: COCO-17 keypoint index (0-16).

    Returns:
        Marker name (e.g., "LShoulder") or empty string if not mapped.
    """
    return COCO_TO_MARKER_NAME.get(coco_index, "")


def get_coco_index(marker_name: str) -> int:
    """Get COCO keypoint index for a marker name.

    Args:
        marker_name: TRC marker name (e.g., "LShoulder").

    Returns:
        COCO index or -1 if not found.
    """
    return MARKER_NAME_TO_COCO.get(marker_name, -1)

# Joint colors by body part
JOINT_COLORS = {
    "head": "#FF6B6B",      # Red - nose, eyes, ears (0-4)
    "shoulder": "#4ECDC4",  # Teal - shoulders (5-6)
    "arm": "#45B7D1",       # Blue - elbows, wrists (7-10)
    "hip": "#96CEB4",       # Green - hips (11-12)
    "leg": "#FFEAA7",       # Yellow - knees, ankles (13-16)
}

# Map joint index to color
def get_joint_color(joint_idx: int) -> str:
    """Get color for joint by index."""
    if joint_idx <= 4:
        return JOINT_COLORS["head"]
    elif joint_idx <= 6:
        return JOINT_COLORS["shoulder"]
    elif joint_idx <= 10:
        return JOINT_COLORS["arm"]
    elif joint_idx <= 12:
        return JOINT_COLORS["hip"]
    else:
        return JOINT_COLORS["leg"]

# Limb colors (for skeleton visualization)
LIMB_COLORS = {
    "left": "#3498DB",   # Blue for left side
    "right": "#E74C3C",  # Red for right side
    "center": "#2ECC71", # Green for center (torso, etc.)
}

def get_limb_color(joint_i: int, joint_j: int) -> str:
    """Get color for limb based on connected joints."""
    name_i = COCO_JOINT_NAMES[joint_i]
    name_j = COCO_JOINT_NAMES[joint_j]

    if "L_" in name_i or "L_" in name_j:
        return LIMB_COLORS["left"]
    elif "R_" in name_i or "R_" in name_j:
        return LIMB_COLORS["right"]
    else:
        return LIMB_COLORS["center"]


# =============================================================================
# MHR Body Model Constants (SAM 3D Body / Meta)
# =============================================================================

# MHR 127-joint skeleton: named indices for the joints we use.
# From FBX skeleton enumeration (body_world=0, root/pelvis=1, ...).
MHR_JOINT_INDICES: Dict[str, int] = {
    "body_world": 0,
    "root": 1,       # pelvis
    "l_upleg": 2,
    "l_lowleg": 3,
    "l_foot": 4,
    "l_talocrural": 5,
    "l_subtalar": 6,
    "l_transversetarsal": 7,
    "l_ball": 8,
    "r_upleg": 18,
    "r_lowleg": 19,
    "r_foot": 20,
    "r_talocrural": 21,
    "r_subtalar": 22,
    "r_transversetarsal": 23,
    "r_ball": 24,
    "c_spine0": 34,
    "c_spine1": 35,
    "c_spine2": 36,
    "c_spine3": 37,
    "r_clavicle": 38,
    "r_uparm": 39,
    "r_lowarm": 40,
    "r_wrist_twist": 41,
    "r_wrist": 42,
    "l_clavicle": 74,
    "l_uparm": 75,
    "l_lowarm": 76,
    "l_wrist_twist": 77,
    "l_wrist": 78,
    "c_neck": 110,
    "c_head": 113,
}

# Reverse mapping: index → name (for the joints we care about)
MHR_JOINT_NAMES: Dict[int, str] = {v: k for k, v in MHR_JOINT_INDICES.items()}

# Skeleton topology — main kinematic chains only (skip twist/proc joints).
# Each tuple is (parent_idx, child_idx).
MHR_SKELETON_EDGES: List[Tuple[int, int]] = [
    # Spine chain: root → spine0 → spine1 → spine2 → spine3
    (1, 34), (34, 35), (35, 36), (36, 37),
    # Head: spine3 → neck → head
    (37, 110), (110, 113),
    # Left leg: root → l_upleg → l_lowleg → l_foot → l_ball
    (1, 2), (2, 3), (3, 4), (4, 8),
    # Right leg: root → r_upleg → r_lowleg → r_foot → r_ball
    (1, 18), (18, 19), (19, 20), (20, 24),
    # Left arm: spine3 → l_clavicle → l_uparm → l_lowarm → l_wrist
    (37, 74), (74, 75), (75, 76), (76, 78),
    # Right arm: spine3 → r_clavicle → r_uparm → r_lowarm → r_wrist
    (37, 38), (38, 39), (39, 40), (40, 42),
]

# 10 MHR joint centers: name → MHR joint index.
# Direct 3D joint positions from SAM 3D body model skeleton.
MHR_JOINT_CENTERS: Dict[str, int] = {
    "MHR_LHip": 2,        # l_upleg
    "MHR_RHip": 18,       # r_upleg
    "MHR_LKnee": 3,       # l_lowleg
    "MHR_RKnee": 19,      # r_lowleg
    "MHR_LAnkle": 4,      # l_foot
    "MHR_RAnkle": 20,     # r_foot
    "MHR_LShoulder": 75,  # l_uparm
    "MHR_RShoulder": 39,  # r_uparm
    "MHR_LElbow": 76,     # l_lowarm
    "MHR_RElbow": 40,     # r_lowarm
}

# 34 MHR70 surface keypoints: MHR70 index → TRC marker name.
# From SAM3D-OpenSim reference (Cho et al. / Iriondo et al.)
MHR70_KEYPOINTS: Dict[int, str] = {
    # Body (COCO17-like + feet)
    0: "Nose", 1: "LEye", 2: "REye", 3: "LEar", 4: "REar",
    5: "LShoulder", 6: "RShoulder",
    7: "LElbow", 8: "RElbow",
    9: "LHip", 10: "RHip",
    11: "LKnee", 12: "RKnee",
    13: "LAnkle", 14: "RAnkle",
    15: "LBigToe", 16: "LSmallToe", 17: "LHeel",
    18: "RBigToe", 19: "RSmallToe", 20: "RHeel",
    # Hand markers for forearm rotation
    28: "RIndex3", 29: "RMiddleTip",
    49: "LIndex3", 50: "LMiddleTip",
    # Wrists
    41: "RWrist", 62: "LWrist",
    # Extra anatomical surface markers
    63: "LOlecranon", 64: "ROlecranon",
    65: "LCubitalFossa", 66: "RCubitalFossa",
    67: "LAcromion", 68: "RAcromion",
    69: "Neck",
}
