"""OpenSim IK worker data constants — marker definitions, weights, and segments.

Pure data module (no numpy/scipy deps) used by src/workers/opensim_ik_worker.py.
Extracted to reduce worker file size and enable reuse.

Coordinate convention (OpenSim): X=forward, Y=up, Z=right(+)/left(-).
"""

from __future__ import annotations


# =============================================================================
# 87-Marker Definitions
# =============================================================================

# (name, opensim_body, x_offset, y_offset, z_offset)
# Body assignments from SAM4Dcap's LaiUhlrich2022_scaled.osim MarkerSet.
# Offsets are approximate generic positions — IK solver refines fit.
OPENSIM_MARKERS: list[tuple[str, str, float, float, float]] = [
    # ── Pelvis (4 surface + 2 computed HJC) ──
    ("r.ASIS_study",    "pelvis",  0.024,  0.016,  0.136),
    ("L.ASIS_study",    "pelvis",  0.027,  0.008, -0.136),
    ("r.PSIS_study",    "pelvis", -0.178,  0.012,  0.053),
    ("L.PSIS_study",    "pelvis", -0.176,  0.009, -0.053),
    ("RHJC_study",      "pelvis", -0.077, -0.073,  0.090),
    ("LHJC_study",      "pelvis", -0.076, -0.080, -0.086),

    # ── Spine / Torso (3: C7 + shoulders) ──
    ("C7_study",             "torso",  -0.068,  0.423,  0.004),
    ("r_shoulder_study",     "torso",   0.017,  0.411,  0.147),
    ("L_shoulder_study",     "torso",   0.015,  0.420, -0.148),

    # ── Right knee lateral + medial ──
    ("r_knee_study",    "femur_r",   0.012, -0.464,  0.077),
    ("r_mknee_study",   "femur_r",   0.015, -0.466, -0.058),

    # ── Left knee lateral + medial ──
    ("L_knee_study",    "femur_l",   0.005, -0.470, -0.084),
    ("L_mknee_study",   "femur_l",   0.006, -0.472,  0.052),

    # ── Right ankle lateral + medial ──
    ("r_ankle_study",   "tibia_r",  -0.033, -0.484,  0.061),
    ("r_mankle_study",  "tibia_r",   0.003, -0.475, -0.045),

    # ── Left ankle lateral + medial ──
    ("L_ankle_study",   "tibia_l",  -0.029, -0.484, -0.058),
    ("L_mankle_study",  "tibia_l",   0.002, -0.474,  0.045),

    # ── Right foot: toe, 5th metatarsal, calcaneus ──
    ("r_toe_study",     "calcn_r",   0.194,  0.015,  0.003),
    ("r_5meta_study",   "calcn_r",   0.144,  0.012,  0.061),
    ("r_calc_study",    "calcn_r",  -0.027,  0.045, -0.013),

    # ── Left foot: toe, 5th metatarsal, calcaneus ──
    ("L_toe_study",     "calcn_l",   0.194,  0.010, -0.005),
    ("L_5meta_study",   "calcn_l",   0.145,  0.009, -0.061),
    ("L_calc_study",    "calcn_l",  -0.023,  0.035,  0.017),

    # ── Right elbow lateral + medial ──
    ("r_lelbow_study",  "humerus_r",  0.017, -0.313,  0.045),
    ("r_melbow_study",  "humerus_r",  0.003, -0.320, -0.056),

    # ── Left elbow lateral + medial ──
    ("L_lelbow_study",  "humerus_l",  0.017, -0.313, -0.045),
    ("L_melbow_study",  "humerus_l",  0.003, -0.320,  0.056),

    # ── Right wrist lateral + medial ──
    ("r_lwrist_study",  "radius_r",   0.001, -0.278,  0.062),
    ("r_mwrist_study",  "radius_r",  -0.027, -0.278, -0.027),

    # ── Left wrist lateral + medial ──
    ("L_lwrist_study",  "radius_l",   0.001, -0.278, -0.062),
    ("L_mwrist_study",  "radius_l",  -0.027, -0.278,  0.027),

    # ── Right thigh clusters ──
    ("r_thigh1_study",  "femur_r",   0.103, -0.157,  0.090),
    ("r_thigh2_study",  "femur_r",   0.065, -0.276,  0.110),
    ("r_thigh3_study",  "femur_r",  -0.030, -0.150,  0.128),

    # ── Left thigh clusters ──
    ("L_thigh1_study",  "femur_l",   0.104, -0.162, -0.089),
    ("L_thigh2_study",  "femur_l",   0.064, -0.281, -0.113),
    ("L_thigh3_study",  "femur_l",  -0.031, -0.156, -0.129),

    # ── Right shank clusters ──
    ("r_sh1_study",     "tibia_r",  -0.008, -0.135,  0.089),
    ("r_sh2_study",     "tibia_r",   0.020, -0.283,  0.099),
    ("r_sh3_study",     "tibia_r",  -0.069, -0.272,  0.096),

    # ── Left shank clusters ──
    ("L_sh1_study",     "tibia_l",  -0.010, -0.139, -0.092),
    ("L_sh2_study",     "tibia_l",   0.021, -0.286, -0.098),
    ("L_sh3_study",     "tibia_l",  -0.067, -0.274, -0.096),

    # ── MHR joint centers (10) — ground-truth skeleton from SAM 3D ──
    ("MHR_RHip",        "pelvis",     -0.077, -0.073,  0.090),
    ("MHR_LHip",        "pelvis",     -0.076, -0.080, -0.086),
    ("MHR_RKnee",       "femur_r",     0.013, -0.465,  0.010),
    ("MHR_LKnee",       "femur_l",     0.006, -0.471, -0.016),
    ("MHR_RAnkle",      "tibia_r",    -0.015, -0.480,  0.008),
    ("MHR_LAnkle",      "tibia_l",    -0.013, -0.480, -0.008),
    ("MHR_RShoulder",   "torso",       0.017,  0.411,  0.147),
    ("MHR_LShoulder",   "torso",       0.015,  0.420, -0.148),
    ("MHR_RElbow",      "humerus_r",   0.010, -0.316,  0.000),
    ("MHR_LElbow",      "humerus_l",   0.010, -0.316,  0.000),

    # ── MHR70 surface keypoints (34) — body/feet/hands/anatomical ──
    # Head (on torso — no head body in LaiUhlrich2022)
    ("Nose",             "torso",      -0.010,  0.530,  0.000),
    ("LEye",             "torso",       0.000,  0.540, -0.030),
    ("REye",             "torso",       0.000,  0.540,  0.030),
    ("LEar",             "torso",      -0.060,  0.530, -0.065),
    ("REar",             "torso",      -0.060,  0.530,  0.065),
    # Body joints
    ("LShoulder",        "torso",       0.015,  0.420, -0.148),
    ("RShoulder",        "torso",       0.017,  0.411,  0.147),
    ("LElbow",           "humerus_l",   0.010, -0.316,  0.000),
    ("RElbow",           "humerus_r",   0.010, -0.316,  0.000),
    ("LHip",             "pelvis",     -0.076, -0.080, -0.086),
    ("RHip",             "pelvis",     -0.077, -0.073,  0.090),
    ("LKnee",            "femur_l",     0.006, -0.471, -0.016),
    ("RKnee",            "femur_r",     0.013, -0.465,  0.010),
    ("LAnkle",           "tibia_l",    -0.013, -0.480, -0.008),
    ("RAnkle",           "tibia_r",    -0.015, -0.480,  0.008),
    # Feet
    ("LBigToe",          "calcn_l",     0.194,  0.012, -0.001),
    ("LSmallToe",        "calcn_l",     0.145,  0.010, -0.061),
    ("LHeel",            "calcn_l",    -0.025,  0.038,  0.010),
    ("RBigToe",          "calcn_r",     0.194,  0.014, -0.001),
    ("RSmallToe",        "calcn_r",     0.144,  0.012,  0.061),
    ("RHeel",            "calcn_r",    -0.027,  0.042, -0.010),
    # Hand markers for forearm rotation
    ("RIndex3",          "hand_r",      0.020, -0.020,  0.020),
    ("RMiddleTip",       "hand_r",      0.050, -0.060,  0.000),
    ("LIndex3",          "hand_l",      0.020, -0.020, -0.020),
    ("LMiddleTip",       "hand_l",      0.050, -0.060,  0.000),
    # Wrists
    ("RWrist",           "radius_r",   -0.013, -0.278,  0.018),
    ("LWrist",           "radius_l",   -0.013, -0.278, -0.018),
    # Posterior/anterior elbow surface
    ("LOlecranon",       "ulna_l",     -0.025, -0.010,  0.020),
    ("ROlecranon",       "ulna_r",     -0.025, -0.010, -0.020),
    ("LCubitalFossa",    "ulna_l",      0.025, -0.010, -0.020),
    ("RCubitalFossa",    "ulna_r",      0.025, -0.010,  0.020),
    # Shoulder surface + neck
    ("LAcromion",        "torso",       0.000,  0.420, -0.180),
    ("RAcromion",        "torso",       0.000,  0.420,  0.180),
    ("Neck",             "torso",      -0.050,  0.450,  0.000),
]


# =============================================================================
# IK Marker Weights
# =============================================================================

# From SAM4Dcap IK setup XML + MHR joint center priorities.
# Higher weight = marker position is more trusted in IK solution.
OPENSIM_MARKER_WEIGHTS: dict[str, float] = {
    # ── Surface markers (vertex lookup) ──
    # Pelvis landmarks (15) — reduced from 25 to let MHR skeleton dominate
    "r.ASIS_study": 5, "L.ASIS_study": 5,
    "r.PSIS_study": 5, "L.PSIS_study": 5,
    # Bell HJC (10) — derived from ASIS/PSIS, not independent; reduced from 25
    "RHJC_study": 10, "LHJC_study": 10,
    # Knee lateral + medial (30)
    "r_knee_study": 30, "r_mknee_study": 30,
    "L_knee_study": 30, "L_mknee_study": 30,
    # Ankle lateral + medial (30)
    "r_ankle_study": 30, "r_mankle_study": 30,
    "L_ankle_study": 30, "L_mankle_study": 30,
    # Foot: toe + 5meta (30), calcaneus (60)
    "r_toe_study": 30, "r_5meta_study": 30, "r_calc_study": 60,
    "L_toe_study": 30, "L_5meta_study": 30, "L_calc_study": 60,
    # Shoulders + C7 (5)
    "C7_study": 5,
    "r_shoulder_study": 5, "L_shoulder_study": 5,
    # Elbows (5)
    "r_lelbow_study": 5, "r_melbow_study": 5,
    "L_lelbow_study": 5, "L_melbow_study": 5,
    # Wrists (5)
    "r_lwrist_study": 5, "r_mwrist_study": 5,
    "L_lwrist_study": 5, "L_mwrist_study": 5,
    # Thigh clusters (4)
    "r_thigh1_study": 4, "r_thigh2_study": 4, "r_thigh3_study": 4,
    "L_thigh1_study": 4, "L_thigh2_study": 4, "L_thigh3_study": 4,
    # Shank clusters (4)
    "r_sh1_study": 4, "r_sh2_study": 4, "r_sh3_study": 4,
    "L_sh1_study": 4, "L_sh2_study": 4, "L_sh3_study": 4,

    # ── MHR joint centers ──
    "MHR_RHip": 100, "MHR_LHip": 100,
    "MHR_RKnee": 25, "MHR_LKnee": 25,
    "MHR_RAnkle": 25, "MHR_LAnkle": 25,
    "MHR_RShoulder": 20, "MHR_LShoulder": 20,
    "MHR_RElbow": 15, "MHR_LElbow": 15,

    # ── MHR70 surface keypoints (0 = disabled for testing) ──
    "Nose": 0, "LEye": 0, "REye": 0, "LEar": 0, "REar": 0,
    "LShoulder": 0, "RShoulder": 0,
    "LElbow": 0, "RElbow": 0,
    "LHip": 0, "RHip": 0,
    "LKnee": 0, "RKnee": 0,
    "LAnkle": 0, "RAnkle": 0,
    "LBigToe": 0, "LSmallToe": 0, "LHeel": 0,
    "RBigToe": 0, "RSmallToe": 0, "RHeel": 0,
    "RIndex3": 0, "RMiddleTip": 0,
    "LIndex3": 0, "LMiddleTip": 0,
    "RWrist": 0, "LWrist": 0,
    "LOlecranon": 0, "ROlecranon": 0,
    "LCubitalFossa": 0, "RCubitalFossa": 0,
    "LAcromion": 0, "RAcromion": 0,
    "Neck": 0,
}


# =============================================================================
# Translation DOFs (skip during angle unwrapping)
# =============================================================================

TRANSLATION_DOFS: set[str] = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}


# =============================================================================
# MHR Segment Scaling Definitions
# =============================================================================

# MHR joint indices for segment length computation.
# MHR body-centric coordinates: X=right, Y=up, Z=backward.
# Rest-pose joint positions encode personalized body proportions from
# the SAM 3D neural shape model (no depth ambiguity).
MHR_SEGMENTS: dict[str, dict] = {
    "femur_r": {
        "pairs": [(18, 19)],   # r_upleg → r_lowleg
        "bodies": ["femur_r", "patella_r"],
        "axes": [0, 1, 2],
    },
    "femur_l": {
        "pairs": [(2, 3)],    # l_upleg → l_lowleg
        "bodies": ["femur_l", "patella_l"],
        "axes": [0, 1, 2],
    },
    "tibia_r": {
        "pairs": [(19, 20)],  # r_lowleg → r_foot
        "bodies": ["tibia_r"],
        "axes": [0, 1, 2],
    },
    "tibia_l": {
        "pairs": [(3, 4)],    # l_lowleg → l_foot
        "bodies": ["tibia_l"],
        "axes": [0, 1, 2],
    },
    "foot": {
        "pairs": [(20, 24), (4, 8)],  # r_foot→r_ball, l_foot→l_ball
        "bodies": ["talus_r", "calcn_r", "toes_r", "talus_l", "calcn_l", "toes_l"],
        "axes": [0, 1, 2],
    },
    "pelvis": {
        "pairs": [(2, 18)],   # l_upleg → r_upleg (hip width)
        "bodies": ["pelvis"],
        "axes": [0, 1, 2],
    },
    "torso": {
        "pairs": [(34, 37)],  # c_spine0 → c_spine3 (lumbar to upper trunk)
        "bodies": ["torso", "scapulaPhantom_r", "scapulaPhantom_l"],
        "axes": [0, 1, 2],
    },
    "humerus": {
        "pairs": [(39, 40), (75, 76)],  # shoulder→elbow (avg L+R)
        "bodies": ["humerus_r", "humerus_l"],
        "axes": [0, 1, 2],
    },
    "radius": {
        "pairs": [(40, 42), (76, 78)],  # elbow→wrist (avg L+R)
        "bodies": ["ulna_r", "radius_r", "hand_r", "ulna_l", "radius_l", "hand_l"],
        "axes": [0, 1, 2],
    },
}

# OpenSim body-pair equivalents for each MHR segment measurement.
# Used to compute the generic model's segment length for ratio comparison.
OSIM_SEGMENT_BODY_PAIRS: dict[str, tuple[str, str]] = {
    "femur_r": ("femur_r", "tibia_r"),
    "femur_l": ("femur_l", "tibia_l"),
    "tibia_r": ("tibia_r", "talus_r"),
    "tibia_l": ("tibia_l", "talus_l"),
    "foot":    ("calcn_r", "toes_r"),
    "pelvis":  ("femur_r", "femur_l"),
    "torso":   ("torso", "humerus_r"),  # torso body extent: lumbar to shoulder
    "humerus": ("humerus_r", "radius_r"),
    "radius":  ("radius_r", "hand_r"),
}


# =============================================================================
# FK Body Export Constants (for demo visualization)
# =============================================================================

# Bodies to skip in FK export (phantom/patella)
FK_SKIP_BODIES: set[str] = {
    "patella_r", "patella_l", "scapulaPhantom_r", "scapulaPhantom_l",
}

# Skeleton edges (parent → child body name) for visualization
FK_SKELETON_EDGES: list[tuple[str, str]] = [
    ("pelvis", "femur_r"), ("pelvis", "femur_l"),
    ("femur_r", "tibia_r"), ("femur_l", "tibia_l"),
    ("tibia_r", "talus_r"), ("tibia_l", "talus_l"),
    ("talus_r", "calcn_r"), ("talus_l", "calcn_l"),
    ("calcn_r", "toes_r"), ("calcn_l", "toes_l"),
    ("pelvis", "torso"),
    ("torso", "humerus_r"), ("torso", "humerus_l"),
    ("humerus_r", "ulna_r"), ("humerus_l", "ulna_l"),
    ("ulna_r", "hand_r"), ("ulna_l", "hand_l"),
]

# Body colors for Three.js visualization
# Green=spine, Red=right, Blue=left, Purple=right arm, Cyan=left arm
FK_BODY_COLORS: dict[str, str] = {
    "pelvis": "#4CAF50", "torso": "#4CAF50",
    "femur_r": "#F44336", "femur_l": "#2196F3",
    "tibia_r": "#F44336", "tibia_l": "#2196F3",
    "talus_r": "#F44336", "talus_l": "#2196F3",
    "calcn_r": "#F44336", "calcn_l": "#2196F3",
    "toes_r": "#F44336", "toes_l": "#2196F3",
    "humerus_r": "#9C27B0", "humerus_l": "#00BCD4",
    "ulna_r": "#9C27B0", "ulna_l": "#00BCD4",
    "radius_r": "#9C27B0", "radius_l": "#00BCD4",
    "hand_r": "#9C27B0", "hand_l": "#00BCD4",
}
