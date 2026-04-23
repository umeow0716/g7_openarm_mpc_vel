import numpy as np

from .openarm_idx import (
    OPENARM_NQ,
    OPENARM_NU,
    OPENARM_JOINT_ALL,
    OPENARM_U_JOINT_ALL,
)

# -----------------------------------------------------------------------------
# Original OpenArm arm joint position limits (18 joints total)
# Kept exactly as before for lossless mapping.
# -----------------------------------------------------------------------------
ARM_Q_MIN = np.array([
     -3.49,  -3.32,  -1.57,  -0.07,  -1.57, -0.785,  -1.57,
     -0.07,  -0.07,   -1.40, -0.174,  -1.57,  -0.07,  -1.57,
    -0.785,  -1.57,  -0.07,  -0.07,
], dtype=np.float64)

ARM_Q_MAX = np.array([
      1.40, 0.174,  1.57,  2.44,  1.57, 0.785,  1.57,
     0.044, 0.044,  3.49,  3.32,  1.57,  2.44,  1.57,
     0.785, 1.57,  0.044, 0.044,
], dtype=np.float64)

# Original OpenArm joint velocity limits.
# Because the new controller uses u = v, these are the arm-part control limits.
ARM_V_MAX = np.array([
    1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 2.0, 0.5,
    1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 2.0, 0.5,
    0.2, 0.2,
], dtype=np.float64)
ARM_V_MIN = -ARM_V_MAX

# -----------------------------------------------------------------------------
# New AMR base control limits.
# Adjust these 3 values to match the real mobile base.
# [vx, vy, wz]
# -----------------------------------------------------------------------------
BASE_V_MAX = np.array([1.0, 1.0, 1.0], dtype=np.float64)
BASE_V_MIN = -BASE_V_MAX

# -----------------------------------------------------------------------------
# Full q limits for the AMR floating-base model.
# Unconstrained entries are left as +/- inf.
# Only the 18 arm joints inherit the original OpenArm limits exactly.
# -----------------------------------------------------------------------------
Q_MIN = np.full((OPENARM_NQ,), -np.inf, dtype=np.float64)
Q_MAX = np.full((OPENARM_NQ,),  np.inf, dtype=np.float64)
Q_MIN[OPENARM_JOINT_ALL] = ARM_Q_MIN
Q_MAX[OPENARM_JOINT_ALL] = ARM_Q_MAX

# Full optimizer control limits.
U_MIN = np.full((OPENARM_NU,), -np.inf, dtype=np.float64)
U_MAX = np.full((OPENARM_NU,),  np.inf, dtype=np.float64)
U_MIN[:3] = BASE_V_MIN
U_MAX[:3] = BASE_V_MAX
U_MIN[OPENARM_U_JOINT_ALL] = ARM_V_MIN
U_MAX[OPENARM_U_JOINT_ALL] = ARM_V_MAX

FINITE_Q_MASK = np.isfinite(Q_MIN) & np.isfinite(Q_MAX)
FINITE_U_MASK = np.isfinite(U_MIN) & np.isfinite(U_MAX)
