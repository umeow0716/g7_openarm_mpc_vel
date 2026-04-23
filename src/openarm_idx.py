"""OpenArm / AMR unified index definitions.

State used by the floating-base model:
- q has length 33
- full library state x_lib = [q, v] has length 65

Optimizer state in the new kinematic MPC:
- x = q
- u = [v_x, v_y, v_w, arm_joint_velocities...]
"""

import numpy as np

# -----------------------------------------------------------------------------
# Kinematics output index
# -----------------------------------------------------------------------------
OPENARM_LEFT_HAND_ALL = slice(0, 7)
OPENARM_LEFT_HAND_POS = slice(0, 3)
OPENARM_LEFT_HAND_QUAT = slice(3, 7)

OPENARM_RIGHT_HAND_ALL = slice(7, 14)
OPENARM_RIGHT_HAND_POS = slice(7, 10)
OPENARM_RIGHT_HAND_QUAT = slice(10, 14)

# -----------------------------------------------------------------------------
# q index for the floating-base model
# q = [world_pos(3), world_quat(4), wheel/base joints(8), left arm(9), right arm(9)]
# total nq = 33
# -----------------------------------------------------------------------------
OPENARM_WORLD_POS = slice(0, 3)
OPENARM_WORLD_QUAT = slice(3, 7)
OPENARM_FLOATING_BASE_ALL = slice(0, 7)

OPENARM_WHEEL_ALL = slice(7, 15)
OPENARM_FL = 7
OPENARM_FLW = 8
OPENARM_FR = 9
OPENARM_FRW = 10
OPENARM_RL = 11
OPENARM_RLW = 12
OPENARM_RR = 13
OPENARM_RRW = 14

OPENARM_LEFT_JOINT_ALL = slice(15, 24)
OPENARM_RIGHT_JOINT_ALL = slice(24, 33)
OPENARM_JOINT_ALL = slice(15, 33)

OPENARM_LEFT_JOINT_IDX = np.arange(15, 24, dtype=np.int32)
OPENARM_RIGHT_JOINT_IDX = np.arange(24, 33, dtype=np.int32)
OPENARM_JOINT_IDX = np.arange(15, 33, dtype=np.int32)

OPENARM_NQ = 33
OPENARM_ARM_NQ = 18

# -----------------------------------------------------------------------------
# v index in the full library state x_lib = [q, v]
# floating base contributes 6 velocities: vx, vy, vz, wx, wy, wz
# total nv = 32 for the user's floating-base model
# -----------------------------------------------------------------------------
OPENARM_V_WORLD_LINEAR = slice(0, 3)
OPENARM_V_WORLD_ANGULAR = slice(3, 6)
OPENARM_V_FLOATING_BASE_ALL = slice(0, 6)
OPENARM_V_WHEEL_ALL = slice(6, 14)
OPENARM_V_LEFT_JOINT_ALL = slice(14, 23)
OPENARM_V_RIGHT_JOINT_ALL = slice(23, 32)
OPENARM_V_JOINT_ALL = slice(14, 32)

OPENARM_NV = 32

# -----------------------------------------------------------------------------
# Optimizer control index
# u = [base_vx, base_vy, base_wz, left_arm(9), right_arm(9)]
# total nu = 21
# -----------------------------------------------------------------------------
OPENARM_U_BASE_VX = 0
OPENARM_U_BASE_VY = 1
OPENARM_U_BASE_WZ = 2
OPENARM_U_BASE_ALL = slice(0, 3)

OPENARM_U_LEFT_JOINT_ALL = slice(3, 12)
OPENARM_U_RIGHT_JOINT_ALL = slice(12, 21)
OPENARM_U_JOINT_ALL = slice(3, 21)
OPENARM_U_ALL = slice(0, 21)

OPENARM_NU = 21
