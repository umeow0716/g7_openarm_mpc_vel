"""Shared actuator/state index mapping.

Public Unitree-style DDS order used by this package:
    LowCmd.motor_cmd[i] and LowState.motor_state[i] use the same actuator order.

Actuator order, length 26:
    0  FL_steer
    1  FR_steer
    2  RL_steer
    3  RR_steer
    4  FL_wheel
    5  FR_wheel
    6  RL_wheel
    7  RR_wheel
    8  L_1
    9  L_2
    10 L_3
    11 L_4
    12 L_5
    13 L_6
    14 L_7
    15 gripper_LL
    16 gripper_LR
    17 R_1
    18 R_2
    19 R_3
    20 R_4
    21 R_5
    22 R_6
    23 R_7
    24 gripper_RL
    25 gripper_RR

MuJoCo / model qpos joint order after floating base is different for the first
8 base joints:
    FL_steer, FL_wheel, FR_steer, FR_wheel, RL_steer, RL_wheel, RR_steer, RR_wheel,
    then left arm 9, right arm 9.

Therefore, convert explicitly whenever data crosses the DDS boundary.
"""

from __future__ import annotations

import numpy as np

NUM_ACTUATORS = 26

STEER_ACT_IDX = np.array([0, 1, 2, 3], dtype=np.int32)
WHEEL_ACT_IDX = np.array([4, 5, 6, 7], dtype=np.int32)
LEFT_ARM_ACT_IDX = np.arange(8, 17, dtype=np.int32)
RIGHT_ARM_ACT_IDX = np.arange(17, 26, dtype=np.int32)
ARM_ACT_IDX = np.arange(8, 26, dtype=np.int32)

# Actuator-order vector -> model joint-order vector qpos[7:33].
ACTUATOR_TO_MODEL_JOINT_ORDER = np.array(
    [0, 4, 1, 5, 2, 6, 3, 7, *range(8, 26)],
    dtype=np.int32,
)

# Model joint-order vector qpos[7:33] -> actuator-order vector.
MODEL_JOINT_TO_ACTUATOR_ORDER = np.array(
    [0, 2, 4, 6, 1, 3, 5, 7, *range(8, 26)],
    dtype=np.int32,
)

# qpos/qvel indices that correspond to actuator order.
ACTUATOR_QPOS_IDX = np.array(
    [7, 9, 11, 13, 8, 10, 12, 14, *range(15, 33)],
    dtype=np.int32,
)
ACTUATOR_QVEL_IDX = np.array(
    [6, 8, 10, 12, 7, 9, 11, 13, *range(14, 32)],
    dtype=np.int32,
)

# Physical DaMiao hand motor mapping used by control_node.py.
# The model has two gripper actuators per hand. The current real hardware list has
# 8 motors per hand, so we drive one gripper slot per hand and mirror its feedback
# into the paired virtual gripper slot to keep LowState length/order consistent.
LEFT_HAND_PHYSICAL_CMD_IDX = np.array([8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32)
LEFT_HAND_MIRROR_STATE_IDX = {15: 16}

RIGHT_HAND_PHYSICAL_CMD_IDX = np.array([17, 18, 19, 20, 21, 22, 23, 24], dtype=np.int32)
RIGHT_HAND_MIRROR_STATE_IDX = {24: 25}
