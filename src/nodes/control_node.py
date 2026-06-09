import time
import numpy as np
import openarm_can as oa

from typing import MutableSequence, cast, TYPE_CHECKING
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import \
    unitree_go_msg_dds__IMUState_ as IMUState_default, \
    unitree_hg_msg_dds__LowCmd_ as LowCmd_default, \
    unitree_hg_msg_dds__LowState_ as LowState_default

from .actuator_mapping import (
    LEFT_HAND_PHYSICAL_CMD_IDX,
    RIGHT_HAND_PHYSICAL_CMD_IDX,
)

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_, MotorState_

# Keep this consistent with sport_node.py.
# Commands below this threshold are treated as stopped wheels.
WHEEL_OMEGA_DEADBAND = 0.4
STEERING_HOLD_VEL = 0.0
STEERING_MOVE_VEL = 1.5


def apply_scalar_deadband(value: float, threshold: float) -> float:
    value = float(value)
    if abs(value) < threshold:
        return 0.0
    return value


class ControlNode:
    def __init__(
        self,
        low_state_topic: str = "rt/lowstate",
        imu_state_topic: str = "rt/imustate",
        low_cmd_topic: str = "rt/lowcmd",
    ):
        self.low_state_pub = ChannelPublisher(low_state_topic, LowState_)
        self.low_state_pub.Init()
        
        self.imu_state_sub = ChannelSubscriber(imu_state_topic, IMUState_)
        self.imu_state_sub.Init(self.imu_state_callback, 10)
        
        self.low_cmd_sub = ChannelSubscriber(low_cmd_topic, LowCmd_)
        self.low_cmd_sub.Init(self.low_cmd_callback, 10)
        
        self.imu_state = IMUState_default()
        self.low_cmd   = LowCmd_default()
        self.low_state = LowState_default()
        
        self.wheel_controller = oa.OpenArm('can0', True)
        self.wheel_motor_types = [
            oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM8009,
            oa.MotorType.DM6006, oa.MotorType.DM6006, oa.MotorType.DM6006, oa.MotorType.DM6006,
        ]
        self.wheel_send_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        self.wheel_recv_ids = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]
        self.wheel_control_modes = [oa.ControlMode.POS_VEL] * 4 + [oa.ControlMode.MIT] * 4
        self.wheel_controller.init_arm_motors(self.wheel_motor_types, self.wheel_send_ids, self.wheel_recv_ids, self.wheel_control_modes)
        
        self.left_hand_controller = oa.OpenArm('can1', True)
        self.left_hand_motor_types = [
            oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM4340, oa.MotorType.DM4340,
            oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
        ]
        self.left_hand_send_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        self.left_hand_recv_ids = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]
        self.left_hand_control_modes = [oa.ControlMode.MIT] * 8
        self.left_hand_controller.init_arm_motors(self.left_hand_motor_types, self.left_hand_send_ids, self.left_hand_recv_ids, self.left_hand_control_modes)
        
        self.right_hand_controller = oa.OpenArm('can2', True)
        self.right_hand_motor_types = [
            oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM4340, oa.MotorType.DM4340,
            oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
        ]
        self.right_hand_send_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        self.right_hand_recv_ids = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]
        self.right_hand_control_modes = [oa.ControlMode.MIT] * 8
        self.right_hand_controller.init_arm_motors(self.right_hand_motor_types, self.right_hand_send_ids, self.right_hand_recv_ids, self.right_hand_control_modes)
        
        self.wheel_controller.enable_all()
        self.left_hand_controller.enable_all()
        self.right_hand_controller.enable_all()
        time.sleep(0.1)
        self.wheel_controller.recv_all()
        self.left_hand_controller.recv_all()
        self.right_hand_controller.recv_all()
        time.sleep(0.1)
        self.wheel_controller.set_callback_mode_all(oa.CallbackMode.STATE)
        self.left_hand_controller.set_callback_mode_all(oa.CallbackMode.STATE)
        self.right_hand_controller.set_callback_mode_all(oa.CallbackMode.STATE)

        self.steering_signs   = np.array([ 1.0, -1.0,  1.0, -1.0], dtype=np.float64)
        self.drive_signs      = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
        self.left_hand_signs  = np.array([-1.0, -1.0,  1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64)
        self.right_hand_signs = np.array([ 1.0, -1.0,  1.0, -1.0, 1.0, -1.0,  1.0, 1.0, 1.0], dtype=np.float64)
        
        self.control_dt = 1 / 50
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt, target=self.control_loop, name="control"
        )
        self.lowCmdWriteThreadPtr.Start()
        
    def imu_state_callback(self, msg: 'IMUState_'):
        self.imu_state = msg
    
    def low_cmd_callback(self, msg: 'LowCmd_'):
        self.low_cmd = msg
    
    def control_loop(self):
        motor_cmd   = cast(MutableSequence['MotorCmd_'], self.low_cmd.motor_cmd)
        motor_state = cast(MutableSequence['MotorState_'], self.low_state.motor_state)
        
        wheel_cmd_dq = np.array(
            [
                apply_scalar_deadband(
                    motor_cmd[4 + i].dq,
                    WHEEL_OMEGA_DEADBAND,
                )
                for i in range(4)
            ],
            dtype=np.float64,
        )
        wheel_is_stopped = np.isclose(wheel_cmd_dq, 0.0)

        for i, motor in enumerate(self.wheel_controller.get_arm().get_motors()[:4]):
            if wheel_is_stopped[i]:
                steer_target_raw = motor.get_position()
                steer_vel = STEERING_HOLD_VEL
            else:
                steer_target_raw = (
                    motor_cmd[i].q * self.steering_signs[i]
                )
                steer_vel = STEERING_MOVE_VEL

            param = oa.PosVelParam(
                q=steer_target_raw,
                dq=steer_vel
            )
            self.wheel_controller.get_arm().posvel_control_one(i, param)

        for i, motor in enumerate(self.wheel_controller.get_arm().get_motors()[4:]):
            param = oa.MITParam(
                q=0.0,
                dq=wheel_cmd_dq[i] * self.drive_signs[i],
                kp=0.0,
                kd=2.0,
                tau=0.0,
            )
            self.wheel_controller.get_arm().mit_control_one(i + 4, param)

        for i, motor in enumerate(self.left_hand_controller.get_arm().get_motors()):
            cmd_idx = int(LEFT_HAND_PHYSICAL_CMD_IDX[i])
            sign = self.left_hand_signs[i]
            param = oa.MITParam(
                q=motor_cmd[cmd_idx].q * sign,
                dq=motor_cmd[cmd_idx].dq * sign,
                kp=motor_cmd[cmd_idx].kp,
                kd=motor_cmd[cmd_idx].kd,
                tau=motor_cmd[cmd_idx].tau * sign,
            )
            self.left_hand_controller.get_arm().mit_control_one(i, param)
        
        for i, motor in enumerate(self.right_hand_controller.get_arm().get_motors()):
            cmd_idx = int(RIGHT_HAND_PHYSICAL_CMD_IDX[i])
            sign = self.right_hand_signs[i]
            param = oa.MITParam(
                q=motor_cmd[cmd_idx].q * sign,
                dq=motor_cmd[cmd_idx].dq * sign,
                kp=motor_cmd[cmd_idx].kp,
                kd=motor_cmd[cmd_idx].kd,
                tau=motor_cmd[cmd_idx].tau * sign,
            )
            self.right_hand_controller.get_arm().mit_control_one(i, param)
        
        self.wheel_controller.refresh_all()
        self.left_hand_controller.refresh_all()
        self.right_hand_controller.refresh_all()
        time.sleep(0.0015)
        self.wheel_controller.recv_all()
        self.left_hand_controller.recv_all()
        self.right_hand_controller.recv_all()
                
        for i, motor in enumerate(self.wheel_controller.get_arm().get_motors()[:4]):
            sign = self.steering_signs[i]
            motor_state[i].q = float(motor.get_position()) * sign
            motor_state[i].dq = float(motor.get_velocity()) * sign
            motor_state[i].tau_est = float(motor.get_torque()) * sign

        for i, motor in enumerate(self.wheel_controller.get_arm().get_motors()[4:]):
            idx = 4 + i
            sign = self.drive_signs[i]
            motor_state[idx].q = float(motor.get_position()) * sign
            motor_state[idx].dq = float(motor.get_velocity()) * sign
            motor_state[idx].tau_est = float(motor.get_torque()) * sign

        for i, motor in enumerate(self.left_hand_controller.get_arm().get_motors()):
            idx = int(LEFT_HAND_PHYSICAL_CMD_IDX[i])
            sign = self.left_hand_signs[i]
            motor_state[idx].q = float(motor.get_position() * sign)
            motor_state[idx].dq = float(motor.get_velocity() * sign)
            motor_state[idx].tau_est = float(motor.get_torque() * sign)

        for i, motor in enumerate(self.right_hand_controller.get_arm().get_motors()):
            idx = int(RIGHT_HAND_PHYSICAL_CMD_IDX[i])
            sign = self.right_hand_signs[i]
            motor_state[idx].q = float(motor.get_position() * sign)
            motor_state[idx].dq = float(motor.get_velocity() * sign)
            motor_state[idx].tau_est = float(motor.get_torque() * sign)

        self.low_state_pub.Write(self.low_state)
            

def main():
    ChannelFactoryInitialize()
    node = ControlNode()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.lowCmdWriteThreadPtr.Wait()
    
if __name__ == '__main__':
    main()