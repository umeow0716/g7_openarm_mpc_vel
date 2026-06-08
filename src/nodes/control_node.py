import time
import numpy as np

from damiao_motor import DaMiaoController
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
    LEFT_HAND_MIRROR_STATE_IDX,
    RIGHT_HAND_PHYSICAL_CMD_IDX,
    RIGHT_HAND_MIRROR_STATE_IDX,
)

# Keep this consistent with sport_node.py.
# Commands below this threshold are treated as stopped wheels.
WHEEL_OMEGA_DEADBAND = 0.4
STEERING_HOLD_VEL = 0.0
STEERING_MOVE_VEL = 20.0


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
        
        self.wheel_controller = DaMiaoController(channel="can0", bustype="socketcan", fd=True)
        self.wheel_motors = [
            self.wheel_controller.add_motor(motor_id=0x01, feedback_id=0x11, motor_type="8009"),
            self.wheel_controller.add_motor(motor_id=0x02, feedback_id=0x12, motor_type="8009"),
            self.wheel_controller.add_motor(motor_id=0x03, feedback_id=0x13, motor_type="8009"),
            self.wheel_controller.add_motor(motor_id=0x04, feedback_id=0x14, motor_type="8009"),
            self.wheel_controller.add_motor(motor_id=0x05, feedback_id=0x15, motor_type="6006"),
            self.wheel_controller.add_motor(motor_id=0x06, feedback_id=0x16, motor_type="6006"),
            self.wheel_controller.add_motor(motor_id=0x07, feedback_id=0x17, motor_type="6006"),
            self.wheel_controller.add_motor(motor_id=0x08, feedback_id=0x18, motor_type="6006"),
        ]
        
        self.left_hand_controller = DaMiaoController(channel="can1", bustype="socketcan", fd=True)
        self.left_hand_motors = [
            self.left_hand_controller.add_motor(motor_id=0x01, feedback_id=0x11, motor_type="8009"),
            self.left_hand_controller.add_motor(motor_id=0x02, feedback_id=0x12, motor_type="8009"),
            self.left_hand_controller.add_motor(motor_id=0x03, feedback_id=0x13, motor_type="4340"),
            self.left_hand_controller.add_motor(motor_id=0x04, feedback_id=0x14, motor_type="4340"),
            self.left_hand_controller.add_motor(motor_id=0x05, feedback_id=0x15, motor_type="4310"),
            self.left_hand_controller.add_motor(motor_id=0x06, feedback_id=0x16, motor_type="4310"),
            self.left_hand_controller.add_motor(motor_id=0x07, feedback_id=0x17, motor_type="4310"),
            self.left_hand_controller.add_motor(motor_id=0x08, feedback_id=0x18, motor_type="4310"),
        ]
        
        self.right_hand_controller = DaMiaoController(channel="can2", bustype="socketcan", fd=True)
        self.right_hand_motors = [
            self.right_hand_controller.add_motor(motor_id=0x01, feedback_id=0x11, motor_type="8009"),
            self.right_hand_controller.add_motor(motor_id=0x02, feedback_id=0x12, motor_type="8009"),
            self.right_hand_controller.add_motor(motor_id=0x03, feedback_id=0x13, motor_type="4340"),
            self.right_hand_controller.add_motor(motor_id=0x04, feedback_id=0x14, motor_type="4340"),
            self.right_hand_controller.add_motor(motor_id=0x05, feedback_id=0x15, motor_type="4310"),
            self.right_hand_controller.add_motor(motor_id=0x06, feedback_id=0x16, motor_type="4310"),
            self.right_hand_controller.add_motor(motor_id=0x07, feedback_id=0x17, motor_type="4310"),
            self.right_hand_controller.add_motor(motor_id=0x08, feedback_id=0x18, motor_type="4310"),
        ]
        
        self.wheel_controller.enable_all()
        self.left_hand_controller.enable_all()
        self.right_hand_controller.enable_all()
        time.sleep(0.1)
        for motor in self.wheel_motors[:4]:
            motor.ensure_control_mode("POSVEL")
        for motor in self.wheel_motors[4:]:
            motor.ensure_control_mode("VEL")
        for motor in self.left_hand_motors + self.right_hand_motors:
            motor.ensure_control_mode("MIT")
        time.sleep(0.1)
        
        self.steering_signs = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        self.drive_signs = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
        
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
        wheel_cmd_dq = np.array(
            [
                apply_scalar_deadband(
                    self.low_cmd.motor_cmd[4 + i].dq,  # type: ignore
                    WHEEL_OMEGA_DEADBAND,
                )
                for i in range(4)
            ],
            dtype=np.float64,
        )
        wheel_is_stopped = np.isclose(wheel_cmd_dq, 0.0)

        for i, motor in enumerate(self.wheel_motors[:4]):
            if wheel_is_stopped[i]:
                steer_target_raw = float(motor.state.get('pos', 0.0))
                steer_vel = STEERING_HOLD_VEL
            else:
                steer_target_raw = (
                    self.low_cmd.motor_cmd[i].q * self.steering_signs[i]  # type: ignore
                )
                steer_vel = STEERING_MOVE_VEL

            motor.send_cmd_pos_vel(
                steer_target_raw,
                steer_vel,
            )

        for i, motor in enumerate(self.wheel_motors[4:]):
            motor.send_cmd_vel(
                wheel_cmd_dq[i] * self.drive_signs[i]
            )

        for i, motor in enumerate(self.left_hand_motors):
            cmd_idx = int(LEFT_HAND_PHYSICAL_CMD_IDX[i])
            motor.send_cmd_mit(
                self.low_cmd.motor_cmd[cmd_idx].q,   # type: ignore
                self.low_cmd.motor_cmd[cmd_idx].dq,  # type: ignore
                self.low_cmd.motor_cmd[cmd_idx].kp,  # type: ignore
                self.low_cmd.motor_cmd[cmd_idx].kd,  # type: ignore
                self.low_cmd.motor_cmd[cmd_idx].tau, # type: ignore
            )
        for i, motor in enumerate(self.right_hand_motors):
            cmd_idx = int(RIGHT_HAND_PHYSICAL_CMD_IDX[i])
            motor.send_cmd_mit(
                self.low_cmd.motor_cmd[cmd_idx].q,   # type: ignore
                self.low_cmd.motor_cmd[cmd_idx].dq,  # type: ignore
                self.low_cmd.motor_cmd[cmd_idx].kp,  # type: ignore
                self.low_cmd.motor_cmd[cmd_idx].kd,  # type: ignore
                self.low_cmd.motor_cmd[cmd_idx].tau, # type: ignore
            )

        for i, motor in enumerate(self.wheel_motors[:4]):
            sign = self.steering_signs[i]
            self.low_state.motor_state[i].q = float(motor.state.get('pos', 0.0)) * sign # type: ignore
            self.low_state.motor_state[i].dq = float(motor.state.get('vel', 0.0)) * sign # type: ignore
            self.low_state.motor_state[i].tau_est = float(motor.state.get('tauq', 0.0)) * sign # type: ignore

        for i, motor in enumerate(self.wheel_motors[4:]):
            idx = 4 + i
            sign = self.drive_signs[i]
            self.low_state.motor_state[idx].q = float(motor.state.get('pos', 0.0)) * sign # type: ignore
            self.low_state.motor_state[idx].dq = float(motor.state.get('vel', 0.0)) * sign # type: ignore
            self.low_state.motor_state[idx].tau_est = float(motor.state.get('tauq', 0.0)) * sign # type: ignore

        for i, motor in enumerate(self.left_hand_motors):
            idx = int(LEFT_HAND_PHYSICAL_CMD_IDX[i])
            self.low_state.motor_state[idx].q = float(motor.state.get('pos', 0.0)) # type: ignore
            self.low_state.motor_state[idx].dq = float(motor.state.get('vel', 0.0)) # type: ignore
            self.low_state.motor_state[idx].tau_est = float(motor.state.get('tauq', 0.0)) # type: ignore

        for src_idx, dst_idx in LEFT_HAND_MIRROR_STATE_IDX.items():
            self.low_state.motor_state[dst_idx].q = self.low_state.motor_state[src_idx].q # type: ignore
            self.low_state.motor_state[dst_idx].dq = self.low_state.motor_state[src_idx].dq # type: ignore
            self.low_state.motor_state[dst_idx].tau_est = self.low_state.motor_state[src_idx].tau_est # type: ignore

        for i, motor in enumerate(self.right_hand_motors):
            idx = int(RIGHT_HAND_PHYSICAL_CMD_IDX[i])
            self.low_state.motor_state[idx].q = float(motor.state.get('pos', 0.0)) # type: ignore
            self.low_state.motor_state[idx].dq = float(motor.state.get('vel', 0.0)) # type: ignore
            self.low_state.motor_state[idx].tau_est = float(motor.state.get('tauq', 0.0)) # type: ignore

        for src_idx, dst_idx in RIGHT_HAND_MIRROR_STATE_IDX.items():
            self.low_state.motor_state[dst_idx].q = self.low_state.motor_state[src_idx].q # type: ignore
            self.low_state.motor_state[dst_idx].dq = self.low_state.motor_state[src_idx].dq # type: ignore
            self.low_state.motor_state[dst_idx].tau_est = self.low_state.motor_state[src_idx].tau_est # type: ignore

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