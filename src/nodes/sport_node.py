import time
import typing
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_, SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from unitree_sdk2py.idl.default import \
    unitree_go_msg_dds__IMUState_ as IMUState_default, \
    unitree_go_msg_dds__SportModeState_ as SportModeState_default, \
    unitree_hg_msg_dds__LowState_ as LowState_default
    
from ..ekf_localization import AMREKF
from ..swerve_kinematics import SwerveKinematics
from ..config import RobotConfig

DT_TARGET = 0.01
STATE_TIMEOUT_S = 0.2
WHEEL_OMEGA_DEADBAND = 0.2
VX_DEADBAND = 0.005
VY_DEADBAND = 0.005
WZ_DEADBAND = 0.01

def apply_scalar_deadband(value: float, threshold: float) -> float:
    if abs(value) < threshold:
        return 0.0
    return float(value)

def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

class SportNode:
    def __init__(
        self,
        low_state_topic: str = 'rt/lowstate',
        imu_state_topic: str = 'rt/imustate',
        sport_state_mode_topic: str = 'rt/sportmodestate'
    ):
        self.cfg = RobotConfig()
        self.swerve_kin = SwerveKinematics(self.cfg)
        self.ekf = AMREKF()
        
        self.low_state_sub = ChannelSubscriber(low_state_topic, LowState_)
        self.low_state_sub.Init(self.low_state_callback, 10)
        
        self.imu_state_sub = ChannelSubscriber(imu_state_topic, IMUState_)
        self.imu_state_sub.Init(self.imu_state_callback, 10)
        
        self.sport_mode_state_pub = ChannelPublisher(sport_state_mode_topic, SportModeState_)
        self.sport_mode_state_pub.Init()
        
        self.low_state: typing.Optional[LowState_] = None
        self.imu_state: typing.Optional[IMUState_] = None
        self.sport_mode_state = SportModeState_default()
        
        self.position = np.zeros((3,), dtype=np.float64)
        self.velocity = np.zeros((3,), dtype=np.float64)
        
        self.last_loop_t: typing.Optional[float] = None
        self.last_low_state_t: typing.Optional[float] = None
        self.last_imu_state_t: typing.Optional[float] = None
        self.yaw_zero: typing.Optional[float] = None
        
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=DT_TARGET, target=self.control_loop, name="sport"
        )
        self.lowCmdWriteThreadPtr.Start()
        
    def imu_state_callback(self, msg: IMUState_):
        self.imu_state = msg
        self.last_imu_state_t = time.perf_counter()

    def low_state_callback(self, msg: LowState_):
        self.low_state = msg
        self.last_low_state_t = time.perf_counter()

    def control_loop(self):
        if self.low_state is None or self.imu_state is None:
            return

        now = time.perf_counter()
        if (
            self.last_low_state_t is None
            or self.last_imu_state_t is None
            or now - self.last_low_state_t > STATE_TIMEOUT_S
            or now - self.last_imu_state_t > STATE_TIMEOUT_S
        ):
            return

        if self.last_loop_t is None:
            self.last_loop_t = now

        dt = now - self.last_loop_t
        if dt < DT_TARGET:
            return
        
        steering_angles = [
            self.low_state.motor_state[i].q for i in range(4)
        ]
        wheel_omegas = [
            apply_scalar_deadband(
                self.low_state.motor_state[i].dq,
                WHEEL_OMEGA_DEADBAND,
            )
            for i in range(4, 8)
        ]
        vx_odom, vy_odom, wz_odom = self.swerve_kin.forward(steering_angles, wheel_omegas)
        vx_odom = apply_scalar_deadband(vx_odom, VX_DEADBAND)
        vy_odom = apply_scalar_deadband(vy_odom, VY_DEADBAND)
        wz_odom = apply_scalar_deadband(wz_odom, WZ_DEADBAND)

        self.ekf.predict_wheel(vx_odom, vy_odom, wz_odom, dt)

        gyro_z_meas = float(self.imu_state.gyroscope[2])
        yaw_raw = float(self.imu_state.rpy[2])

        if np.isfinite(gyro_z_meas):
            self.ekf.update_gyro_z(gyro_z_meas, wz_odom)

        if np.isfinite(yaw_raw):
            if self.yaw_zero is None:
                self.yaw_zero = yaw_raw
            yaw_meas = wrap_to_pi(yaw_raw - self.yaw_zero)

            if abs(wz_odom) < 0.2:
                self.ekf.update_yaw(yaw_meas)

        px, py, _yaw = self.ekf.pose
        vx, vy, wz   = self.ekf.global_velocity
        
        self.sport_mode_state.position[0] = px
        self.sport_mode_state.position[1] = py
        self.sport_mode_state.position[2] = 0.0
        self.sport_mode_state.velocity[0] = vx
        self.sport_mode_state.velocity[1] = vy
        self.sport_mode_state.yaw_speed   = wz
        self.sport_mode_state.imu_state   = self.imu_state
        self.sport_mode_state_pub.Write(self.sport_mode_state)

        self.last_loop_t = now

def main():
    ChannelFactoryInitialize()
    
    node = SportNode()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        node.lowCmdWriteThreadPtr.Wait()

if __name__ == '__main__':
    main()