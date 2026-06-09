import time
import numpy as np
import mujoco
import mujoco.viewer

from typing import TYPE_CHECKING, Optional, MutableSequence, cast

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default, unitree_go_msg_dds__SportModeState_ as SportModeState_default

from .msg.mid_cmd import MidCmd
from .msg.target_msg import TargetMsg, TargetMsg_default
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorState_, MotorCmd_

from typing import Literal

EnvType = Literal["real", "sim"]

LEFT_EE_BODY = 'L_gripper_tcp_link'
RIGHT_EE_BODY = 'R_gripper_tcp_link'

from .actuator_mapping import ACTUATOR_QPOS_IDX, ACTUATOR_QVEL_IDX

def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

class SimulationNode:
    def __init__(self, env_type: EnvType, scene_file='g7_openarm_mujoco/scene.xml', sport_mode_state_topic: str = "rt/sportmodestate", low_state_topic: str = "rt/lowstate", low_cmd_topic: str = "rt/lowcmd", target_msg_topic: str = "rt/targetmsg", mid_cmd_topic: str = "rt/midcmd"):
        self.env_type = env_type
        self.scene_file = scene_file
        
        self.low_cmd_sub: Optional[ChannelSubscriber] = None
        self.mid_cmd_sub: Optional[ChannelSubscriber] = None
        self.low_state_sub: Optional[ChannelSubscriber] = None
        self.sport_mode_state_sub: Optional[ChannelSubscriber] = None
        self.low_state_pub: Optional[ChannelPublisher] = None
        self.sport_mode_state_pub: Optional[ChannelPublisher] = None
        
        # In sim mode this node is the SportModeState publisher (ground-truth state).
        # In real mode this node never publishes SportModeState; SportNode should be used
        # there to estimate and publish rt/sportmodestate from LowState + IMUState.
        self.target_msg_pub: ChannelPublisher = ChannelPublisher(target_msg_topic, TargetMsg)
        self.target_msg_pub.Init()

        self.sim_init()
        self.qpos_sub = self.data.qpos.copy()
        self.qvel_sub = self.data.qvel.copy()
        self.low_cmd: Optional[LowCmd_] = None
        self.mid_cmd: Optional[MidCmd]  = None

        if self.env_type == "sim":
            self.low_cmd_sub = ChannelSubscriber(low_cmd_topic, LowCmd_)
            self.low_cmd_sub.Init(self.low_cmd_callback, 10)
            self.mid_cmd_sub = ChannelSubscriber(mid_cmd_topic, MidCmd)
            self.mid_cmd_sub.Init(self.mid_cmd_callback, 10)
            self.low_state_pub = ChannelPublisher(low_state_topic, LowState_)
            self.low_state_pub.Init()
            self.sport_mode_state_pub = ChannelPublisher(sport_mode_state_topic, SportModeState_)
            self.sport_mode_state_pub.Init()
        elif self.env_type == "real":
            self.low_state_sub = ChannelSubscriber(low_state_topic, LowState_)
            self.low_state_sub.Init(self.low_state_callback, 10)
            self.sport_mode_state_sub = ChannelSubscriber(sport_mode_state_topic, SportModeState_)
            self.sport_mode_state_sub.Init(self.sport_mode_state_callback, 10)
        else:
            raise ValueError(f"Invalid env_type: {self.env_type}. Must be 'real' or 'sim'.")
        
    def low_cmd_callback(self, msg: LowCmd_):
        self.low_cmd = msg
    
    def mid_cmd_callback(self, msg: MidCmd):
        self.mid_cmd = msg

    def low_state_callback(self, msg: LowState_):
        motor_state = cast(MutableSequence['MotorState_'], msg.motor_state)
        self.qpos_sub[ACTUATOR_QPOS_IDX] = [state.q for state in motor_state[:26]]
        self.qvel_sub[ACTUATOR_QVEL_IDX] = [state.dq for state in motor_state[:26]]

    def sport_mode_state_callback(self, msg: SportModeState_):
        msg_pos  = cast(MutableSequence[float], msg.position)
        msg_vel  = cast(MutableSequence[float], msg.velocity)
        msg_quat = cast(MutableSequence[float], msg.imu_state.quaternion)
        msg_gyro = cast(MutableSequence[float], msg.imu_state.gyroscope)
        msg.imu_state
        
        self.qpos_sub[0] = float(msg_pos[0])
        self.qpos_sub[1] = float(msg_pos[1])
        self.qpos_sub[2] = float(msg_pos[2])
        self.qpos_sub[3] = float(msg_quat[0])
        self.qpos_sub[4] = float(msg_quat[1])
        self.qpos_sub[5] = float(msg_quat[2])
        self.qpos_sub[6] = float(msg_quat[3])

        self.qvel_sub[0] = float(msg_vel[0])
        self.qvel_sub[1] = float(msg_vel[1])
        self.qvel_sub[2] = float(msg_vel[2])
        self.qvel_sub[3] = float(msg_gyro[0])
        self.qvel_sub[4] = float(msg_gyro[1])
        self.qvel_sub[5] = float(msg_gyro[2])
    
    def send_state(self):
        assert self.env_type == 'sim' and self.sport_mode_state_pub and self.low_state_pub
        sport_mode_state = SportModeState_default()
        
        pos  = cast(MutableSequence[float], sport_mode_state.position)
        vel  = cast(MutableSequence[float], sport_mode_state.velocity)
        quat = cast(MutableSequence[float], sport_mode_state.imu_state.quaternion)
        gyro = cast(MutableSequence[float], sport_mode_state.imu_state.gyroscope)
        
        pos[0] = float(self.data.qpos[0])
        pos[1] = float(self.data.qpos[1])
        pos[2] = float(self.data.qpos[2])
        vel[0] = float(self.data.qvel[0])
        vel[1] = float(self.data.qvel[1])
        vel[2] = float(self.data.qvel[2])
        quat[0] = float(self.data.qpos[3])
        quat[1] = float(self.data.qpos[4])
        quat[2] = float(self.data.qpos[5])
        quat[3] = float(self.data.qpos[6])
        gyro[0] = float(self.data.qvel[3])
        gyro[1] = float(self.data.qvel[4])
        gyro[2] = float(self.data.qvel[5])
        sport_mode_state.yaw_speed = float(self.data.qvel[5])
        
        low_state = LowState_default()
        motor_state = cast(MutableSequence['MotorState_'], low_state.motor_state)
        for i in range(26):
            motor_state[i].q = float(self.data.qpos[ACTUATOR_QPOS_IDX[i]])
            motor_state[i].dq = float(self.data.qvel[ACTUATOR_QVEL_IDX[i]])
        
        self.sport_mode_state_pub.Write(sport_mode_state)
        self.low_state_pub.Write(low_state)
    
    def send_target(self):
        left_target_pos = self.data.body('left_target').xpos.copy()
        left_target_quat = self.data.body('left_target').xquat.copy()
        
        right_target_pos = self.data.body('right_target').xpos.copy()
        right_target_quat = self.data.body('right_target').xquat.copy()
        
        target_msg = TargetMsg_default()
        
        target_msg.left.position.x = left_target_pos[0]
        target_msg.left.position.y = left_target_pos[1]
        target_msg.left.position.z = left_target_pos[2]
        target_msg.left.orientation.w = left_target_quat[0]
        target_msg.left.orientation.x = left_target_quat[1]
        target_msg.left.orientation.y = left_target_quat[2]
        target_msg.left.orientation.z = left_target_quat[3]
        
        target_msg.right.position.x = right_target_pos[0]
        target_msg.right.position.y = right_target_pos[1]
        target_msg.right.position.z = right_target_pos[2]
        target_msg.right.orientation.w = right_target_quat[0]
        target_msg.right.orientation.x = right_target_quat[1]
        target_msg.right.orientation.y = right_target_quat[2]
        target_msg.right.orientation.z = right_target_quat[3]
        
        self.target_msg_pub.Write(target_msg)

    def get_motor_des(self):
        assert self.low_cmd is not None
        
        kp     = np.zeros((26,), dtype=np.float64)
        kd     = np.zeros((26,), dtype=np.float64)
        q_des  = np.zeros((26,), dtype=np.float64)
        dq_des = np.zeros((26,), dtype=np.float64)
        tau_ff = np.zeros((26,), dtype=np.float64)
        
        motor_cmd = cast(MutableSequence['MotorCmd_'], self.low_cmd.motor_cmd)
        for i, cmd in enumerate(motor_cmd[:26]):
            kp[i]     = cmd.kp
            kd[i]     = cmd.kd
            q_des[i]  = cmd.q
            dq_des[i] = cmd.dq
            tau_ff[i] = cmd.tau
        return kp, kd, q_des, dq_des, tau_ff
    
    def sim_loop(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                cycle_end_time = time.time() + self.model.opt.timestep
                
                if self.env_type == "sim" and self.low_cmd is not None:
                    kp, kd, q_des, dq_des, tau_ff = self.get_motor_des()
                    q_err  = q_des  - self.data.qpos[ACTUATOR_QPOS_IDX]
                    dq_err = dq_des - self.data.qvel[ACTUATOR_QVEL_IDX]
                    q_err[:4] = wrap_to_pi(q_err[:4])

                    self.data.ctrl[:] = kp * q_err + kd * dq_err + tau_ff
                if self.env_type == "sim":
                    mujoco.mj_step(self.model, self.data)
                    self.send_state()
                elif self.env_type == "real":
                    self.data.qpos[:] = self.qpos_sub[:]
                    self.data.qvel[:] = self.qvel_sub[:]
                    mujoco.mj_forward(self.model, self.data)

                viewer.sync()
                self.send_target()
                
                now = time.time()
                sleep_time = cycle_end_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    def sim_init(self):
        self.spec = mujoco.MjSpec.from_file(self.scene_file)
        
        self.left_target = self.spec.worldbody.add_body(
            name='left_target',
            mocap=True,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        self.left_target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05],
            rgba=[1, 0, 0, 0.3],
            contype=0,
            conaffinity=0,
        )

        self.right_target = self.spec.worldbody.add_body(
            name='right_target',
            mocap=True,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        self.right_target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05],
            rgba=[0, 0, 1, 0.3],
            contype=0,
            conaffinity=0,
        )
        
        self.model = self.spec.compile()
        self.data  = mujoco.MjData(self.model)
        
        mujoco.mj_forward(self.model, self.data)
        
        left_hand_pos = self.data.body(LEFT_EE_BODY).xpos.copy()
        right_hand_pos = self.data.body(RIGHT_EE_BODY).xpos.copy()
        left_hand_quat = self.data.body(LEFT_EE_BODY).xquat.copy()
        right_hand_quat = self.data.body(RIGHT_EE_BODY).xquat.copy()
        
        left_target_mocap_id = self.model.body_mocapid[self.model.body('left_target').id]
        right_target_mocap_id = self.model.body_mocapid[self.model.body('right_target').id]

        self.data.mocap_pos[left_target_mocap_id] = left_hand_pos
        self.data.mocap_quat[left_target_mocap_id] = left_hand_quat
        self.data.mocap_pos[right_target_mocap_id] = right_hand_pos
        self.data.mocap_quat[right_target_mocap_id] = right_hand_quat

def main(env_type: EnvType = 'sim'):
    ChannelFactoryInitialize()
    
    node = SimulationNode(env_type)
    node.sim_loop()

if __name__ == '__main__':
    main()