import time
import numpy as np

from typing import TYPE_CHECKING, Optional

from ..low_level_controller_wbc import LowLevelController

from .msg.mid_cmd import MidCmd
from .actuator_mapping import ACTUATOR_TO_MODEL_JOINT_ORDER

from typing import MutableSequence, cast
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmd_default
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_, MotorState_

LOW_CTRL_RATE_HZ = 100.0

class LowControllerNode:
    def __init__(self,
        sport_mode_state_topic: str = "rt/sportmodestate",
        low_state_topic: str = "rt/lowstate",
        mid_cmd_topic: str = "rt/midcmd",
        low_cmd_topic: str = "rt/lowcmd"
    ):
        self.sport_mode_state_subscriber = ChannelSubscriber(sport_mode_state_topic, SportModeState_)
        self.sport_mode_state_subscriber.Init(self.sport_mode_state_callback, 10)

        self.low_state_sub = ChannelSubscriber(low_state_topic, LowState_)
        self.low_state_sub.Init(self.low_state_callback, 10)

        self.mid_cmd_sub = ChannelSubscriber(mid_cmd_topic, MidCmd)
        self.mid_cmd_sub.Init(self.mid_cmd_callback, 10)

        self.low_cmd_pub = ChannelPublisher(low_cmd_topic, LowCmd_)
        self.low_cmd_pub.Init()
        
        self.low_state: Optional[LowState_] = None
        self.sport_state: Optional[SportModeState_] = None
        self.mid_cmd: Optional[MidCmd] = None
        
        self.controller = LowLevelController()
    
    def sport_mode_state_callback(self, msg: SportModeState_):
        self.sport_state = msg
        
    def low_state_callback(self, msg: LowState_):
        self.low_state = msg
    
    def mid_cmd_callback(self, msg: MidCmd):
        self.mid_cmd = msg

    def make_q(self):
        assert self.sport_state is not None and self.low_state is not None
        
        imu_state: 'IMUState_' = self.sport_state.imu_state
        motor_state = cast(MutableSequence['MotorState_'], self.low_state.motor_state)
            
        pos = np.array(self.sport_state.position, dtype=np.float64) 
        quat = np.array(imu_state.quaternion, dtype=np.float64) 
        motor_q_actuator_order = np.array(
            [motor_state[i].q for i in range(26)],
            dtype=np.float64,
        )
        motor_q_model_order = motor_q_actuator_order[ACTUATOR_TO_MODEL_JOINT_ORDER]
        
        return np.concatenate([pos, quat, motor_q_model_order])
    
    def make_dq(self):
        assert self.sport_state is not None and self.low_state is not None
        
        imu_state: 'IMUState_' = self.sport_state.imu_state 
        motor_state = cast(MutableSequence['MotorState_'], self.low_state.motor_state)
        
        vel = np.array(self.sport_state.velocity, dtype=np.float64) 
        omega = np.array(imu_state.gyroscope, dtype=np.float64) 
        motor_dq_actuator_order = np.array(
            [motor_state[i].dq for i in range(26)], 
            dtype=np.float64,
        )
        motor_dq_model_order = motor_dq_actuator_order[ACTUATOR_TO_MODEL_JOINT_ORDER]
        
        return np.concatenate([vel, omega, motor_dq_model_order])
    
    def control_loop(self):
        period = 1.0 / LOW_CTRL_RATE_HZ
        next_t = time.perf_counter()

        while True:
            if self.sport_state is None or self.low_state is None or self.mid_cmd is None:
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t = now + period
            
            qpos = self.make_q()
            qvel = self.make_dq()
            u    = np.array(self.mid_cmd.u, dtype=np.float64) 
            
            assert qpos.shape == (33,) and np.isfinite(qpos).all()
            assert qvel.shape == (32,) and np.isfinite(qvel).all()
            assert u.shape    == (21,) and np.isfinite(u).all()
            
            cmd  = self.controller.update(qpos, qvel, u)
            
            low_cmd = LowCmd_default()
            motor_cmd = cast(MutableSequence['MotorCmd_'], low_cmd.motor_cmd)
            
            for i in range(26):
                motor_cmd[i].q   = cmd.pos_des[i] 
                motor_cmd[i].dq  = cmd.vel_des[i] 
                motor_cmd[i].kp  = cmd.kp[i] 
                motor_cmd[i].kd  = cmd.kd[i] 
                motor_cmd[i].tau = cmd.tau_ff[i] 
            
            self.low_cmd_pub.Write(low_cmd)

def main():
    ChannelFactoryInitialize()
    node = LowControllerNode()
    node.control_loop()
    
if __name__ == '__main__':
    main()