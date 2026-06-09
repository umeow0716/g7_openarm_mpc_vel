import time
import numpy as np
import numpy.typing as npt

from typing import Optional, TYPE_CHECKING, cast, MutableSequence

from ..mpc_solver import OpenArmMPCSolver

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize

from .msg.mid_cmd import MidCmd
from .msg.target_msg import TargetMsg
from .actuator_mapping import ACTUATOR_TO_MODEL_JOINT_ORDER

MPC_RATE_HZ = 50.0

if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorState_

class MidControllerNode:
    def __init__(self, sport_mode_state_topic: str = "rt/sportmodestate", low_state_topic: str = "rt/lowstate", mid_cmd_topic: str = "rt/midcmd", target_topic: str = "rt/targetmsg"):
        self.sport_mode_state_subscriber = ChannelSubscriber(sport_mode_state_topic, SportModeState_)
        self.sport_mode_state_subscriber.Init(self.sport_mode_state_callback, 10)
        
        self.low_state_sub = ChannelSubscriber(low_state_topic, LowState_)
        self.low_state_sub.Init(self.low_state_callback, 10)
        
        self.target_sub = ChannelSubscriber(target_topic, TargetMsg)
        self.target_sub.Init(self.target_callback, 10)
        
        self.mid_cmd_pub = ChannelPublisher(mid_cmd_topic, MidCmd)
        self.mid_cmd_pub.Init()
        
        self.mpc_solver = OpenArmMPCSolver()
        self.x = np.zeros((self.mpc_solver.nx,), dtype=np.float64)
        
        self.sport_state: Optional[SportModeState_] = None
        self.low_state: Optional[LowState_] = None
        self.target_msg: Optional[TargetMsg] = None
        
    def target_callback(self, msg: TargetMsg):
        self.target_msg = msg
    
    def sport_mode_state_callback(self, msg: SportModeState_):
        self.sport_state = msg

    def low_state_callback(self, msg: LowState_):
        self.low_state = msg
    
    def send_mid_cmd(self, u_cmd: npt.NDArray[np.float64]):
        mid_cmd = MidCmd(u_cmd.tolist())
        self.mid_cmd_pub.Write(mid_cmd)
    
    def make_state(self):
        assert self.sport_state is not None and self.low_state is not None
        
        imu_state: IMUState_ = self.sport_state.imu_state 
        motor_state = cast(MutableSequence['MotorState_'], self.low_state.motor_state)
            
        pos = np.array(self.sport_state.position, dtype=np.float64) 
        quat = np.array(imu_state.quaternion, dtype=np.float64) 
        motor_q_actuator_order = np.array(
            [motor_state[i].q for i in range(26)], 
            dtype=np.float64,
        )
        motor_q_model_order = motor_q_actuator_order[ACTUATOR_TO_MODEL_JOINT_ORDER]

        return np.concatenate([pos, quat, motor_q_model_order])
    
    def make_target(self):
        assert self.target_msg is not None
        
        left_pos = np.array([self.target_msg.left.position.x, self.target_msg.left.position.y, self.target_msg.left.position.z], dtype=np.float64) 
        left_quat = np.array([self.target_msg.left.orientation.w, self.target_msg.left.orientation.x, self.target_msg.left.orientation.y, self.target_msg.left.orientation.z], dtype=np.float64) 
        right_pos = np.array([self.target_msg.right.position.x, self.target_msg.right.position.y, self.target_msg.right.position.z], dtype=np.float64) 
        right_quat = np.array([self.target_msg.right.orientation.w, self.target_msg.right.orientation.x, self.target_msg.right.orientation.y, self.target_msg.right.orientation.z], dtype=np.float64) 
        
        return np.concatenate([
            left_pos, left_quat,
            right_pos, right_quat
        ]).astype(np.float64)

    def solve_loop(self):
        U_init = np.zeros((self.mpc_solver.N, self.mpc_solver.nu), dtype=np.float64)
        max_iter = 10
        u_cmd = np.zeros((self.mpc_solver.nu,), dtype=np.float64)
        period = 1.0 / MPC_RATE_HZ
        next_t = time.perf_counter()
        
        while True:
            if self.sport_state is None or self.low_state is None or self.target_msg is None:
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t = now + period

            x = self.make_state()
            target = self.make_target()
            
            assert x.shape == (self.mpc_solver.nx,)
            assert target.shape == (14,)

            _, U_sol, success = self.mpc_solver.solve_slq(
                x0=x,
                target=target,
                U_init=U_init,
                max_iter=max_iter,
                shift_warm_start=True,
            )
            
            if success and np.all(np.isfinite(U_sol)):
                U_init = U_sol.copy()
                u_cmd = U_sol[0].copy()
            else:
                U_init[:] = 0.0
                u_cmd = np.zeros((self.mpc_solver.nu,), dtype=np.float64)
 
            self.send_mid_cmd(u_cmd)

def main():
    ChannelFactoryInitialize()
    
    node = MidControllerNode()
    node.solve_loop()