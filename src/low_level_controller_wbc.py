from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import time
import numpy as np
import numpy.typing as npt

from .utils import quat_to_rotmat

from .pinnzoo_binding import PinnZooModel
from .pinnzoo import M_func, inverse_dynamics

from .openarm_idx import (
    OPENARM_NQ,
    OPENARM_NV,
    OPENARM_NU,
    OPENARM_U_BASE_VX,
    OPENARM_U_BASE_VY,
    OPENARM_U_BASE_WZ,
    OPENARM_WORLD_QUAT,
)

EnvType = Literal["sim", "real"]

@dataclass(slots=True)
class MITCommand:
    """MIT motor command arrays in actuator order."""

    motor_names: list[str]
    pos_des: npt.NDArray[np.float64]
    vel_des: npt.NDArray[np.float64]
    kp: npt.NDArray[np.float64]
    kd: npt.NDArray[np.float64]
    tau_ff: npt.NDArray[np.float64]

@dataclass(slots=True)
class LowLevelControllerConfig:
    env_type: EnvType = "sim"
    
    wheel_radius_m: float = 0.052
    fl_pos_xy_m: tuple[float, float] = ( 0.198,  0.13)
    fr_pos_xy_m: tuple[float, float] = ( 0.198, -0.13)
    rl_pos_xy_m: tuple[float, float] = (-0.198,  0.13)
    rr_pos_xy_m: tuple[float, float] = (-0.198, -0.13)

    min_wheel_speed_m_s: float = 1e-4

    wheel_vel_limit_rad_s: float = 30.0

    arm_acc_limit_rad_s2: float = 80.0
    
    base_steering_kp: float = 20.0
    base_steering_kd: float = 0.02
    base_wheel_kd: float = 5.0

    base_idle_linear_threshold_m_s: float = 1e-2
    base_idle_angular_threshold_rad_s: float = 5e-2
    
    arm_motor_kd: npt.NDArray[np.float64] | float = field(
        default_factory=lambda: np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0] * 2,
            dtype=np.float64,
        )
    )
    arm_motor_kp: npt.NDArray[np.float64] | float = field(
        default_factory=lambda: np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2,
            dtype=np.float64,
        )
    )


class LowLevelController:
    motor_names = [
        "FL_motor",
        "FR_motor",
        "RL_motor",
        "RR_motor",
        "FL_wheel",
        "FR_wheel",
        "RL_wheel",
        "RR_wheel",
        "L_1_motor",
        "L_2_motor",
        "L_3_motor",
        "L_4_motor",
        "L_5_motor",
        "L_6_motor",
        "L_7_motor",
        "gripper_LL_motor",
        "gripper_LR_motor",
        "R_1_motor",
        "R_2_motor",
        "R_3_motor",
        "R_4_motor",
        "R_5_motor",
        "R_6_motor",
        "R_7_motor",
        "gripper_RL_motor",
        "gripper_RR_motor",
    ]
    
    tau_static = np.array([
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.30,
        0.15,
        0.15,
    ] * 2)

    # qpos indices in configuration order.
    _steer_qpos_idx = np.array([7, 9, 11, 13], dtype=np.int32)
    _wheel_qpos_idx = np.array([8, 10, 12, 14], dtype=np.int32)
    _arm_qpos_idx = np.arange(15, 33, dtype=np.int32)

    # qvel indices in generalized velocity order.
    _steer_qvel_idx = np.array([6, 8, 10, 12], dtype=np.int32)
    _wheel_qvel_idx = np.array([7, 9, 11, 13], dtype=np.int32)
    _arm_qvel_idx = np.arange(14, 32, dtype=np.int32)

    _steer_act_idx = np.array([0, 1, 2, 3], dtype=np.int32)
    _wheel_act_idx = np.array([4, 5, 6, 7], dtype=np.int32)
    _arm_act_idx = np.arange(8, 26, dtype=np.int32)

    _arm_xml_tau_limit = np.array(
        [
            40.0,
            40.0,
            27.0,
            27.0,
            7.0,
            7.0,
            7.0,
            7.0,
            7.0,
            40.0,
            40.0,
            27.0,
            27.0,
            7.0,
            7.0,
            7.0,
            7.0,
            7.0,
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        config: LowLevelControllerConfig | None = None,
        lib_path: str = "include/libg7_openarm_quat.so",
    ) -> None:
        self.config = config if config is not None else LowLevelControllerConfig()
        self.lib_path = lib_path
        self.model = PinnZooModel(lib_path)

        self.num_motors = len(self.motor_names)
        self._prev_arm_vel_des = np.zeros(18, dtype=np.float64)

        self._wheel_xy = np.array(
            [
                self.config.fl_pos_xy_m,
                self.config.fr_pos_xy_m,
                self.config.rl_pos_xy_m,
                self.config.rr_pos_xy_m,
            ],
            dtype=np.float64,
        )
        
        self._tau_min = np.zeros(self.num_motors, dtype=np.float64)
        self._tau_max = np.zeros(self.num_motors, dtype=np.float64)
        self._tau_min[self._arm_act_idx] = -self._arm_xml_tau_limit
        self._tau_max[self._arm_act_idx] = self._arm_xml_tau_limit
        self.prev_time = time.perf_counter()
    
    def compute_arm_kd_from_mass_matrix(
        self,
        x: npt.NDArray[np.float64],
        zeta=0.7,
        omega=8.0,
    ):
        M = M_func(self.model, x)
        M_diag = np.diag(M)
        Kd = 2.0 * zeta * omega * np.sqrt(M_diag)
        return Kd[self._arm_qvel_idx]

    def _zero_command(self) -> MITCommand:
        zeros = np.zeros(self.num_motors, dtype=np.float64)
        return MITCommand(
            motor_names=self.motor_names.copy(),
            pos_des=zeros.copy(),
            vel_des=zeros.copy(),
            kp=zeros.copy(),
            kd=zeros.copy(),
            tau_ff=zeros.copy(),
        )

    def update(
        self,
        qpos: npt.NDArray[np.float64],
        qvel: npt.NDArray[np.float64],
        u_des: npt.NDArray[np.float64]
    ) -> MITCommand:
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        u_des = np.asarray(u_des, dtype=np.float64)

        if qpos.shape != (OPENARM_NQ,):
            raise ValueError(f"qpos must have shape ({OPENARM_NQ},), got {qpos.shape}")
        if qvel.shape != (OPENARM_NV,):
            raise ValueError(f"qvel must have shape ({OPENARM_NV},), got {qvel.shape}")
        if u_des.shape != (OPENARM_NU,):
            raise ValueError(f"u_des must have shape ({OPENARM_NU},), got {u_des.shape}")
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)) or not np.all(np.isfinite(u_des)):
            self._prev_arm_vel_des[:] = 0.0
            return self._zero_command()
        
        arm_vel_des = u_des[3:]

        base_command_is_idle = self._is_base_command_idle(u_des)
        
        if base_command_is_idle:
            steer_pos_des = qpos[self._steer_qpos_idx].copy()
            wheel_vel_des = np.zeros((4,), dtype=np.float64)
        else:
            steer_pos_des, wheel_vel_des = self._base_velocity_to_swerve_targets(
                qpos=qpos,
                u_cmd=u_des,
            )

        acc_act_des = self._build_desired_actuator_acceleration(
            qpos=qpos,
            qvel=qvel,
            arm_vel_des=arm_vel_des,
        )
        
        x = np.concatenate([qpos, qvel])
        acc_des = np.concatenate([np.zeros(6, dtype=np.float64), acc_act_des])
        tau_act = inverse_dynamics(
            model=self.model,
            x=x,
            vdot=acc_des,
        )[6:]
        
        if not np.all(np.isfinite(tau_act)):
            self._prev_arm_vel_des[:] = 0.0
            return self._zero_command()
        
        want_move = np.abs(u_des[3:]) > 4e-2
        tau_bias = self.tau_static * want_move * np.sign(u_des[3:])
        tau_act[8:] += tau_bias

        tau_act[self._steer_act_idx] = 0.0
        tau_act[self._wheel_act_idx] = 0.0
        tau_act = np.clip(tau_act, self._tau_min, self._tau_max)

        pos_des = np.zeros(self.num_motors, dtype=np.float64)
        vel_des = np.zeros(self.num_motors, dtype=np.float64)
        kp = np.zeros(self.num_motors, dtype=np.float64)
        kd = np.zeros(self.num_motors, dtype=np.float64)

        pos_des[self._steer_act_idx] = steer_pos_des
        vel_des[self._wheel_act_idx] = wheel_vel_des
        vel_des[self._arm_act_idx] = arm_vel_des

        if not base_command_is_idle:
            kp[self._steer_act_idx] = self.config.base_steering_kp
            kd[self._steer_act_idx] = self.config.base_steering_kd
            kd[self._wheel_act_idx] = self.config.base_wheel_kd
        kp[self._arm_act_idx] = self.config.arm_motor_kp
        kd[self._arm_act_idx] = self.config.arm_motor_kd

        return MITCommand(
            motor_names=self.motor_names.copy(),
            pos_des=pos_des,
            vel_des=vel_des,
            kp=kp,
            kd=kd,
            tau_ff=tau_act,
        )

    def _is_base_command_idle(
        self,
        u_cmd: npt.NDArray[np.float64]
    ) -> bool:
        linear_speed = float(np.hypot(
            u_cmd[OPENARM_U_BASE_VX],
            u_cmd[OPENARM_U_BASE_VY],
        ))
        angular_speed = abs(float(u_cmd[OPENARM_U_BASE_WZ]))
        
        return (
            linear_speed < self.config.base_idle_linear_threshold_m_s
            and angular_speed < self.config.base_idle_angular_threshold_rad_s
        )

    def _base_velocity_to_swerve_targets(
        self,
        qpos: npt.NDArray[np.float64],
        u_cmd: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        vx = float(u_cmd[OPENARM_U_BASE_VX])
        vy = float(u_cmd[OPENARM_U_BASE_VY])
        wz = float(u_cmd[OPENARM_U_BASE_WZ])

        current_steer = qpos[self._steer_qpos_idx]
        steer_pos_des = current_steer.copy()
        wheel_vel_des = np.zeros(4, dtype=np.float64)

        for i, (wheel_x, wheel_y) in enumerate(self._wheel_xy):
            wheel_vx = vx - wz * wheel_y
            wheel_vy = vy + wz * wheel_x
            wheel_speed = float(np.hypot(wheel_vx, wheel_vy))

            if wheel_speed < self.config.min_wheel_speed_m_s:
                continue
            
            steer_pos_des[i] = float(np.arctan2(wheel_vy, wheel_vx))
            wheel_vel_des[i] = wheel_speed / self.config.wheel_radius_m
            
            angle_err = self._wrap_to_pi(steer_pos_des[i] - current_steer[i])
            if abs(angle_err) > 0.5 * np.pi:
                steer_pos_des[i] = self._wrap_to_pi(steer_pos_des[i] + np.pi)
                wheel_vel_des[i] *= -1.0
                
        wheel_vel_des = np.clip(
            wheel_vel_des,
            -self.config.wheel_vel_limit_rad_s,
            self.config.wheel_vel_limit_rad_s,
        )

        return steer_pos_des, wheel_vel_des

    def _build_desired_actuator_acceleration(
        self,
        qpos: npt.NDArray[np.float64],
        qvel: npt.NDArray[np.float64],
        arm_vel_des: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        dt = time.perf_counter() - self.prev_time
        arm_acc_ff = (arm_vel_des - self._prev_arm_vel_des) / dt
        self._prev_arm_vel_des[:] = arm_vel_des
        self.prev_time = time.perf_counter()
        
        arm_vel_err = arm_vel_des - qvel[self._arm_qvel_idx]
        
        x = np.concatenate([qpos, qvel])
        
        KDs = self.compute_arm_kd_from_mass_matrix(
            x,
            zeta=0.7,
            omega=8.0,
        )
        
        acc = np.zeros(self.num_motors, dtype=np.float64)
        acc[self._arm_act_idx] = (
            arm_acc_ff + KDs * arm_vel_err
        )
        acc[self._arm_act_idx] = np.clip(
            acc[self._arm_act_idx],
            -self.config.arm_acc_limit_rad_s2,
            self.config.arm_acc_limit_rad_s2,
        )
        return acc

    @staticmethod
    def _wrap_to_pi(angle: npt.NDArray[np.float64] | float):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi