from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from .pinnzoo_binding import PinnZooModel
from .pinnzoo import M_func, inverse_dynamics

from .openarm_idx import (
    OPENARM_NQ,
    OPENARM_NV,
    OPENARM_NU,
    OPENARM_WORLD_QUAT,
    OPENARM_U_BASE_VX,
    OPENARM_U_BASE_VY,
    OPENARM_U_BASE_WZ,
)
from .utils import quat_to_rotmat


FloatArray = npt.NDArray[np.float64]
VelocityFrame = Literal["body", "world"]
EnvType = Literal["sim", "real"]


@dataclass(slots=True)
class MITCommand:
    """MIT motor command arrays in actuator order."""

    motor_names: list[str]
    pos_des: FloatArray
    vel_des: FloatArray
    kp: FloatArray
    kd: FloatArray
    tau_ff: FloatArray

@dataclass(slots=True)
class LowLevelControllerConfig:
    dt: float = 0.01
    base_velocity_frame: VelocityFrame = "world"
    env_type: EnvType = "sim"
    
    wheel_radius_m: float = 0.052
    fl_pos_xy_m: tuple[float, float] = (0.198, 0.13)
    fr_pos_xy_m: tuple[float, float] = (0.198, -0.13)
    rl_pos_xy_m: tuple[float, float] = (-0.198, 0.13)
    rr_pos_xy_m: tuple[float, float] = (-0.198, -0.13)

    min_module_speed_m_s: float = 1e-4

    steering_vel_limit_rad_s: float = 6.0
    wheel_vel_limit_rad_s: float = 30.0

    arm_acc_limit_rad_s2: float = 80.0
    
    base_steering_kp: float = 20.0
    base_steering_kd: float = 0.02
    base_wheel_kd: float = 5.0

    base_idle_linear_threshold_m_s: float = 1e-3
    base_idle_angular_threshold_rad_s: float = 1e-3
    
    arm_motor_kd: FloatArray | float = field(
        default_factory=lambda: np.array(
            [8.0, 8.0, 6.0, 6.0, 2.0, 3.0, 1.5, 0.0, 0.0] * 2,
            dtype=np.float64,
        )
    )
    arm_motor_kp: FloatArray | float = field(
        default_factory=lambda: np.array(
            [3.0, 3.0, 2.5, 2.5, 1.0, 1.0, 0.8, 0.0, 0.0] * 2,
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

    # qpos indices in configuration order.
    _steer_qpos_idx = np.array([7, 9, 11, 13], dtype=np.int32)
    _wheel_qpos_idx = np.array([8, 10, 12, 14], dtype=np.int32)
    _arm_qpos_idx = np.arange(15, 33, dtype=np.int32)

    # qvel indices in generalized velocity order.
    _steer_qvel_idx = np.array([6, 8, 10, 12], dtype=np.int32)
    _wheel_qvel_idx = np.array([7, 9, 11, 13], dtype=np.int32)
    _arm_qvel_idx = np.arange(14, 32, dtype=np.int32)

    # Actuator order -> generalized velocity row index.
    _actuated_v_idx = np.array(
        [6, 8, 10, 12, 7, 9, 11, 13, *range(14, 32)],
        dtype=np.int32,
    )

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
        self._initialized = False

        self._module_xy = np.array(
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

        self._selection_act = np.zeros((self.num_motors, OPENARM_NV), dtype=np.float64)
        self._selection_act[np.arange(self.num_motors), self._actuated_v_idx] = 1.0
    
    def compute_kd_from_mass_matrix(
        self,
        x: FloatArray,
        zeta=0.7,
        omega_n=8.0,
        per_joint_omega=None,
    ):
        M = M_func(self.model, x)
        M_diag = np.diag(M)
        
        if per_joint_omega is not None:
            omega = np.asarray(per_joint_omega)
        else:
            omega = omega_n
            
        Kd = 2.0 * zeta * omega * np.sqrt(M_diag)
        
        return Kd

    def reset(self) -> None:
        self._prev_arm_vel_des[:] = 0.0
        self._initialized = False

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

    def update(self, qpos: FloatArray, qvel: FloatArray, u_des: FloatArray) -> MITCommand:
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
            self.reset()
            return self._zero_command()

        base_command_is_idle = self._is_base_command_idle(u_des)
        steer_pos_des, steer_vel_des, wheel_vel_des = self._base_velocity_to_swerve_targets(
            qpos=qpos,
            u_cmd=u_des,
        )
        
        if base_command_is_idle:
            steer_pos_des = qpos[self._steer_qpos_idx].copy()
            steer_vel_des = np.zeros_like(steer_vel_des)
            wheel_vel_des = np.zeros_like(wheel_vel_des)

        arm_vel_des = u_des[3:]

        if (
            not np.all(np.isfinite(steer_pos_des))
            or not np.all(np.isfinite(steer_vel_des))
            or not np.all(np.isfinite(wheel_vel_des))
            or not np.all(np.isfinite(arm_vel_des))
        ):
            self.reset()
            return self._zero_command()

        desired_act_acc = self._build_desired_actuator_acceleration(
            qpos=qpos,
            qvel=qvel,
            arm_vel_des=u_des[3:],
        )
        
        x = np.concatenate([qpos, qvel])
        desired_vdot = np.concatenate([np.zeros(6, dtype=np.float64), desired_act_acc])
        tau_act = inverse_dynamics(
            model=self.model,
            x=x,
            vdot=desired_vdot,
        )[6:]
        
        tau0 = np.array([
            0.15,   # joint 1
            0.15,   # joint 2
            0.15,   # joint 3
            0.15,   # joint 4
            0.15,   # joint 5
            0.15,   # joint 6
            0.30,  # joint 7
            0.15,    # joint 8
            0.15,    # joint 9
        ] * 2)
        
        is_stuck = np.abs(u_des[3:]) > 3e-2

        tau_bias = tau0 * is_stuck * np.sign(u_des[3:])
        tau_act[8:] += tau_bias
        
        if not np.all(np.isfinite(tau_act)):
            self.reset()
            return self._zero_command()

        # Base is controlled by MIT kp/kd feedback in the returned command.
        # Do not precompute base PID torque into tau_ff; otherwise the gains are
        # evaluated against stale state before sim_viewer.py applies the command.
        tau_act[self._steer_act_idx] = 0.0
        tau_act[self._wheel_act_idx] = 0.0
        tau_act = np.clip(tau_act, self._tau_min, self._tau_max)

        self._prev_arm_vel_des[:] = arm_vel_des

        pos_des = np.zeros(self.num_motors, dtype=np.float64)
        vel_des = np.zeros(self.num_motors, dtype=np.float64)
        kp = np.zeros(self.num_motors, dtype=np.float64)
        kd = np.zeros(self.num_motors, dtype=np.float64)

        pos_des[self._steer_act_idx] = steer_pos_des
        vel_des[self._steer_act_idx] = steer_vel_des
        pos_des[self._wheel_act_idx] = qpos[self._wheel_qpos_idx]
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

    def _is_base_command_idle(self, u_cmd: FloatArray) -> bool:
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
        qpos: FloatArray,
        u_cmd: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        vx = float(u_cmd[OPENARM_U_BASE_VX])
        vy = float(u_cmd[OPENARM_U_BASE_VY])
        wz = float(u_cmd[OPENARM_U_BASE_WZ])

        if self.config.base_velocity_frame == "world":
            rot_world_from_body = quat_to_rotmat(qpos[OPENARM_WORLD_QUAT])
            v_body = rot_world_from_body.T @ np.array([vx, vy, 0.0], dtype=np.float64)
            vx_body = float(v_body[0])
            vy_body = float(v_body[1])
        elif self.config.base_velocity_frame == "body":
            vx_body = vx
            vy_body = vy
        else:
            raise ValueError(f"Unsupported base_velocity_frame: {self.config.base_velocity_frame}")

        current_steer = qpos[self._steer_qpos_idx]
        steer_pos_des = np.empty(4, dtype=np.float64)
        wheel_vel_des = np.empty(4, dtype=np.float64)

        for i, (module_x, module_y) in enumerate(self._module_xy):
            module_vx = vx_body - wz * module_y
            module_vy = vy_body + wz * module_x
            module_speed = float(np.hypot(module_vx, module_vy))

            if module_speed < self.config.min_module_speed_m_s:
                target_angle = float(current_steer[i])
                target_wheel_vel = 0.0
            else:
                raw_angle = float(np.arctan2(module_vy, module_vx))
                raw_wheel_vel = module_speed / self.config.wheel_radius_m
                target_angle, target_wheel_vel = self._optimize_steering_angle(
                    raw_angle=raw_angle,
                    raw_wheel_vel=raw_wheel_vel,
                    current_angle=float(current_steer[i]),
                )

            steer_pos_des[i] = target_angle
            wheel_vel_des[i] = target_wheel_vel

        wheel_vel_des = np.clip(
            wheel_vel_des,
            -self.config.wheel_vel_limit_rad_s,
            self.config.wheel_vel_limit_rad_s,
        )

        steering_error = self._wrap_to_pi(steer_pos_des - current_steer)
        steer_vel_des = np.clip(
            steering_error / max(self.config.dt, 1e-9),
            -self.config.steering_vel_limit_rad_s,
            self.config.steering_vel_limit_rad_s,
        )

        return steer_pos_des, steer_vel_des, wheel_vel_des

    def _build_desired_actuator_acceleration(
        self,
        qpos: FloatArray,
        qvel: FloatArray,
        arm_vel_des: FloatArray,
    ) -> FloatArray:
        acc = np.zeros(self.num_motors, dtype=np.float64)

        arm_acc_ff = (arm_vel_des - self._prev_arm_vel_des) / max(self.config.dt, 1e-9)
        arm_vel_err = arm_vel_des - qvel[self._arm_qvel_idx]
        
        x = np.concatenate([qpos, qvel])
        
        KDs = self.compute_kd_from_mass_matrix(
            x,
            zeta=0.7,
            omega_n=8.0,
        )
        
        arm_kd = KDs[6+8:6+8+18]
        
        acc[self._arm_act_idx] = (
            arm_acc_ff + arm_kd * arm_vel_err
        )
        acc[self._arm_act_idx] = np.clip(
            acc[self._arm_act_idx],
            -self.config.arm_acc_limit_rad_s2,
            self.config.arm_acc_limit_rad_s2,
        )

        if not np.all(np.isfinite(acc)):
            return np.zeros(self.num_motors, dtype=np.float64)
        return acc

    @staticmethod
    def _wrap_to_pi(angle: FloatArray | float) -> FloatArray | float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _optimize_steering_angle(
        self,
        raw_angle: float,
        raw_wheel_vel: float,
        current_angle: float,
    ) -> tuple[float, float]:
        angle_error = float(self._wrap_to_pi(raw_angle - current_angle))
        if abs(angle_error) > 0.5 * np.pi:
            target_angle = float(self._wrap_to_pi(raw_angle + np.pi))
            target_wheel_vel = -raw_wheel_vel
        else:
            target_angle = float(self._wrap_to_pi(raw_angle))
            target_wheel_vel = raw_wheel_vel
        return target_angle, target_wheel_vel