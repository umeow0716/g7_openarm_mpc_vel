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
    OPENARM_U_LEFT_JOINT_ALL,
    OPENARM_U_RIGHT_JOINT_ALL,
    OPENARM_LEFT_HAND_POS,
    OPENARM_RIGHT_HAND_POS,
)
from .utils import quat_to_rotmat


FloatArray = npt.NDArray[np.float64]
VelocityFrame = Literal["body", "world"]


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

    wheel_radius_m: float = 0.052
    fl_pos_xy_m: tuple[float, float] = (0.198, 0.13)
    fr_pos_xy_m: tuple[float, float] = (0.198, -0.13)
    rl_pos_xy_m: tuple[float, float] = (-0.198, 0.13)
    rr_pos_xy_m: tuple[float, float] = (-0.198, -0.13)

    return_zero_command_on_nonfinite: bool = True
    min_module_speed_m_s: float = 1e-4

    steering_vel_limit_rad_s: float = 6.0
    wheel_vel_limit_rad_s: float = 30.0

    arm_acc_limit_rad_s2: float = 80.0

    # Match the direct qvel demo path in sim_viewer.py: the two gripper joints
    # per arm are not part of the end-effector task, so do not let numerical MPC
    # noise move them.
    zero_gripper_velocity_command: bool = True

    # Base feedback gains used directly by sim_viewer.py's MIT equation:
    #     tau = kp * (pos_des - pos) + kd * (vel_des - vel) + tau_ff
    # Steering uses position + velocity feedback. Wheels use velocity feedback.
    base_steering_kp: float = 20.0
    base_steering_kd: float = 0.02
    base_wheel_kd: float = 5.0

    # If the chassis command is essentially zero, disable base MIT feedback so
    # steering and wheel motors do not keep hunting around numerical noise.
    base_idle_linear_threshold_m_s: float = 1e-3
    base_idle_angular_threshold_rad_s: float = 1e-3

    # Use the existing MITCommand path as the fast velocity feedback loop:
    #
    #     tau_cmd = kd * (vel_des - qvel) + tau_ff
    #
    # tau_ff is still produced by the inverse-dynamics QP.  This is not the old
    # PID controller: no integral term is used, and the QP supplies the gravity /
    # bias feed-forward torque.
    use_motor_velocity_feedback: bool = True
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
    """Whole-body QP low-level controller.

    Inputs
    ------
    qpos: shape (33,)
        MuJoCo qpos / library configuration q.
    qvel: shape (32,)
        MuJoCo qvel / generalized velocity v.
    u_des: shape (21,)
        Mid-level velocity command.

    Output
    ------
    MITCommand in the same 26-actuator order used by g7_openarm.xml.
    The QP-computed torque is returned in tau_ff. kp/kd are zero by default so
    sim_viewer.py can keep using:

        tau_cmd = kp*qpos_err + kd*qvel_err + tau_ff
    """

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
    _floating_v_idx = np.arange(0, 6, dtype=np.int32)

    _steer_act_idx = np.array([0, 1, 2, 3], dtype=np.int32)
    _wheel_act_idx = np.array([4, 5, 6, 7], dtype=np.int32)
    _arm_act_idx = np.arange(8, 26, dtype=np.int32)

    # Rows in the 14-D kinematics output used by the TSID position task:
    # left TCP xyz and right TCP xyz.
    _ee_pos_rows = np.array(
        [
            *range(OPENARM_LEFT_HAND_POS.start, OPENARM_LEFT_HAND_POS.stop),
            *range(OPENARM_RIGHT_HAND_POS.start, OPENARM_RIGHT_HAND_POS.stop),
        ],
        dtype=np.int32,
    )

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
        self._prev_u_des = np.zeros(OPENARM_NU, dtype=np.float64)
        self._prev_arm_vel_des = np.zeros(18, dtype=np.float64)
        self._arm_pos_des = np.zeros(18, dtype=np.float64)
        self._arm_pos_des_initialized = False
        self._prev_tau = np.zeros(self.num_motors, dtype=np.float64)
        self._initialized = False
        self._ee_pos_target: FloatArray | None = None

        self._module_xy = np.array(
            [
                self.config.fl_pos_xy_m,
                self.config.fr_pos_xy_m,
                self.config.rl_pos_xy_m,
                self.config.rr_pos_xy_m,
            ],
            dtype=np.float64,
        )

        # Only arm tau_ff is produced by this controller.  Base torque is
        # generated later by sim_viewer.py from MIT kp/kd, so base tau_ff is
        # clamped to zero here.
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
        """
        根據 mass matrix 自動生成 Kd

        Args:
            model: PinnZooModel
            q: 當前 state（或 zero_state）
            zeta: 阻尼比（建議 0.7）
            omega_n: 頻率（scalar 或 float）
            per_joint_omega: (optional) 每個 joint 不同頻率 array

        Returns:
            Kd: np.ndarray (n_dof,)
        """

        # 取得 mass matrix
        M = M_func(self.model, x)

        # 取 diagonal inertia
        M_diag = np.diag(M)

        # 決定 omega
        if per_joint_omega is not None:
            omega = np.asarray(per_joint_omega)
        else:
            omega = omega_n

        # 計算 Kd
        Kd = 2.0 * zeta * omega * np.sqrt(M_diag)

        return Kd

    def set_ee_position_target(
        self,
        left_pos_target: FloatArray,
        right_pos_target: FloatArray,
    ) -> None:
        """Enable TSID Cartesian position feedback for both TCPs.

        The target is optional because the existing low-level API only receives
        ``u_des``.  If this is never called, the EE task is velocity-only.
        """
        left = np.asarray(left_pos_target, dtype=np.float64)
        right = np.asarray(right_pos_target, dtype=np.float64)
        if left.shape != (3,) or right.shape != (3,):
            raise ValueError("left_pos_target and right_pos_target must both have shape (3,)")
        target = np.empty(6, dtype=np.float64)
        target[:3] = left
        target[3:] = right
        if not np.all(np.isfinite(target)):
            raise ValueError("EE position target must be finite")
        self._ee_pos_target = target

    def clear_ee_position_target(self) -> None:
        """Disable Cartesian position feedback and keep velocity-only TSID."""
        self._ee_pos_target = None

    def reset(self) -> None:
        self._prev_u_des[:] = 0.0
        self._prev_arm_vel_des[:] = 0.0
        self._arm_pos_des[:] = 0.0
        self._arm_pos_des_initialized = False
        self._prev_tau[:] = 0.0
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
            if self.config.return_zero_command_on_nonfinite:
                self.reset()
                return self._zero_command()
            raise ValueError("qpos, qvel, and u_des must be finite")

        base_command_is_idle = self._is_base_command_idle(u_des)
        steer_pos_des, steer_vel_des, wheel_vel_des = self._base_velocity_to_swerve_targets(
            qpos=qpos,
            u_cmd=u_des,
        )
        if base_command_is_idle:
            # Stop commanding the base when the requested chassis velocity is
            # only numerical noise.  Gains are also disabled below, but keeping
            # targets equal to the current state makes debug plots easier to
            # read and prevents stale targets from being exposed.
            steer_pos_des = qpos[self._steer_qpos_idx].copy()
            steer_vel_des = np.zeros_like(steer_vel_des)
            wheel_vel_des = np.zeros_like(wheel_vel_des)

        arm_vel_des = self._arm_velocity_command(u_des)

        if (
            not np.all(np.isfinite(steer_pos_des))
            or not np.all(np.isfinite(steer_vel_des))
            or not np.all(np.isfinite(wheel_vel_des))
            or not np.all(np.isfinite(arm_vel_des))
        ):
            if self.config.return_zero_command_on_nonfinite:
                self.reset()
                return self._zero_command()
            raise FloatingPointError("desired velocity targets contain non-finite values")

        if not self._arm_pos_des_initialized:
            self._arm_pos_des[:] = qpos[self._arm_qpos_idx]
            self._arm_pos_des_initialized = True
        self._arm_pos_des += self.config.dt * arm_vel_des

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
        print(is_stuck)
        
        tau_bias = tau0 * is_stuck * np.sign(u_des[3:])
        tau_act[8:] += tau_bias
        
        if not np.all(np.isfinite(tau_act)):
            if self.config.return_zero_command_on_nonfinite:
                self.reset()
                return self._zero_command()
            raise FloatingPointError("QP torque output contains non-finite values")

        # Base is controlled by MIT kp/kd feedback in the returned command.
        # Do not precompute base PID torque into tau_ff; otherwise the gains are
        # evaluated against stale state before sim_viewer.py applies the command.
        tau_act[self._steer_act_idx] = 0.0
        tau_act[self._wheel_act_idx] = 0.0
        tau_act = np.clip(tau_act, self._tau_min, self._tau_max)

        self._prev_arm_vel_des[:] = arm_vel_des
        self._prev_tau[:] = tau_act

        pos_des = np.zeros(self.num_motors, dtype=np.float64)
        vel_des = np.zeros(self.num_motors, dtype=np.float64)
        kp = np.zeros(self.num_motors, dtype=np.float64)
        kd = np.zeros(self.num_motors, dtype=np.float64)

        pos_des[self._steer_act_idx] = steer_pos_des
        vel_des[self._steer_act_idx] = steer_vel_des
        pos_des[self._wheel_act_idx] = qpos[self._wheel_qpos_idx]
        vel_des[self._wheel_act_idx] = wheel_vel_des
        pos_des[self._arm_act_idx] = self._arm_pos_des
        vel_des[self._arm_act_idx] = arm_vel_des

        # Base MIT feedback is active only when the chassis command is not
        # effectively zero.  Near zero, leave kp/kd at zero so steering and
        # wheel motors are not controlled by tiny numerical commands.
        if not base_command_is_idle:
            kp[self._steer_act_idx] = self.config.base_steering_kp
            kd[self._steer_act_idx] = self.config.base_steering_kd
            kd[self._wheel_act_idx] = self.config.base_wheel_kd

        if self.config.use_motor_velocity_feedback:
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

    def _arm_velocity_command(self, u_cmd: FloatArray) -> FloatArray:
        arm_vel_des = np.concatenate(
            [u_cmd[OPENARM_U_LEFT_JOINT_ALL], u_cmd[OPENARM_U_RIGHT_JOINT_ALL]]
        ).astype(np.float64)

        if self.config.zero_gripper_velocity_command:
            # left gripper_LL/LR and right gripper_RL/RR inside the 18-D arm vector
            arm_vel_des[[7, 8, 16, 17]] = 0.0

        return arm_vel_des

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
            if self.config.return_zero_command_on_nonfinite:
                return np.zeros(self.num_motors, dtype=np.float64)
            raise FloatingPointError("desired actuator acceleration contains non-finite values")
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