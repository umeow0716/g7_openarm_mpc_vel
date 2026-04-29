"""Whole-body QP low-level controller for the G7 OpenArm mobile manipulator.

Drop-in replacement for src/low_level_controller_pid.py.

Current architecture
--------------------
The mid-level MPC outputs a 21-D velocity command:

    u_des = [base_vx, base_vy, base_wz, left_arm_vel(9), right_arm_vel(9)]

This controller maps that command into the 26 MuJoCo / MIT actuator order:

    [FL, FR, RL, RR, FLW, FRW, RLW, RRW,
     L_1 ... L_7, gripper_LL, gripper_LR,
     R_1 ... R_7, gripper_RL, gripper_RR]

Unlike the PID version, the feedback loop is acceleration-level and the torque is
computed through an inverse-dynamics QP using the library equations:

    q_dot = E(q) v
    M(q) v_dot + C(q, v) = B tau + J(q)^T lambda

For this project file, no contact Jacobian / contact force lambda is available in
wrapper.c, so the implemented equality is the actuated part of inverse dynamics:

    S_act (M(q) v_dot + C(q, v)) - tau_act = 0

where S_act selects the 26 actuated generalized-velocity rows in MuJoCo actuator
order. Floating-base/contact dynamics can be added later by introducing contact
force variables lambda and contact constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import os

import numpy as np
import numpy.typing as npt
from cffi import FFI

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
)
from .utils import quat_to_rotmat


FloatArray = npt.NDArray[np.float64]
VelocityFrame = Literal["body", "world"]


@dataclass(slots=True)
class PIDGains:
    """Simple PID gains for the base steering / wheel loops."""

    kp: FloatArray | float = 0.0
    ki: FloatArray | float = 0.0
    kd: FloatArray | float = 0.0


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
class WBCQPWeights:
    """Diagonal QP weights.

    The QP tracks desired generalized accelerations of the actuated coordinates.
    Larger acceleration weights make the controller follow the mid-level velocity
    command more aggressively. Larger torque weights make the torque smaller.
    """

    steering_acc: float = 20.0
    wheel_acc: float = 10.0
    arm_acc: FloatArray | float = field(
        default_factory=lambda: np.array(
            [25.0, 25.0, 20.0, 20.0, 10.0, 10.0, 8.0, 5.0, 5.0] * 2,
            dtype=np.float64,
        )
    )
    torque: float = 1e-4
    torque_rate: float = 1e-5
    vdot_regularization: float = 1e-6

    # Keep this at 0.0 until contact forces lambda are included. Enforcing the
    # floating-base rows without lambda makes the robot behave like a free flyer,
    # not like a wheeled robot on the ground.
    floating_base_dynamics: float = 0.0


@dataclass(slots=True)
class WBCLowLevelControllerConfig:
    dt: float = 0.01
    base_velocity_frame: VelocityFrame = "world"

    wheel_radius_m: float = 0.052
    fl_pos_xy_m: tuple[float, float] = (0.198, 0.13)
    fr_pos_xy_m: tuple[float, float] = (0.198, -0.13)
    rl_pos_xy_m: tuple[float, float] = (-0.198, 0.13)
    rr_pos_xy_m: tuple[float, float] = (-0.198, -0.13)

    # QP acceleration feed-forward gains.  The arm and wheel velocity feedback
    # is closed through the MIT kd field below, so these are zero by default.
    # Keeping velocity feedback outside of the QP makes the controller respond
    # at every MuJoCo step instead of only when the 100 Hz low-level process
    # recomputes tau_ff.
    steering_kp: float = 80.0
    steering_kd: float = 12.0
    wheel_kd: float = 0.0
    arm_kd: FloatArray | float = field(
        default_factory=lambda: np.zeros(18, dtype=np.float64)
    )

    command_smoothing_alpha: float = 0.10
    return_zero_command_on_nonfinite: bool = True
    min_module_speed_m_s: float = 1e-4

    steering_vel_limit_rad_s: float = 6.0
    wheel_vel_limit_rad_s: float = 30.0
    arm_vel_limit_rad_s: float = 2.0

    steering_acc_limit_rad_s2: float = 80.0
    wheel_acc_limit_rad_s2: float = 250.0
    arm_acc_limit_rad_s2: float = 80.0

    # The mid-level MPC command is already a velocity command. Differentiating
    # it with (v_des[k] - v_des[k-1]) / dt creates large acceleration spikes
    # whenever the MPC solution changes. Keep this off by default so the WBC
    # behaves like computed-torque velocity tracking instead of a jerky
    # velocity-feedforward controller.
    arm_velocity_feedforward_gain: float = 0.0
    wheel_velocity_feedforward_gain: float = 0.0

    # Match the direct qvel demo path in sim_viewer.py: the two gripper joints
    # per arm are not part of the end-effector task, so do not let numerical MPC
    # noise move them.
    zero_gripper_velocity_command: bool = True

    # The swerve base is intentionally kept on the same stable PID path as
    # src/low_level_controller_pid.py. The current WBC QP does not include
    # wheel-ground contact force variables, so using inverse dynamics for the
    # wheel/steering torques can inject unrealistic feed-forward torque and make
    # the base shoot away. WBC feed-forward is used for the arms; base torque is
    # produced by these original PID loops.
    use_original_base_pid: bool = True
    steering_position_pid: PIDGains = field(
        default_factory=lambda: PIDGains(kp=1.0, ki=0.0, kd=0.001)
    )
    wheel_velocity_pid: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.5, ki=0.0, kd=0.008)
    )
    steering_integral_limit: float = 0.8
    wheel_integral_limit: float = 5.0

    # Use the existing MITCommand path as the fast velocity feedback loop:
    #
    #     tau_cmd = kd * (vel_des - qvel) + tau_ff
    #
    # tau_ff is still produced by the inverse-dynamics QP.  This is not the old
    # PID controller: no integral term is used, and the QP supplies the gravity /
    # bias feed-forward torque.
    use_motor_velocity_feedback: bool = True
    wheel_motor_kd: float = 0.0
    arm_motor_kd: FloatArray | float = field(
        default_factory=lambda: np.array(
            [8.0, 8.0, 6.0, 6.0, 2.0, 3.0, 1.5, 0.0, 0.0] * 2,
            dtype=np.float64,
        )
    )

    # Evaluate C(q, v) near the velocity command instead of at stale measured
    # qvel.  This follows the original PID implementation's gravity/bias
    # compensation behavior more closely.
    bias_use_desired_velocity: bool = True
    bias_position_preview_s: float = 0.05

    steering_tau_limit: float = 23.7
    wheel_tau_limit: float = 15.0
    arm_tau_limit_scale: float = 1.0

    max_active_set_iter: int = 8
    qp_condition_regularization: float = 1e-9

    # Important for the current project model. The QP does not yet include
    # wheel-ground contact forces lambda. If the six floating-base
    # accelerations are left free, the optimizer can use unreal base
    # accelerations to cancel arm gravity and reduce tau. Locking them to
    # zero inside the inverse-dynamics QP makes the actuator rows behave like
    # a supported/fixed-base inverse dynamics calculation, so C(q, v) becomes
    # real gravity/bias compensation for the arms.
    constrain_floating_base_acceleration: bool = True

    weights: WBCQPWeights = field(default_factory=WBCQPWeights)


class WholeBodyDynamicsLibrary:
    """Minimal cffi binding for functions exposed by wrapper.c."""

    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"file `{lib_path}` not found")

        self.lib_path = lib_path
        self.ffi = FFI()
        self.ffi.cdef(
            """
            extern const char* config_names[];
            extern const char* vel_names[];
            extern const char* torque_names[];
            void M_func_wrapper(double* x_in, double* M_out);
            void C_func_wrapper(double* x_in, double* C_out);
            void velocity_kinematics_wrapper(double* x_in, double* E_out);
            void inverse_dynamics_wrapper(double* x_in, double* vdot_in, double* tau_out);
            """
        )
        self.lib = self.ffi.dlopen(os.path.abspath(lib_path))

        self.nq = self._get_c_array_len(self.lib.config_names)  # type: ignore[attr-defined]
        self.nv = self._get_c_array_len(self.lib.vel_names)  # type: ignore[attr-defined]
        self.ntorque_names = self._get_c_array_len(self.lib.torque_names)  # type: ignore[attr-defined]
        self.nx = self.nq + self.nv

        if self.nq != OPENARM_NQ or self.nv != OPENARM_NV:
            raise ValueError(
                f"Unexpected dynamics size: nq={self.nq}, nv={self.nv}; "
                f"expected nq={OPENARM_NQ}, nv={OPENARM_NV}"
            )

    def _get_c_array_len(self, ptr) -> int:
        count = 0
        while ptr[count] != self.ffi.NULL:
            count += 1
        return count

    def make_state(self, qpos: FloatArray, qvel: FloatArray) -> FloatArray:
        x = np.empty(self.nx, dtype=np.float64)
        x[: self.nq] = qpos
        x[self.nq :] = qvel
        return x

    def mass_matrix(self, x: FloatArray) -> FloatArray:
        out = np.empty(self.nv * self.nv, dtype=np.float64)
        self.lib.M_func_wrapper(  # type: ignore[attr-defined]
            self.ffi.cast("double*", x.ctypes.data),
            self.ffi.cast("double*", out.ctypes.data),
        )
        return out.reshape((self.nv, self.nv), order="F")

    def bias_force(self, x: FloatArray) -> FloatArray:
        out = np.empty(self.nv, dtype=np.float64)
        self.lib.C_func_wrapper(  # type: ignore[attr-defined]
            self.ffi.cast("double*", x.ctypes.data),
            self.ffi.cast("double*", out.ctypes.data),
        )
        return out

    def velocity_kinematics_matrix(self, x: FloatArray) -> FloatArray:
        out = np.empty(self.nq * self.nv, dtype=np.float64)
        self.lib.velocity_kinematics_wrapper(  # type: ignore[attr-defined]
            self.ffi.cast("double*", x.ctypes.data),
            self.ffi.cast("double*", out.ctypes.data),
        )
        return out.reshape((self.nq, self.nv), order="F")

    def inverse_dynamics_generalized(self, x: FloatArray, vdot: FloatArray) -> FloatArray:
        out = np.empty(self.nv, dtype=np.float64)
        self.lib.inverse_dynamics_wrapper(  # type: ignore[attr-defined]
            self.ffi.cast("double*", x.ctypes.data),
            self.ffi.cast("double*", vdot.ctypes.data),
            self.ffi.cast("double*", out.ctypes.data),
        )
        return out


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

    _arm_xml_tau_limit = np.array(
        [
            40.0,
            40.0,
            27.0,
            27.0,
            7.0,
            7.0,
            7.0,
            15.0,
            15.0,
            40.0,
            40.0,
            27.0,
            27.0,
            7.0,
            7.0,
            7.0,
            15.0,
            15.0,
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        config: WBCLowLevelControllerConfig | None = None,
        lib_path: str = "include/libg7_openarm_quat.so",
    ) -> None:
        self.config = config if config is not None else WBCLowLevelControllerConfig()
        self.lib_path = lib_path
        self.dyn = WholeBodyDynamicsLibrary(lib_path)

        self.num_motors = len(self.motor_names)
        self._prev_u_des = np.zeros(OPENARM_NU, dtype=np.float64)
        self._prev_wheel_vel_des = np.zeros(4, dtype=np.float64)
        self._prev_arm_vel_des = np.zeros(18, dtype=np.float64)
        self._prev_tau = np.zeros(self.num_motors, dtype=np.float64)
        self._prev_steering_error = np.zeros(4, dtype=np.float64)
        self._prev_wheel_vel_error = np.zeros(4, dtype=np.float64)
        self._steering_integral = np.zeros(4, dtype=np.float64)
        self._wheel_integral = np.zeros(4, dtype=np.float64)
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

        self._tau_min = np.empty(self.num_motors, dtype=np.float64)
        self._tau_max = np.empty(self.num_motors, dtype=np.float64)
        self._tau_min[self._steer_act_idx] = -self.config.steering_tau_limit
        self._tau_max[self._steer_act_idx] = self.config.steering_tau_limit
        self._tau_min[self._wheel_act_idx] = -self.config.wheel_tau_limit
        self._tau_max[self._wheel_act_idx] = self.config.wheel_tau_limit
        arm_limit = self.config.arm_tau_limit_scale * self._arm_xml_tau_limit
        self._tau_min[self._arm_act_idx] = -arm_limit
        self._tau_max[self._arm_act_idx] = arm_limit

        self._selection_act = np.zeros((self.num_motors, OPENARM_NV), dtype=np.float64)
        self._selection_act[np.arange(self.num_motors), self._actuated_v_idx] = 1.0

    def reset(self) -> None:
        self._prev_u_des[:] = 0.0
        self._prev_wheel_vel_des[:] = 0.0
        self._prev_arm_vel_des[:] = 0.0
        self._prev_tau[:] = 0.0
        self._prev_steering_error[:] = 0.0
        self._prev_wheel_vel_error[:] = 0.0
        self._steering_integral[:] = 0.0
        self._wheel_integral[:] = 0.0
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

        u_cmd = self._smooth_command(u_des)

        steer_pos_des, steer_vel_des, wheel_vel_des = self._base_velocity_to_swerve_targets(
            qpos=qpos,
            u_cmd=u_cmd,
        )
        arm_vel_des = self._arm_velocity_command(u_cmd)

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

        desired_act_acc = self._build_desired_actuator_acceleration(
            qpos=qpos,
            qvel=qvel,
            steer_pos_des=steer_pos_des,
            steer_vel_des=steer_vel_des,
            wheel_vel_des=wheel_vel_des,
            arm_vel_des=arm_vel_des,
        )

        x_ff = self._make_feedforward_state(
            qpos=qpos,
            qvel=qvel,
            steer_vel_des=steer_vel_des,
            wheel_vel_des=wheel_vel_des,
            arm_vel_des=arm_vel_des,
        )
        mass_matrix = self.dyn.mass_matrix(x_ff)
        bias_force = self.dyn.bias_force(x_ff)
        if not np.all(np.isfinite(mass_matrix)) or not np.all(np.isfinite(bias_force)):
            if self.config.return_zero_command_on_nonfinite:
                self.reset()
                return self._zero_command()
            raise FloatingPointError("mass matrix or bias force contains non-finite values")

        _, tau_act = self._solve_inverse_dynamics_qp(
            mass_matrix=mass_matrix,
            bias_force=bias_force,
            desired_act_acc=desired_act_acc,
        )
        if not np.all(np.isfinite(tau_act)):
            if self.config.return_zero_command_on_nonfinite:
                self.reset()
                return self._zero_command()
            raise FloatingPointError("QP torque output contains non-finite values")

        if self.config.use_original_base_pid:
            tau_act[self._steer_act_idx] = self._position_pid(
                pos_now=qpos[self._steer_qpos_idx],
                vel_now=qvel[self._steer_qvel_idx],
                pos_des=steer_pos_des,
                vel_des=steer_vel_des,
                gains=self.config.steering_position_pid,
                integral=self._steering_integral,
                prev_error=self._prev_steering_error,
                integral_limit=self.config.steering_integral_limit,
            )
            tau_act[self._wheel_act_idx] = self._velocity_pid(
                vel_now=qvel[self._wheel_qvel_idx],
                vel_des=wheel_vel_des,
                gains=self.config.wheel_velocity_pid,
                integral=self._wheel_integral,
                prev_error=self._prev_wheel_vel_error,
                integral_limit=self.config.wheel_integral_limit,
            )
            # sim_viewer.py adds MIT kp/kd feedback after reading this command.
            # Keep the base command fully bounded here and leave wheel_motor_kd
            # at zero by default, otherwise the final torque can exceed the
            # base actuator ranges before MuJoCo clamps it.
            tau_act = np.clip(tau_act, self._tau_min, self._tau_max)

        self._prev_wheel_vel_des[:] = wheel_vel_des
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
        pos_des[self._arm_act_idx] = qpos[self._arm_qpos_idx]
        vel_des[self._arm_act_idx] = arm_vel_des

        if self.config.use_motor_velocity_feedback:
            if not self.config.use_original_base_pid:
                kd[self._wheel_act_idx] = self.config.wheel_motor_kd
            kd[self._arm_act_idx] = self.config.arm_motor_kd

        return MITCommand(
            motor_names=self.motor_names.copy(),
            pos_des=pos_des,
            vel_des=vel_des,
            kp=kp,
            kd=kd,
            tau_ff=tau_act,
        )

    def _make_feedforward_state(
        self,
        qpos: FloatArray,
        qvel: FloatArray,
        steer_vel_des: FloatArray,
        wheel_vel_des: FloatArray,
        arm_vel_des: FloatArray,
    ) -> FloatArray:
        qpos_ff = qpos.copy()
        qvel_ff = qvel.copy()

        if self.config.bias_position_preview_s > 0.0:
            preview_s = float(self.config.bias_position_preview_s)
            qpos_ff[self._arm_qpos_idx] = (
                qpos_ff[self._arm_qpos_idx] + preview_s * arm_vel_des
            )

        if self.config.bias_use_desired_velocity:
            # When the base is controlled by the original PID path, do not feed
            # desired wheel speeds into the inverse-dynamics bias calculation.
            # Wheel-ground contact forces are not modeled in the QP, so those
            # terms are not reliable feed-forward for the base.
            if not self.config.use_original_base_pid:
                qvel_ff[self._steer_qvel_idx] = steer_vel_des
                qvel_ff[self._wheel_qvel_idx] = wheel_vel_des
            qvel_ff[self._arm_qvel_idx] = arm_vel_des

        return self.dyn.make_state(qpos_ff, qvel_ff)

    def _smooth_command(self, u_des: FloatArray) -> FloatArray:
        alpha = float(np.clip(self.config.command_smoothing_alpha, 0.0, 0.999))
        if not self._initialized:
            self._prev_u_des[:] = u_des
            self._initialized = True
            return u_des.copy()

        u_cmd = alpha * self._prev_u_des + (1.0 - alpha) * u_des
        self._prev_u_des[:] = u_cmd
        return u_cmd

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

        return np.clip(
            arm_vel_des,
            -self.config.arm_vel_limit_rad_s,
            self.config.arm_vel_limit_rad_s,
        )

    def _build_desired_actuator_acceleration(
        self,
        qpos: FloatArray,
        qvel: FloatArray,
        steer_pos_des: FloatArray,
        steer_vel_des: FloatArray,
        wheel_vel_des: FloatArray,
        arm_vel_des: FloatArray,
    ) -> FloatArray:
        acc = np.zeros(self.num_motors, dtype=np.float64)

        steering_pos_err = self._wrap_to_pi(steer_pos_des - qpos[self._steer_qpos_idx])
        steering_vel_err = steer_vel_des - qvel[self._steer_qvel_idx]
        acc[self._steer_act_idx] = (
            self.config.steering_kp * steering_pos_err
            + self.config.steering_kd * steering_vel_err
        )
        acc[self._steer_act_idx] = np.clip(
            acc[self._steer_act_idx],
            -self.config.steering_acc_limit_rad_s2,
            self.config.steering_acc_limit_rad_s2,
        )

        wheel_acc_ff = (wheel_vel_des - self._prev_wheel_vel_des) / max(self.config.dt, 1e-9)
        wheel_vel_err = wheel_vel_des - qvel[self._wheel_qvel_idx]
        acc[self._wheel_act_idx] = (
            self.config.wheel_velocity_feedforward_gain * wheel_acc_ff
            + self.config.wheel_kd * wheel_vel_err
        )
        acc[self._wheel_act_idx] = np.clip(
            acc[self._wheel_act_idx],
            -self.config.wheel_acc_limit_rad_s2,
            self.config.wheel_acc_limit_rad_s2,
        )

        arm_acc_ff = (arm_vel_des - self._prev_arm_vel_des) / max(self.config.dt, 1e-9)
        arm_vel_err = arm_vel_des - qvel[self._arm_qvel_idx]
        acc[self._arm_act_idx] = (
            self.config.arm_velocity_feedforward_gain * arm_acc_ff
            + self.config.arm_kd * arm_vel_err
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

    def _solve_inverse_dynamics_qp(
        self,
        mass_matrix: FloatArray,
        bias_force: FloatArray,
        desired_act_acc: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        nv = OPENARM_NV
        na = self.num_motors
        nvar = nv + na

        selection = self._selection_act
        mass_act = mass_matrix[self._actuated_v_idx, :]
        bias_act = bias_force[self._actuated_v_idx]

        acc_weight_diag = self._acceleration_weight_diag()
        torque_weight_diag = np.full(na, self.config.weights.torque, dtype=np.float64)
        torque_rate_weight_diag = np.full(na, self.config.weights.torque_rate, dtype=np.float64)

        hessian = np.zeros((nvar, nvar), dtype=np.float64)
        gradient = np.zeros(nvar, dtype=np.float64)

        hessian[:nv, :nv] += selection.T @ (acc_weight_diag[:, None] * selection)
        gradient[:nv] += -(selection.T @ (acc_weight_diag * desired_act_acc))

        hessian[nv:, nv:] += np.diag(torque_weight_diag + torque_rate_weight_diag)
        gradient[nv:] += -(torque_rate_weight_diag * self._prev_tau)

        hessian[:nv, :nv] += self.config.weights.vdot_regularization * np.eye(nv, dtype=np.float64)

        if self.config.weights.floating_base_dynamics > 0.0:
            mass_base = mass_matrix[self._floating_v_idx, :]
            bias_base = bias_force[self._floating_v_idx]
            w_base = self.config.weights.floating_base_dynamics
            hessian[:nv, :nv] += w_base * (mass_base.T @ mass_base)
            gradient[:nv] += w_base * (mass_base.T @ bias_base)

        hessian += self.config.qp_condition_regularization * np.eye(nvar, dtype=np.float64)
        hessian = 0.5 * (hessian + hessian.T)

        # S_act (M vdot + C) - tau = 0 -> S_act M vdot - tau = -S_act C
        equality = np.zeros((na, nvar), dtype=np.float64)
        equality[:, :nv] = mass_act
        equality[:, nv:] = -np.eye(na, dtype=np.float64)
        equality_rhs = -bias_act

        if self.config.constrain_floating_base_acceleration:
            # Without contact-force variables, these six floating-base
            # accelerations are otherwise fake optimization slack variables.
            # Constraining them prevents gravity from being cancelled by an
            # unreal base acceleration and restores arm gravity compensation.
            base_acc_equality = np.zeros((len(self._floating_v_idx), nvar), dtype=np.float64)
            base_acc_equality[:, self._floating_v_idx] = np.eye(
                len(self._floating_v_idx),
                dtype=np.float64,
            )
            equality = np.vstack([equality, base_acc_equality])
            equality_rhs = np.concatenate([
                equality_rhs,
                np.zeros(len(self._floating_v_idx), dtype=np.float64),
            ])

        active_tau_idx: list[int] = []
        active_tau_value: list[float] = []

        solution = self._solve_equality_qp(hessian, gradient, equality, equality_rhs)
        tau = solution[nv:].copy()

        for _ in range(self.config.max_active_set_iter):
            lower_violation = tau < self._tau_min
            upper_violation = tau > self._tau_max
            violated = np.where(lower_violation | upper_violation)[0]
            new_active = False

            for idx in violated:
                idx_int = int(idx)
                if idx_int in active_tau_idx:
                    continue
                active_tau_idx.append(idx_int)
                if lower_violation[idx_int]:
                    active_tau_value.append(float(self._tau_min[idx_int]))
                else:
                    active_tau_value.append(float(self._tau_max[idx_int]))
                new_active = True

            if not new_active:
                break

            bound_equality = np.zeros((len(active_tau_idx), nvar), dtype=np.float64)
            for row, idx_int in enumerate(active_tau_idx):
                bound_equality[row, nv + idx_int] = 1.0
            augmented_equality = np.vstack([equality, bound_equality])
            augmented_rhs = np.concatenate([equality_rhs, np.array(active_tau_value, dtype=np.float64)])

            solution = self._solve_equality_qp(
                hessian,
                gradient,
                augmented_equality,
                augmented_rhs,
            )
            tau = solution[nv:].copy()

        tau = np.clip(tau, self._tau_min, self._tau_max)
        vdot = solution[:nv].copy()
        return vdot, tau

    def _solve_equality_qp(
        self,
        hessian: FloatArray,
        gradient: FloatArray,
        equality: FloatArray,
        equality_rhs: FloatArray,
    ) -> FloatArray:
        nvar = hessian.shape[0]
        neq = equality.shape[0]
        kkt = np.block(
            [
                [hessian, equality.T],
                [equality, np.zeros((neq, neq), dtype=np.float64)],
            ]
        )
        rhs = np.concatenate([-gradient, equality_rhs])

        try:
            sol = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(kkt, rhs, rcond=1e-10)[0]

        return sol[:nvar]

    def _acceleration_weight_diag(self) -> FloatArray:
        weights = np.empty(self.num_motors, dtype=np.float64)
        if self.config.use_original_base_pid:
            # Base tracking is handled by the stable original PID path.
            # Do not let base acceleration objectives shape the WBC feed-forward.
            weights[self._steer_act_idx] = 0.0
            weights[self._wheel_act_idx] = 0.0
        else:
            weights[self._steer_act_idx] = self.config.weights.steering_acc
            weights[self._wheel_act_idx] = self.config.weights.wheel_acc
        weights[self._arm_act_idx] = self.config.weights.arm_acc
        return weights

    def _position_pid(
        self,
        pos_now: FloatArray,
        vel_now: FloatArray,
        pos_des: FloatArray,
        vel_des: FloatArray,
        gains: PIDGains,
        integral: FloatArray,
        prev_error: FloatArray,
        integral_limit: float,
    ) -> FloatArray:
        error = self._wrap_to_pi(pos_des - pos_now)
        error_rate = vel_des - vel_now

        if not np.isfinite(integral).all():
            integral.fill(0.0)

        integral += error * self.config.dt
        np.clip(integral, -integral_limit, integral_limit, out=integral)

        tau = gains.kp * error + gains.ki * integral + gains.kd * error_rate
        prev_error[:] = error
        return np.asarray(tau, dtype=np.float64)

    def _velocity_pid(
        self,
        vel_now: FloatArray,
        vel_des: FloatArray,
        gains: PIDGains,
        integral: FloatArray,
        prev_error: FloatArray,
        integral_limit: float,
    ) -> FloatArray:
        error = vel_des - vel_now

        if not np.isfinite(integral).all():
            integral.fill(0.0)

        integral += error * self.config.dt
        np.clip(integral, -integral_limit, integral_limit, out=integral)

        error_dot = (error - prev_error) / max(self.config.dt, 1e-9)
        tau = gains.kp * error + gains.ki * integral + gains.kd * error_dot
        prev_error[:] = error
        return np.asarray(tau, dtype=np.float64)

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
