"""TSID-lite low-level controller with base MIT feedback for G7 OpenArm.

Drop-in API:

    command = LowLevelController.update(qpos, qvel, u_des)

where ``u_des`` is the existing 21-D velocity-planner output:

    [base_vx, base_vy, base_wz, left_arm_vel(9), right_arm_vel(9)]

Runtime architecture
--------------------
Base:
    base_vx/vy/wz -> swerve steering angle + wheel velocity targets
                  -> MITCommand kp/kd feedback in sim_viewer.py

Arm:
    TSID-lite inverse-dynamics QP -> tau_ff
    optional MIT velocity feedback through kd

The base no longer computes an internal PID torque into tau_ff.  Its PID/PD
feedback is exposed through the MITCommand fields, so the simulator applies it
against the current qpos/qvel state.
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
class TSIDTaskData:
    """Acceleration-level task written as ``A @ vdot ~= rhs``."""

    jacobian: FloatArray
    rhs: FloatArray
    weight_diag: FloatArray


@dataclass(slots=True)
class TSIDQPWeights:
    """Diagonal TSID QP weights.

    ``ee_pos_acc`` weights the true task-space acceleration objective.
    ``arm_acc`` is a joint-space posture / velocity-tracking secondary task.
    Larger torque weights make the torque smaller and less aggressive.
    """

    ee_pos_acc: float = 120.0
    arm_acc: FloatArray | float = field(
        default_factory=lambda: np.array(
            [25.0, 25.0, 20.0, 20.0, 10.0, 10.0, 8.0, 5.0, 5.0] * 2,
            dtype=np.float64,
        )
    )
    torque: float = 1e-4
    torque_rate: float = 1e-5
    vdot_regularization: float = 1e-6



@dataclass(slots=True)
class TSIDLowLevelControllerConfig:
    dt: float = 0.01
    base_velocity_frame: VelocityFrame = "world"

    wheel_radius_m: float = 0.052
    fl_pos_xy_m: tuple[float, float] = (0.198, 0.13)
    fr_pos_xy_m: tuple[float, float] = (0.198, -0.13)
    rl_pos_xy_m: tuple[float, float] = (-0.198, 0.13)
    rr_pos_xy_m: tuple[float, float] = (-0.198, -0.13)

    # Arm desired acceleration gain used by the QP secondary task.
    # Base gains are exposed directly as MITCommand kp/kd below.
    arm_kd: FloatArray | float = field(
        default_factory=lambda: np.array(
            [28.0, 28.0, 22.0, 22.0, 10.0, 10.0, 8.0, 4.0, 4.0] * 2,
            dtype=np.float64,
        ) * 2.0
    )

    # TSID end-effector task.  Without an explicit EE position target, this
    # behaves as an EE velocity-tracking task generated from the planner's
    # desired arm joint velocity.  If set_ee_position_target() is called, the
    # same task also adds position feedback.
    enable_ee_position_task: bool = True
    ee_position_kp: float = 80.0
    ee_velocity_kd: float = 24.0
    ee_acc_limit_m_s2: float = 25.0
    use_base_velocity_in_ee_reference: bool = False

    enable_arm_joint_task: bool = True

    command_smoothing_alpha: float = 0.10
    return_zero_command_on_nonfinite: bool = True
    min_module_speed_m_s: float = 1e-4

    steering_vel_limit_rad_s: float = 6.0
    wheel_vel_limit_rad_s: float = 30.0
    arm_vel_limit_rad_s: float = 2.0

    arm_acc_limit_rad_s2: float = 80.0

    # The mid-level MPC command is already a velocity command. Differentiating
    # it with (v_des[k] - v_des[k-1]) / dt creates large acceleration spikes
    # whenever the MPC solution changes. Keep this off by default.
    arm_velocity_feedforward_gain: float = 0.0

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
    base_idle_linear_threshold_m_s: float = 1e-2
    base_idle_angular_threshold_rad_s: float = 1e-2

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

    # Evaluate C(q, v) near the velocity command instead of at stale measured
    # qvel.  This follows the original PID implementation's gravity/bias
    # compensation behavior more closely.
    bias_use_desired_velocity: bool = True
    bias_position_preview_s: float = 0.05

    arm_tau_limit_scale: float = 1.0

    max_active_set_iter: int = 8
    qp_condition_regularization: float = 1e-9


    weights: TSIDQPWeights = field(default_factory=TSIDQPWeights)


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
            extern const char* kinematics_bodies[];
            void M_func_wrapper(double* x_in, double* M_out);
            void C_func_wrapper(double* x_in, double* C_out);
            void velocity_kinematics_wrapper(double* x_in, double* E_out);
            void inverse_dynamics_wrapper(double* x_in, double* vdot_in, double* tau_out);
            void kinematics_wrapper(double* x_in, double* locs_out);
            void kinematics_velocity_wrapper(double* x_in, double* locs_dot_out);
            void kinematics_velocity_jacobian_wrapper(double* x_in, double* J_dot_out);
            """
        )
        self.lib = self.ffi.dlopen(os.path.abspath(lib_path))

        self.nq = self._get_c_array_len(self.lib.config_names)  # type: ignore[attr-defined]
        self.nv = self._get_c_array_len(self.lib.vel_names)  # type: ignore[attr-defined]
        self.ntorque_names = self._get_c_array_len(self.lib.torque_names)  # type: ignore[attr-defined]
        self.bodies_count = self._get_c_array_len(self.lib.kinematics_bodies)  # type: ignore[attr-defined]
        self.kinematics_size = 7 * self.bodies_count
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

    def kinematics(self, x: FloatArray) -> FloatArray:
        out = np.empty(self.kinematics_size, dtype=np.float64)
        self.lib.kinematics_wrapper(  # type: ignore[attr-defined]
            self.ffi.cast("double*", x.ctypes.data),
            self.ffi.cast("double*", out.ctypes.data),
        )
        return out

    def kinematics_velocity(self, x: FloatArray) -> FloatArray:
        out = np.empty(self.kinematics_size, dtype=np.float64)
        self.lib.kinematics_velocity_wrapper(  # type: ignore[attr-defined]
            self.ffi.cast("double*", x.ctypes.data),
            self.ffi.cast("double*", out.ctypes.data),
        )
        return out

    def kinematics_velocity_jacobian(self, x: FloatArray) -> FloatArray:
        out = np.empty(self.kinematics_size * self.nx, dtype=np.float64)
        self.lib.kinematics_velocity_jacobian_wrapper(  # type: ignore[attr-defined]
            self.ffi.cast("double*", x.ctypes.data),
            self.ffi.cast("double*", out.ctypes.data),
        )
        return out.reshape((self.kinematics_size, self.nx), order="F")

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
        config: TSIDLowLevelControllerConfig | None = None,
        lib_path: str = "include/libg7_openarm_quat.so",
    ) -> None:
        self.config = config if config is not None else TSIDLowLevelControllerConfig()
        self.lib_path = lib_path
        self.dyn = WholeBodyDynamicsLibrary(lib_path)

        self.num_motors = len(self.motor_names)
        self._prev_u_des = np.zeros(OPENARM_NU, dtype=np.float64)
        self._prev_arm_vel_des = np.zeros(18, dtype=np.float64)
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
        arm_limit = self.config.arm_tau_limit_scale * self._arm_xml_tau_limit
        self._tau_min[self._arm_act_idx] = -arm_limit
        self._tau_max[self._arm_act_idx] = arm_limit

        self._selection_act = np.zeros((self.num_motors, OPENARM_NV), dtype=np.float64)
        self._selection_act[np.arange(self.num_motors), self._actuated_v_idx] = 1.0

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

        u_cmd = self._smooth_command(u_des)

        base_command_is_idle = self._is_base_command_idle(u_cmd)
        steer_pos_des, steer_vel_des, wheel_vel_des = self._base_velocity_to_swerve_targets(
            qpos=qpos,
            u_cmd=u_cmd,
        )
        if base_command_is_idle:
            # Stop commanding the base when the requested chassis velocity is
            # only numerical noise.  Gains are also disabled below, but keeping
            # targets equal to the current state makes debug plots easier to
            # read and prevents stale targets from being exposed.
            steer_pos_des = qpos[self._steer_qpos_idx].copy()
            steer_vel_des = np.zeros_like(steer_vel_des)
            wheel_vel_des = np.zeros_like(wheel_vel_des)

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
            qvel=qvel,
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

        tsid_task = self._build_tsid_task(
            qpos=qpos,
            qvel=qvel,
            u_cmd=u_cmd,
            arm_vel_des=arm_vel_des,
        )

        _, tau_act = self._solve_tsid_qp(
            mass_matrix=mass_matrix,
            bias_force=bias_force,
            desired_act_acc=desired_act_acc,
            tsid_task=tsid_task,
        )
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
        pos_des[self._arm_act_idx] = qpos[self._arm_qpos_idx]
        vel_des[self._arm_act_idx] = arm_vel_des

        # Base MIT feedback is active only when the chassis command is not
        # effectively zero.  Near zero, leave kp/kd at zero so steering and
        # wheel motors are not controlled by tiny numerical commands.
        if not base_command_is_idle:
            kp[self._steer_act_idx] = self.config.base_steering_kp
            kd[self._steer_act_idx] = self.config.base_steering_kd
            kd[self._wheel_act_idx] = self.config.base_wheel_kd

        if self.config.use_motor_velocity_feedback:
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
            # Only preview arm velocity for the arm/WBC feed-forward term.
            # Base steering and wheel tracking are handled by MIT kp/kd.
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

        return np.clip(
            arm_vel_des,
            -self.config.arm_vel_limit_rad_s,
            self.config.arm_vel_limit_rad_s,
        )

    def _build_desired_actuator_acceleration(
        self,
        qvel: FloatArray,
        arm_vel_des: FloatArray,
    ) -> FloatArray:
        acc = np.zeros(self.num_motors, dtype=np.float64)

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

    def _build_tsid_task(
        self,
        qpos: FloatArray,
        qvel: FloatArray,
        u_cmd: FloatArray,
        arm_vel_des: FloatArray,
    ) -> TSIDTaskData:
        """Build the Cartesian TSID task ``A @ vdot ~= rhs``.

        The generated ``kinematics_velocity_jacobian`` returns
        ``d(kinematics_velocity) / d[q, v]``.  Since
        ``kinematics_velocity = J(q) v``, we recover:

            J       = d(kinematics_velocity) / dv
            Jdot v  = d(kinematics_velocity) / dq * qdot

        Therefore task acceleration is:

            xddot = J vdot + Jdot v
        """
        if not self.config.enable_ee_position_task:
            return TSIDTaskData(
                jacobian=np.zeros((0, OPENARM_NV), dtype=np.float64),
                rhs=np.zeros(0, dtype=np.float64),
                weight_diag=np.zeros(0, dtype=np.float64),
            )

        x_now = self.dyn.make_state(qpos, qvel)
        task_vel_now_all = self.dyn.kinematics_velocity(x_now)
        task_vel_jac_all = self.dyn.kinematics_velocity_jacobian(x_now)
        velocity_kinematics = self.dyn.velocity_kinematics_matrix(x_now)
        qdot = velocity_kinematics @ qvel

        rows = self._ee_pos_rows
        task_jacobian = task_vel_jac_all[rows, OPENARM_NQ:]
        jdot_v = task_vel_jac_all[rows, :OPENARM_NQ] @ qdot
        task_vel_now = task_vel_now_all[rows]

        qvel_ref = qvel.copy()
        qvel_ref[self._arm_qvel_idx] = arm_vel_des

        if self.config.use_base_velocity_in_ee_reference:
            vx = float(u_cmd[OPENARM_U_BASE_VX])
            vy = float(u_cmd[OPENARM_U_BASE_VY])
            wz = float(u_cmd[OPENARM_U_BASE_WZ])
            if self.config.base_velocity_frame == "body":
                rot_world_from_body = quat_to_rotmat(qpos[OPENARM_WORLD_QUAT])
                v_world = rot_world_from_body @ np.array([vx, vy, 0.0], dtype=np.float64)
                qvel_ref[0] = float(v_world[0])
                qvel_ref[1] = float(v_world[1])
            else:
                qvel_ref[0] = vx
                qvel_ref[1] = vy
            qvel_ref[5] = wz

        x_ref = self.dyn.make_state(qpos, qvel_ref)
        task_vel_ref = self.dyn.kinematics_velocity(x_ref)[rows]

        task_acc_des = self.config.ee_velocity_kd * (task_vel_ref - task_vel_now)

        if self._ee_pos_target is not None:
            task_pos_now = self.dyn.kinematics(x_now)[rows]
            task_acc_des += self.config.ee_position_kp * (self._ee_pos_target - task_pos_now)

        task_acc_des = np.clip(
            task_acc_des,
            -self.config.ee_acc_limit_m_s2,
            self.config.ee_acc_limit_m_s2,
        )

        rhs = task_acc_des - jdot_v
        weight_diag = np.full(
            len(rows),
            self.config.weights.ee_pos_acc,
            dtype=np.float64,
        )

        if not (
            np.all(np.isfinite(task_jacobian))
            and np.all(np.isfinite(rhs))
            and np.all(np.isfinite(weight_diag))
        ):
            if self.config.return_zero_command_on_nonfinite:
                return TSIDTaskData(
                    jacobian=np.zeros((0, OPENARM_NV), dtype=np.float64),
                    rhs=np.zeros(0, dtype=np.float64),
                    weight_diag=np.zeros(0, dtype=np.float64),
                )
            raise FloatingPointError("TSID task contains non-finite values")

        return TSIDTaskData(
            jacobian=task_jacobian,
            rhs=rhs,
            weight_diag=weight_diag,
        )

    def _solve_tsid_qp(
        self,
        mass_matrix: FloatArray,
        bias_force: FloatArray,
        desired_act_acc: FloatArray,
        tsid_task: TSIDTaskData,
    ) -> tuple[FloatArray, FloatArray]:
        nv = OPENARM_NV
        na = self.num_motors
        nvar = nv + na

        selection = self._selection_act
        mass_act = mass_matrix[self._actuated_v_idx, :]
        bias_act = bias_force[self._actuated_v_idx]

        hessian = np.zeros((nvar, nvar), dtype=np.float64)
        gradient = np.zeros(nvar, dtype=np.float64)

        # Primary TSID Cartesian task: J_ee vdot ~= xddot_des - Jdot v.
        if tsid_task.jacobian.shape[0] > 0:
            task_a = tsid_task.jacobian
            task_b = tsid_task.rhs
            task_w = tsid_task.weight_diag
            hessian[:nv, :nv] += task_a.T @ (task_w[:, None] * task_a)
            gradient[:nv] += -(task_a.T @ (task_w * task_b))

        # Secondary joint-space task. This keeps the solution close to the
        # velocity planner's arm command even when the Cartesian task is
        # underdetermined or locally singular.
        if self.config.enable_arm_joint_task:
            acc_weight_diag = self._acceleration_weight_diag()
            hessian[:nv, :nv] += selection.T @ (acc_weight_diag[:, None] * selection)
            gradient[:nv] += -(selection.T @ (acc_weight_diag * desired_act_acc))

        torque_weight_diag = np.full(na, self.config.weights.torque, dtype=np.float64)
        torque_rate_weight_diag = np.full(na, self.config.weights.torque_rate, dtype=np.float64)
        hessian[nv:, nv:] += np.diag(torque_weight_diag + torque_rate_weight_diag)
        gradient[nv:] += -(torque_rate_weight_diag * self._prev_tau)

        hessian[:nv, :nv] += self.config.weights.vdot_regularization * np.eye(nv, dtype=np.float64)


        hessian += self.config.qp_condition_regularization * np.eye(nvar, dtype=np.float64)
        hessian = 0.5 * (hessian + hessian.T)

        # S_act (M vdot + C) - tau = 0 -> S_act M vdot - tau = -S_act C
        equality = np.zeros((na, nvar), dtype=np.float64)
        equality[:, :nv] = mass_act
        equality[:, nv:] = -np.eye(na, dtype=np.float64)
        equality_rhs = -bias_act


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
        weights = np.zeros(self.num_motors, dtype=np.float64)
        weights[self._arm_act_idx] = self.config.weights.arm_acc
        return weights

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
