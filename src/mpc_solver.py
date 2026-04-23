from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .openarm_idx import *
from .openarm_limit import *
from .pinnzoo_binding import PinnZooModel
from .openarm_dynamic import OpenArmDynamic
from .pinnzoo import kinematics, kinematics_jacobian
from .utils import (
    quat_orientation_error_from_matrix,
    orientation_error_jacobian_wrt_quat_from_matrix,
    quat_error_matrix_from_target,
)
from .mpc_types import (
    TargetData,
    StateEvaluation,
    StageQuadratic,
    TerminalQuadratic,
    TrajectoryQuadratics,
)


class OpenArmMPCSolver:
    def __init__(
        self,
        lib_path: str = 'include/libg7_openarm_quat.so',
    ) -> None:
        self.lib_path = lib_path
        self.model = PinnZooModel(self.lib_path)
        self.dynamic = OpenArmDynamic(self.model)

        self.nx = OPENARM_NQ
        self.nu = OPENARM_NU
        self.lib_nx = self.model.nx

        self.horizon = 0.2
        self.dt = 0.01
        self.N = int(round(self.horizon / self.dt))

        self.pos_weight = 400.0
        self.rot_weight = 25.0
        
        self.u_weight = 10.0
        self.base_u_weight = 40.0

        self.R_diag = np.full((self.nu,), self.u_weight, dtype=np.float64)
        self.R_diag[0] = self.base_u_weight
        self.R_diag[1] = self.base_u_weight
        self.R_diag[2] = self.base_u_weight

        self.R = np.diag(self.R_diag)
        self._luu_const = self.R.copy()
        self._lux_zero = np.zeros((self.nu, self.nx), dtype=np.float64)

        self._line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)
        self._cost_tol = 1e-9
        self._reg = 1e-8

    def _to_lib_state(
        self,
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        x_lib = np.zeros((self.lib_nx,), dtype=np.float64)
        x_lib[:self.nx] = x
        return x_lib

    def build_target_data(
        self,
        target: npt.NDArray[np.float64],
    ) -> TargetData:
        left_target_quat = target[3:7]
        right_target_quat = target[10:14]
        return TargetData(
            left_target_pos=target[0:3].copy(),
            right_target_pos=target[7:10].copy(),
            left_target_matrix=quat_error_matrix_from_target(left_target_quat),
            right_target_matrix=quat_error_matrix_from_target(right_target_quat),
        )

    def compute_kinematics_jacobian(
        self,
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        x_lib = self._to_lib_state(x)
        return kinematics_jacobian(self.model, x_lib)[:, :self.nx]

    def evaluate_state(
        self,
        x: npt.NDArray[np.float64],
        target: TargetData,
        *,
        need_jacobian: bool,
    ) -> StateEvaluation:
        x_lib = self._to_lib_state(x)
        kin = kinematics(self.model, x_lib)
        Jkin = self.compute_kinematics_jacobian(x) if need_jacobian else None

        left_hand_pos = kin[OPENARM_LEFT_HAND_POS].copy()
        left_hand_quat = kin[OPENARM_LEFT_HAND_QUAT].copy()
        right_hand_pos = kin[OPENARM_RIGHT_HAND_POS].copy()
        right_hand_quat = kin[OPENARM_RIGHT_HAND_QUAT].copy()

        e_p_left = left_hand_pos - target.left_target_pos
        e_p_right = right_hand_pos - target.right_target_pos
        e_R_left = quat_orientation_error_from_matrix(left_hand_quat, target.left_target_matrix)
        e_R_right = quat_orientation_error_from_matrix(right_hand_quat, target.right_target_matrix)

        return StateEvaluation(
            left_hand_pos=left_hand_pos,
            left_hand_quat=left_hand_quat,
            right_hand_pos=right_hand_pos,
            right_hand_quat=right_hand_quat,
            e_p_left=e_p_left,
            e_R_left=e_R_left,
            e_p_right=e_p_right,
            e_R_right=e_R_right,
            Jkin=Jkin,
        )

    def state_error_cost(
        self,
        evaluation: StateEvaluation,
    ) -> float:
        return float(
            0.5 * self.pos_weight * (evaluation.e_p_left @ evaluation.e_p_left)
            + 0.5 * self.rot_weight * (evaluation.e_R_left @ evaluation.e_R_left)
            + 0.5 * self.pos_weight * (evaluation.e_p_right @ evaluation.e_p_right)
            + 0.5 * self.rot_weight * (evaluation.e_R_right @ evaluation.e_R_right)
        )

    def control_cost(
        self,
        u: npt.NDArray[np.float64],
    ) -> float:
        return float(0.5 * np.dot(self.R_diag, u * u))

    def stage_cost(
        self,
        x: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
        target: TargetData,
    ) -> float:
        evaluation = self.evaluate_state(x, target, need_jacobian=False)
        return self.state_error_cost(evaluation) + self.control_cost(u)

    def terminal_cost(
        self,
        x: npt.NDArray[np.float64],
        target: TargetData,
    ) -> float:
        evaluation = self.evaluate_state(x, target, need_jacobian=False)
        return self.state_error_cost(evaluation)

    def stage_derivatives(
        self,
        x: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
        target: TargetData,
    ) -> StageQuadratic:
        evaluation = self.evaluate_state(x, target, need_jacobian=True)
        if evaluation.Jkin is None:
            raise ValueError('stage derivatives require evaluation.Jkin')

        Jkin = evaluation.Jkin
        l = self.state_error_cost(evaluation) + self.control_cost(u)
        lx = np.zeros((self.nx,), dtype=np.float64)
        lxx = np.zeros((self.nx, self.nx), dtype=np.float64)

        Jp_left = Jkin[OPENARM_LEFT_HAND_POS, :]
        Jp_right = Jkin[OPENARM_RIGHT_HAND_POS, :]

        lx += self.pos_weight * (Jp_left.T @ evaluation.e_p_left)
        lx += self.pos_weight * (Jp_right.T @ evaluation.e_p_right)
        lxx += self.pos_weight * (Jp_left.T @ Jp_left)
        lxx += self.pos_weight * (Jp_right.T @ Jp_right)

        Jq_left = Jkin[OPENARM_LEFT_HAND_QUAT, :]
        Jq_right = Jkin[OPENARM_RIGHT_HAND_QUAT, :]

        Jr_left = orientation_error_jacobian_wrt_quat_from_matrix(
            evaluation.left_hand_quat,
            target.left_target_matrix,
        ) @ Jq_left
        Jr_right = orientation_error_jacobian_wrt_quat_from_matrix(
            evaluation.right_hand_quat,
            target.right_target_matrix,
        ) @ Jq_right

        lx += self.rot_weight * (Jr_left.T @ evaluation.e_R_left)
        lx += self.rot_weight * (Jr_right.T @ evaluation.e_R_right)
        lxx += self.rot_weight * (Jr_left.T @ Jr_left)
        lxx += self.rot_weight * (Jr_right.T @ Jr_right)

        lxx = 0.5 * (lxx + lxx.T)
        return StageQuadratic(
            l=l,
            lx=lx,
            lu=self.R_diag * u,
            lxx=lxx,
            luu=self._luu_const,
            lux=self._lux_zero,
        )

    def terminal_derivatives(
        self,
        x: npt.NDArray[np.float64],
        target: TargetData,
    ) -> TerminalQuadratic:
        evaluation = self.evaluate_state(x, target, need_jacobian=True)
        if evaluation.Jkin is None:
            raise ValueError('terminal derivatives require evaluation.Jkin')

        Jkin = evaluation.Jkin
        phi = self.state_error_cost(evaluation)
        phix = np.zeros((self.nx,), dtype=np.float64)
        phixx = np.zeros((self.nx, self.nx), dtype=np.float64)

        Jp_left = Jkin[OPENARM_LEFT_HAND_POS, :]
        Jp_right = Jkin[OPENARM_RIGHT_HAND_POS, :]

        phix += self.pos_weight * (Jp_left.T @ evaluation.e_p_left)
        phix += self.pos_weight * (Jp_right.T @ evaluation.e_p_right)
        phixx += self.pos_weight * (Jp_left.T @ Jp_left)
        phixx += self.pos_weight * (Jp_right.T @ Jp_right)

        Jq_left = Jkin[OPENARM_LEFT_HAND_QUAT, :]
        Jq_right = Jkin[OPENARM_RIGHT_HAND_QUAT, :]

        Jr_left = orientation_error_jacobian_wrt_quat_from_matrix(
            evaluation.left_hand_quat,
            target.left_target_matrix,
        ) @ Jq_left
        Jr_right = orientation_error_jacobian_wrt_quat_from_matrix(
            evaluation.right_hand_quat,
            target.right_target_matrix,
        ) @ Jq_right

        phix += self.rot_weight * (Jr_left.T @ evaluation.e_R_left)
        phix += self.rot_weight * (Jr_right.T @ evaluation.e_R_right)
        phixx += self.rot_weight * (Jr_left.T @ Jr_left)
        phixx += self.rot_weight * (Jr_right.T @ Jr_right)

        phixx = 0.5 * (phixx + phixx.T)
        return TerminalQuadratic(
            phi=phi,
            phix=phix,
            phixx=phixx,
        )

    def evaluate_trajectory(
        self,
        X: npt.NDArray[np.float64],
        U: npt.NDArray[np.float64],
        target: TargetData,
    ) -> TrajectoryQuadratics:
        N = U.shape[0]
        stages = [self.stage_derivatives(X[k], U[k], target) for k in range(N)]
        terminal = self.terminal_derivatives(X[N], target)
        return TrajectoryQuadratics(stages=stages, terminal=terminal)

    def cost_from_trajectory(
        self,
        trajectory: TrajectoryQuadratics,
    ) -> float:
        J = 0.0
        for stage in trajectory.stages:
            J += self.dt * stage.l
        J += trajectory.terminal.phi
        return float(J)

    def rollout_nominal(
        self,
        x0: npt.NDArray[np.float64],
        U: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self.dynamic.rollout_nominal(x0, U, self.dt)

    def shift_warm_start(
        self,
        U: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        U_shift = np.empty_like(U)
        U_shift[:-1] = U[1:]
        U_shift[-1] = U[-1]
        return U_shift

    def backward_pass(
        self,
        X: npt.NDArray[np.float64],
        U: npt.NDArray[np.float64],
        target: TargetData,
        *,
        reg: float = 1e-10,
        trajectory: TrajectoryQuadratics | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        TrajectoryQuadratics,
    ]:
        N = U.shape[0]
        k_seq = np.zeros((N, self.nu), dtype=np.float64)
        K_seq = np.zeros((N, self.nu, self.nx), dtype=np.float64)
        dV_linear = np.zeros((N,), dtype=np.float64)
        dV_quadratic = np.zeros((N,), dtype=np.float64)

        if trajectory is None:
            trajectory = self.evaluate_trajectory(X, U, target)

        Vx = trajectory.terminal.phix.copy()
        Vxx = trajectory.terminal.phixx.copy()
        reg_eye = reg * np.eye(self.nu, dtype=np.float64)

        for k in range(N - 1, -1, -1):
            stage = trajectory.stages[k]
            A, B = self.dynamic.discrete_dynamics_jacobian(X[k], U[k], self.dt)

            lx = stage.lx * self.dt
            lu = stage.lu * self.dt
            lxx = stage.lxx * self.dt
            luu = stage.luu * self.dt
            lux = stage.lux * self.dt

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            Quu = 0.5 * (Quu + Quu.T)
            Quu_reg = Quu + reg_eye

            k_ff = -np.linalg.solve(Quu_reg, Qu)
            K_fb = -np.linalg.solve(Quu_reg, Qux)

            k_seq[k] = k_ff
            K_seq[k] = K_fb
            dV_linear[k] = Qu @ k_ff
            dV_quadratic[k] = 0.5 * (k_ff @ Quu @ k_ff)

            Vx = Qx + K_fb.T @ Quu @ k_ff + K_fb.T @ Qu + Qux.T @ k_ff
            Vxx = Qxx + K_fb.T @ Quu @ K_fb + K_fb.T @ Qux + Qux.T @ K_fb
            Vxx = 0.5 * (Vxx + Vxx.T)

        return k_seq, K_seq, dV_linear, dV_quadratic, trajectory

    def forward_pass(
        self,
        x0: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        U: npt.NDArray[np.float64],
        k_seq: npt.NDArray[np.float64],
        K_seq: npt.NDArray[np.float64],
        alpha: float,
        target: TargetData,
        *,
        compute_cost: bool = False,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float | None,
    ]:
        N = U.shape[0]
        X_new = np.empty_like(X)
        U_new = np.empty_like(U)
        X_new[0] = x0
        J_new = 0.0 if compute_cost else None

        for k in range(N):
            dx = X_new[k] - X[k]
            uk = U[k] + alpha * k_seq[k] + K_seq[k] @ dx
            uk = np.clip(uk, U_MIN, U_MAX)
            U_new[k] = uk

            if J_new is not None:
                J_new += self.dt * self.stage_cost(X_new[k], uk, target)

            X_new[k + 1] = self.dynamic.discrete_dynamics(X_new[k], uk, self.dt)

        if J_new is not None:
            J_new += self.terminal_cost(X_new[N], target)

        return X_new, U_new, J_new

    def solve_slq(
        self,
        x0: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        U_init: npt.NDArray[np.float64] | None = None,
        max_iter: int = 20,
        shift_warm_start: bool = False,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
        target_data = self.build_target_data(target)

        if U_init is None:
            U = np.zeros((self.N, self.nu), dtype=np.float64)
        else:
            U = self.shift_warm_start(U_init) if shift_warm_start else U_init.copy()

        X = self.rollout_nominal(x0, U)
        trajectory = self.evaluate_trajectory(X, U, target_data)
        J = self.cost_from_trajectory(trajectory)

        accepted_total = 0

        for _ in range(max_iter):
            k_seq, K_seq, dV_linear, dV_quadratic, trajectory = self.backward_pass(
                X,
                U,
                target_data,
                reg=self._reg,
                trajectory=trajectory,
            )

            accepted = False

            for alpha in self._line_search_alphas:
                expected_reduction = -alpha * float(np.sum(dV_linear)) - (alpha * alpha) * float(np.sum(dV_quadratic))
                if expected_reduction < 0.0:
                    continue

                X_trial, U_trial, J_trial = self.forward_pass(
                    x0,
                    X,
                    U,
                    k_seq,
                    K_seq,
                    alpha,
                    target_data,
                    compute_cost=True,
                )

                actual_reduction = J - J_trial

                if actual_reduction > self._cost_tol:
                    X = X_trial
                    U = U_trial
                    J = J_trial
                    trajectory = self.evaluate_trajectory(X, U, target_data)
                    accepted = True
                    break

            if not accepted:
                break
            accepted_total += 1

        success = accepted_total > 0
        return X, U, success
