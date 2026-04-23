import typing
import numpy as np
import numpy.typing as npt

from .openarm_idx import (
    OPENARM_NQ,
    OPENARM_NU,
    OPENARM_WORLD_POS,
    OPENARM_WORLD_QUAT,
    OPENARM_JOINT_ALL,
    OPENARM_U_BASE_VX,
    OPENARM_U_BASE_VY,
    OPENARM_U_BASE_WZ,
    OPENARM_U_JOINT_ALL,
)
from .openarm_limit import Q_MIN, Q_MAX, FINITE_Q_MASK
from .utils import quat_normalize, quat_multiply, quat_to_rotmat

if typing.TYPE_CHECKING:
    from .pinnzoo_binding import PinnZooModel


def _delta_quat_from_yaw(yaw_delta: float) -> npt.NDArray[np.float64]:
    half = 0.5 * yaw_delta
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

def _input_quat_normalize_jacobian(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Jacobian of normalize(q) w.r.t. q, evaluated at unit quaternion q.
    Since q should already be normalized in nominal state, this reduces to:
        I - q q^T
    """
    qn = quat_normalize(q)
    return np.eye(4, dtype=np.float64) - np.outer(qn, qn)

def _normalize_jacobian(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]
    n = float(np.linalg.norm(y))
    if n < 1e-12:
        return np.eye(4, dtype=np.float64)
    I = np.eye(4, dtype=np.float64)
    return I / n - np.outer(y, y) / (n ** 3)


def _world_pos_jacobian_wrt_quat_and_u(
    quat: npt.NDArray[np.float64],
    vx: float,
    vy: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    p_dot = R(q) @ [vx, vy, 0]
    returns:
      J_q: (3,4) wrt quat [w,x,y,z]
      J_u_lin: (3,2) wrt [vx, vy]
    """
    w, x, y, z = quat

    J_q = np.array([
        [-2.0 * z * vy,               2.0 * y * vy,                  -4.0 * y * vx + 2.0 * x * vy,   -4.0 * z * vx - 2.0 * w * vy],
        [ 2.0 * z * vx,               2.0 * y * vx - 4.0 * x * vy,   2.0 * x * vx,                    2.0 * w * vx - 4.0 * z * vy],
        [-2.0 * y * vx + 2.0 * x * vy, 2.0 * z * vx + 2.0 * w * vy, -2.0 * w * vx + 2.0 * z * vy,   2.0 * x * vx + 2.0 * y * vy],
    ], dtype=np.float64)

    R = quat_to_rotmat(quat)
    J_u_lin = R[:, :2].copy()

    return J_q, J_u_lin


class OpenArmDynamic:
    def __init__(self, model: 'PinnZooModel'):
        self.model = model
        self.nq = model.nq
        self.nv = model.nv
        self.lib_nx = model.nx

        self.nx = OPENARM_NQ
        self.nu = OPENARM_NU

        if self.nq != OPENARM_NQ:
            raise ValueError(f"Expected model.nq == {OPENARM_NQ}, got {self.nq}")

    def _clip_q_in_place(self, q: npt.NDArray[np.float64]) -> None:
        q[FINITE_Q_MASK] = np.clip(q[FINITE_Q_MASK], Q_MIN[FINITE_Q_MASK], Q_MAX[FINITE_Q_MASK])
        q[OPENARM_WORLD_QUAT] = quat_normalize(q[OPENARM_WORLD_QUAT])

    def control_to_qdot(
        self,
        x: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        assert x.shape == (self.nx,)
        assert u.shape == (self.nu,)

        qdot = np.zeros((self.nx,), dtype=np.float64)

        quat = x[OPENARM_WORLD_QUAT]
        R = quat_to_rotmat(quat)
        v_body = np.array([u[OPENARM_U_BASE_VX], u[OPENARM_U_BASE_VY], 0.0], dtype=np.float64)
        v_world = R @ v_body
        qdot[OPENARM_WORLD_POS] = v_world

        wz = float(u[OPENARM_U_BASE_WZ])
        omega_quat = np.array([0.0, 0.0, 0.0, wz], dtype=np.float64)
        qdot[OPENARM_WORLD_QUAT] = 0.5 * quat_multiply(quat, omega_quat)

        qdot[OPENARM_JOINT_ALL] = u[OPENARM_U_JOINT_ALL]
        return qdot

    def dynamics(
        self,
        x: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        assert x.shape == (self.nx,)
        assert u.shape == (self.nu,)
        return self.control_to_qdot(x, u)

    def dynamics_jacobian(
        self,
        x: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        assert x.shape == (self.nx,)
        assert u.shape == (self.nu,)

        A = np.zeros((self.nx, self.nx), dtype=np.float64)
        B = np.zeros((self.nx, self.nu), dtype=np.float64)

        quat = x[OPENARM_WORLD_QUAT]
        vx = float(u[OPENARM_U_BASE_VX])
        vy = float(u[OPENARM_U_BASE_VY])
        wz = float(u[OPENARM_U_BASE_WZ])

        Pq = _input_quat_normalize_jacobian(quat)

        # ------------------------------------------------------------------
        # position dynamics
        # p_dot = R(normalize(q)) @ [vx, vy, 0]
        # ------------------------------------------------------------------
        J_q_pos_raw, J_u_lin = _world_pos_jacobian_wrt_quat_and_u(quat, vx, vy)
        J_q_pos = J_q_pos_raw @ Pq
        A[OPENARM_WORLD_POS, OPENARM_WORLD_QUAT] = J_q_pos
        B[OPENARM_WORLD_POS, OPENARM_U_BASE_VX] = J_u_lin[:, 0]
        B[OPENARM_WORLD_POS, OPENARM_U_BASE_VY] = J_u_lin[:, 1]

        # ------------------------------------------------------------------
        # quaternion dynamics
        # qdot = 0.5 * (normalize(q) ⊗ [0,0,0,wz])
        # ------------------------------------------------------------------
        w, xq, yq, zq = quat_normalize(quat)

        A_quat_raw = 0.5 * np.array([
            [0.0,  0.0,  0.0, -wz],
            [0.0,  0.0,  wz,  0.0],
            [0.0, -wz,  0.0,  0.0],
            [wz,   0.0,  0.0,  0.0],
        ], dtype=np.float64)
        A_quat = A_quat_raw @ Pq

        B_quat_wz = 0.5 * np.array([-zq, yq, -xq, w], dtype=np.float64)

        A[OPENARM_WORLD_QUAT, OPENARM_WORLD_QUAT] = A_quat
        B[OPENARM_WORLD_QUAT, OPENARM_U_BASE_WZ] = B_quat_wz

        # ------------------------------------------------------------------
        # joint dynamics
        # q_joint_dot = u_joint
        # ------------------------------------------------------------------
        joint_dim = OPENARM_JOINT_ALL.stop - OPENARM_JOINT_ALL.start
        B[OPENARM_JOINT_ALL, OPENARM_U_JOINT_ALL] = np.eye(joint_dim, dtype=np.float64)

        return A, B

    # -------------------------------------------------------------------------
    # Discrete-time dynamics
    # -------------------------------------------------------------------------
    def discrete_dynamics(
        self,
        x: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        assert x.shape == (self.nx,)
        assert u.shape == (self.nu,)

        x_next = x.copy()

        quat = x[OPENARM_WORLD_QUAT]
        R = quat_to_rotmat(quat)
        v_body = np.array([u[OPENARM_U_BASE_VX], u[OPENARM_U_BASE_VY], 0.0], dtype=np.float64)
        v_world = R @ v_body
        x_next[OPENARM_WORLD_POS] += dt * v_world

        dq_yaw = _delta_quat_from_yaw(float(u[OPENARM_U_BASE_WZ]) * dt)
        x_next[OPENARM_WORLD_QUAT] = quat_normalize(quat_multiply(quat, dq_yaw))

        x_next[OPENARM_JOINT_ALL] += dt * u[OPENARM_U_JOINT_ALL]
        self._clip_q_in_place(x_next)
        return x_next

    def discrete_dynamics_jacobian(
        self,
        x: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        assert x.shape == (self.nx,)
        assert u.shape == (self.nu,)

        A = np.eye(self.nx, dtype=np.float64)
        B = np.zeros((self.nx, self.nu), dtype=np.float64)

        quat = x[OPENARM_WORLD_QUAT]
        quat_n = quat_normalize(quat)

        vx = float(u[OPENARM_U_BASE_VX])
        vy = float(u[OPENARM_U_BASE_VY])
        wz = float(u[OPENARM_U_BASE_WZ])

        Pq = _input_quat_normalize_jacobian(quat_n)

        # ------------------------------------------------------------------
        # position update
        # p_next = p + dt * R(normalize(q)) @ [vx, vy, 0]
        # ------------------------------------------------------------------
        J_q_pos_raw, J_u_lin = _world_pos_jacobian_wrt_quat_and_u(quat_n, vx, vy)
        J_q_pos = J_q_pos_raw @ Pq

        A[OPENARM_WORLD_POS, OPENARM_WORLD_QUAT] += dt * J_q_pos
        B[OPENARM_WORLD_POS, OPENARM_U_BASE_VX] = dt * J_u_lin[:, 0]
        B[OPENARM_WORLD_POS, OPENARM_U_BASE_VY] = dt * J_u_lin[:, 1]

        # ------------------------------------------------------------------
        # quaternion update
        # q_next = normalize(normalize(q) ⊗ dq_yaw)
        #
        # chain rule:
        # d q_next / d q_raw
        #   = J_norm_out @ d(qn ⊗ dq)/d qn @ J_norm_in
        # ------------------------------------------------------------------
        delta = wz * dt
        half = 0.5 * delta
        c = float(np.cos(half))
        s = float(np.sin(half))

        w, xq, yq, zq = quat_n

        y = np.array([
            w * c - zq * s,
            xq * c + yq * s,
            yq * c - xq * s,
            zq * c + w * s,
        ], dtype=np.float64)

        J_norm_out = _normalize_jacobian(y)

        dy_dqnorm = np.array([
            [c,   0.0, 0.0, -s],
            [0.0, c,   s,   0.0],
            [0.0, -s,  c,   0.0],
            [s,   0.0, 0.0, c],
        ], dtype=np.float64)

        dy_ddelta = np.array([
            -0.5 * (w * s + zq * c),
            0.5 * (-xq * s + yq * c),
            0.5 * (-yq * s - xq * c),
            0.5 * (-zq * s + w * c),
        ], dtype=np.float64)

        A_quat = J_norm_out @ dy_dqnorm @ Pq
        B_quat_wz = (J_norm_out @ dy_ddelta) * dt

        A[OPENARM_WORLD_QUAT, OPENARM_WORLD_QUAT] = A_quat
        B[OPENARM_WORLD_QUAT, OPENARM_U_BASE_WZ] = B_quat_wz

        # ------------------------------------------------------------------
        # joints with clip-aware Jacobian
        # q_joint_next = clip(q_joint + dt * u_joint, q_min, q_max)
        # ------------------------------------------------------------------
        joint_slice = OPENARM_JOINT_ALL
        u_joint_slice = OPENARM_U_JOINT_ALL
        joint_dim = joint_slice.stop - joint_slice.start
    
        q_joint = x[joint_slice].copy()
        u_joint = u[u_joint_slice].copy()
        q_pre = q_joint + dt * u_joint

        q_min_joint = Q_MIN[joint_slice]
        q_max_joint = Q_MAX[joint_slice]
        finite_joint_mask = FINITE_Q_MASK[joint_slice]

        A_joint_diag = np.ones((joint_dim,), dtype=np.float64)
        B_joint_diag = np.full((joint_dim,), dt, dtype=np.float64)

        clip_tol = 1e-12
        
        lower_active = (
            finite_joint_mask
            & (q_pre <= q_min_joint + clip_tol)
            & (u_joint < 0.0)
        )
        upper_active = (
            finite_joint_mask
            & (q_pre >= q_max_joint - clip_tol)
            & (u_joint > 0.0)
        )
        active_clip = lower_active | upper_active

        A_joint_diag[active_clip] = 0.0
        B_joint_diag[active_clip] = 0.0

        A[joint_slice, joint_slice] = np.diag(A_joint_diag)
        B[joint_slice, u_joint_slice] = np.diag(B_joint_diag)

        return A, B

    def rollout_nominal(
        self,
        x0: npt.NDArray[np.float64],
        U: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        assert x0.shape == (self.nx,)
        assert U.ndim == 2 and U.shape[1] == self.nu

        N = U.shape[0]
        X = np.empty((N + 1, self.nx), dtype=np.float64)
        X[0] = x0
        for k in range(N):
            X[k + 1] = self.discrete_dynamics(X[k], U[k], dt)
        return X