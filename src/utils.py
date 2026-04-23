import numpy as np
import numpy.typing as npt

_SMALL_NORM = 1e-12

def quat_normalize(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_conjugate(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(
    q1: npt.NDArray[np.float64],
    q2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_error_matrix_from_target(
    quat_target: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    wt, xt, yt, zt = quat_target
    return np.array([
        [wt,  xt,  yt,  zt],
        [-xt, wt,  zt, -yt],
        [-yt, -zt, wt,  xt],
        [-zt, yt, -xt,  wt],
    ], dtype=np.float64)


def _quat_error_from_matrix(
    quat: npt.NDArray[np.float64],
    target_matrix: npt.NDArray[np.float64],
) -> tuple[float, float, float, float, float]:
    qe = target_matrix @ quat
    if qe[0] < 0.0:
        qe = -qe

    w = float(np.clip(qe[0], -1.0, 1.0))
    vx = float(qe[1])
    vy = float(qe[2])
    vz = float(qe[3])
    vn = float(np.sqrt(vx * vx + vy * vy + vz * vz))
    return w, vx, vy, vz, vn


def quat_orientation_error_from_matrix(
    quat: npt.NDArray[np.float64],
    target_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    w, vx, vy, vz, vn = _quat_error_from_matrix(quat, target_matrix)

    if vn < _SMALL_NORM:
        return np.array([2.0 * vx, 2.0 * vy, 2.0 * vz], dtype=np.float64)

    angle = 2.0 * np.arctan2(vn, w)
    scale = angle / vn
    return np.array([scale * vx, scale * vy, scale * vz], dtype=np.float64)


def quat_orientation_error(
    q: npt.NDArray[np.float64],
    q_des: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return quat_orientation_error_from_matrix(q, quat_error_matrix_from_target(q_des))


def orientation_error_jacobian_wrt_quat_from_matrix(
    quat: npt.NDArray[np.float64],
    target_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    qe_raw = target_matrix @ quat
    sign = -1.0 if qe_raw[0] < 0.0 else 1.0
    qe = sign * qe_raw

    w = float(np.clip(qe[0], -1.0, 1.0))
    vx = float(qe[1])
    vy = float(qe[2])
    vz = float(qe[3])

    vn2 = vx * vx + vy * vy + vz * vz
    vn = float(np.sqrt(vn2))

    dedd = np.zeros((3, 4), dtype=np.float64)

    if vn < _SMALL_NORM:
        dedd[0, 1] = 2.0
        dedd[1, 2] = 2.0
        dedd[2, 3] = 2.0
        return dedd @ (sign * target_matrix)

    denom = w * w + vn2
    angle = 2.0 * np.arctan2(vn, w)
    f = angle / vn
    df_dw = -2.0 / denom
    df_dvn = 2.0 * w / (vn * denom) - angle / vn2
    c = df_dvn / vn

    dedd[0, 0] = df_dw * vx
    dedd[1, 0] = df_dw * vy
    dedd[2, 0] = df_dw * vz

    dedd[0, 1] = f + c * vx * vx
    dedd[0, 2] = c * vx * vy
    dedd[0, 3] = c * vx * vz

    dedd[1, 1] = c * vy * vx
    dedd[1, 2] = f + c * vy * vy
    dedd[1, 3] = c * vy * vz

    dedd[2, 1] = c * vz * vx
    dedd[2, 2] = c * vz * vy
    dedd[2, 3] = f + c * vz * vz

    return dedd @ (sign * target_matrix)


def orientation_error_jacobian_wrt_quat(
    quat: npt.NDArray[np.float64],
    quat_target: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return orientation_error_jacobian_wrt_quat_from_matrix(
        quat,
        quat_error_matrix_from_target(quat_target),
    )

def quat_to_rotmat(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)