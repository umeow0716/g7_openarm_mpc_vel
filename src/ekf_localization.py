import numpy as np


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def body_to_global_velocity(
    vx_body: float,
    vy_body: float,
    yaw: float,
) -> tuple[float, float]:
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    vx_global = cy * vx_body - sy * vy_body
    vy_global = sy * vx_body + cy * vy_body

    return float(vx_global), float(vy_global)


def global_to_body_velocity(
    vx_global: float,
    vy_global: float,
    yaw: float,
) -> tuple[float, float]:
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    vx_body = cy * vx_global + sy * vy_global
    vy_body = -sy * vx_global + cy * vy_global

    return float(vx_body), float(vy_body)


class AMREKF:
    """
    State:
        x[0] = px
        x[1] = py
        x[2] = yaw
        x[3] = gyro_z_bias
    """

    def __init__(self):
        self.x = np.zeros(4)

        self.P = np.diag([
            0.05**2,             # px
            0.05**2,             # py
            np.deg2rad(5.0)**2,  # yaw
            np.deg2rad(1.0)**2,  # gyro bias
        ])

        self.Q = np.diag([
            0.02**2,
            0.02**2,
            np.deg2rad(1.0)**2,
            np.deg2rad(0.05)**2,
        ])

        self.R_yaw = np.array([[np.deg2rad(3.0)**2]])
        self.R_gyro = np.array([[np.deg2rad(2.0)**2]])

        # Internal velocity storage.
        self._body_velocity = np.zeros(3)
        self._global_velocity = np.zeros(3)

    def predict_wheel(self, vx: float, vy: float, wz: float, dt: float):
        """
        EKF prediction using wheel odometry.

        Args:
            vx: body-frame x velocity, m/s
            vy: body-frame y velocity, m/s
            wz: body-frame yaw rate, rad/s
            dt: timestep, s
        """
        px, py, yaw, bz = self.x

        vx_global, vy_global = body_to_global_velocity(
            vx_body=vx,
            vy_body=vy,
            yaw=yaw,
        )

        # Store latest velocity inside EKF.
        self._body_velocity = np.array([
            float(vx),
            float(vy),
            float(wz),
        ])

        self._global_velocity = np.array([
            float(vx_global),
            float(vy_global),
            float(wz),
        ])

        px_new = px + vx_global * dt
        py_new = py + vy_global * dt
        yaw_new = wrap_pi(yaw + wz * dt)
        bz_new = bz

        self.x = np.array([px_new, py_new, yaw_new, bz_new])

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        F = np.eye(4)

        F[0, 2] = (-sy * vx - cy * vy) * dt
        F[1, 2] = (cy * vx - sy * vy) * dt

        self.P = F @ self.P @ F.T + self.Q

    def update_yaw(self, yaw_meas: float):
        """
        Update heading using yaw measurement.
        """
        H = np.array([[0.0, 0.0, 1.0, 0.0]])
        y = wrap_pi(yaw_meas - self.x[2])

        S = H @ self.P @ H.T + self.R_yaw
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K.flatten() * y)
        self.x[2] = wrap_pi(self.x[2])

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

        # yaw 被修正後，global velocity 也要用新的 yaw 重新整理一次。
        vx_body, vy_body, wz_body = self._body_velocity
        vx_global, vy_global = body_to_global_velocity(
            vx_body=vx_body,
            vy_body=vy_body,
            yaw=self.x[2],
        )

        self._global_velocity = np.array([
            float(vx_global),
            float(vy_global),
            float(wz_body),
        ])

    def update_gyro_z(self, gyro_z_meas: float, wz_wheel: float):
        """
        Update gyro z bias.

        Model:
            gyro_z_meas ≈ wz_wheel + bias
        """
        H = np.array([[0.0, 0.0, 0.0, 1.0]])
        pred = wz_wheel + self.x[3]
        y = gyro_z_meas - pred

        S = H @ self.P @ H.T + self.R_gyro
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K.flatten() * y)

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    @property
    def pose(self) -> tuple[float, float, float]:
        """
        Returns:
            px, py, yaw
        """
        return (
            float(self.x[0]),
            float(self.x[1]),
            float(self.x[2]),
        )

    @property
    def body_velocity(self) -> tuple[float, float, float]:
        """
        Returns latest body-frame velocity:
            vx_body, vy_body, wz_body
        """
        return (
            float(self._body_velocity[0]),
            float(self._body_velocity[1]),
            float(self._body_velocity[2]),
        )

    @property
    def global_velocity(self) -> tuple[float, float, float]:
        """
        Returns latest global-frame velocity:
            vx_global, vy_global, wz_global

        wz_global 在 2D yaw-only 模型下等同 wz_body。
        """
        return (
            float(self._global_velocity[0]),
            float(self._global_velocity[1]),
            float(self._global_velocity[2]),
        )

    @property
    def gyro_z_bias(self) -> float:
        return float(self.x[3])