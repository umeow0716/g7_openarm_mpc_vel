import numpy as np

from .config import RobotConfig


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_pi_array(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class SwerveKinematics:
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.pos = cfg.wheel_positions

    def inverse(self, vx: float, vy: float, wz: float):
        deltas = []
        wheel_omegas = []

        for x_i, y_i in self.pos:
            vix = vx - wz * y_i
            viy = vy + wz * x_i

            delta = np.arctan2(viy, vix)
            speed = np.hypot(vix, viy)

            delta = (delta + np.pi) % (2 * np.pi) - np.pi

            omega = speed / self.cfg.wheel_radius

            deltas.append(delta)
            wheel_omegas.append(omega)

        return np.array(deltas), np.array(wheel_omegas)

    def forward(self, steering_angles: np.ndarray, wheel_omegas: np.ndarray):
        steering_angles = np.asarray(steering_angles, dtype=float)
        wheel_omegas = np.asarray(wheel_omegas, dtype=float)

        wheel_speeds = wheel_omegas * self.cfg.wheel_radius

        A = []
        b = []

        for i in range(4):
            rx, ry = self.cfg.wheel_positions[i]

            delta = steering_angles[i]
            speed = wheel_speeds[i]

            c = np.cos(delta)
            s = np.sin(delta)

            vix = speed * c
            viy = speed * s

            A.append([1.0, 0.0, -ry])
            b.append(vix)
            
            A.append([0.0, 1.0, rx])
            b.append(viy)

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        vx, vy, wz = np.linalg.lstsq(A, b, rcond=None)[0]

        return float(vx), float(vy), float(wz)