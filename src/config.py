from dataclasses import dataclass
import numpy as np


@dataclass
class RobotConfig:
    wheel_radius: float = 0.0525
    wheel_base:   float = 0.396
    track_width:  float = 0.260

    drive_gear_ratio: float = 1.0
    steering_gear_ratio: float = 1.0

    drive_signs: tuple[float, float, float, float] = (1.0, -1.0, 1.0, -1.0)
    steering_signs: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)

    # 輪子順序：FL, FR, RL, RR
    @property
    def wheel_positions(self) -> np.ndarray:
        L = self.wheel_base
        W = self.track_width
        return np.array([
            [ L / 2.0,  W / 2.0],
            [ L / 2.0, -W / 2.0],
            [-L / 2.0,  W / 2.0],
            [-L / 2.0, -W / 2.0],
        ], dtype=float)