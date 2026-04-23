from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class TargetData:
    left_target_pos: npt.NDArray[np.float64]
    right_target_pos: npt.NDArray[np.float64]
    left_target_matrix: npt.NDArray[np.float64]
    right_target_matrix: npt.NDArray[np.float64]


@dataclass(slots=True)
class StateEvaluation:
    left_hand_pos: npt.NDArray[np.float64]
    left_hand_quat: npt.NDArray[np.float64]
    right_hand_pos: npt.NDArray[np.float64]
    right_hand_quat: npt.NDArray[np.float64]
    e_p_left: npt.NDArray[np.float64]
    e_R_left: npt.NDArray[np.float64]
    e_p_right: npt.NDArray[np.float64]
    e_R_right: npt.NDArray[np.float64]
    Jkin: npt.NDArray[np.float64] | None = None


@dataclass(slots=True)
class StageQuadratic:
    l: float
    lx: npt.NDArray[np.float64]
    lu: npt.NDArray[np.float64]
    lxx: npt.NDArray[np.float64]
    luu: npt.NDArray[np.float64]
    lux: npt.NDArray[np.float64]


@dataclass(slots=True)
class TerminalQuadratic:
    phi: float
    phix: npt.NDArray[np.float64]
    phixx: npt.NDArray[np.float64]


@dataclass(slots=True)
class TrajectoryQuadratics:
    stages: list[StageQuadratic]
    terminal: TerminalQuadratic
