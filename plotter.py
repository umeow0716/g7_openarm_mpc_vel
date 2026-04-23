import time
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from multiprocessing import shared_memory

from src.pinnzoo import kinematics
from src.pinnzoo_binding import PinnZooModel

from src.utils import (
    quat_orientation_error_from_matrix,
    quat_error_matrix_from_target,
)
from src.openarm_idx import (
    OPENARM_LEFT_HAND_POS,
    OPENARM_LEFT_HAND_QUAT,
    OPENARM_RIGHT_HAND_POS,
    OPENARM_RIGHT_HAND_QUAT,
)

X_SHARED_SHAPE = (33,)
U_SHARED_SHAPE = (21,)
LEFT_TARGET_POS_SHARED_SHAPE = (3,)
LEFT_TARGET_QUAT_SHARED_SHAPE = (4,)
RIGHT_TARGET_POS_SHARED_SHAPE = (3,)
RIGHT_TARGET_QUAT_SHARED_SHAPE = (4,)
DTYPE = np.float64

def plot_shared_memory_x_target_realtime(
    x_shm_name: str,
    u_shm_name: str,
    left_target_pos_shm_name: str,
    left_target_quat_shm_name: str,
    right_target_pos_shm_name: str,
    right_target_quat_shm_name: str,
    *,
    x_shared_shape: tuple[int, ...] = X_SHARED_SHAPE,
    u_shared_shape: tuple[int, ...] = U_SHARED_SHAPE,
    left_target_pos_shared_shape: tuple[int, ...] = LEFT_TARGET_POS_SHARED_SHAPE,
    left_target_quat_shared_shape: tuple[int, ...] = LEFT_TARGET_QUAT_SHARED_SHAPE,
    right_target_pos_shared_shape: tuple[int, ...] = RIGHT_TARGET_POS_SHARED_SHAPE,
    right_target_quat_shared_shape: tuple[int, ...] = RIGHT_TARGET_QUAT_SHARED_SHAPE,
    dtype=DTYPE,
    lib_path: str = 'include/libg7_openarm_quat.so',
    poll_dt: float = 0.03,
    max_samples: int = 3000,
    position_png_path: str = 'plotter_result/position_realtime.png',
    velocity_png_path: str = 'plotter_result/velocity_realtime.png',
    error_png_path: str = 'plotter_result/error_realtime.png',
) -> None:
    x_shm = shared_memory.SharedMemory(name=x_shm_name)
    u_shm = shared_memory.SharedMemory(name=u_shm_name)
    left_target_pos_shm = shared_memory.SharedMemory(name=left_target_pos_shm_name)
    left_target_quat_shm = shared_memory.SharedMemory(name=left_target_quat_shm_name)
    right_target_pos_shm = shared_memory.SharedMemory(name=right_target_pos_shm_name)
    right_target_quat_shm = shared_memory.SharedMemory(name=right_target_quat_shm_name)

    model = PinnZooModel(lib_path)

    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        q = q.astype(np.float64)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / n

    def _quat_rot_error(q_current: np.ndarray, q_target: np.ndarray) -> float:
        qc = _normalize_quat(q_current)
        qt = _normalize_quat(q_target)
        dot = float(np.abs(np.dot(qc, qt)))
        dot = np.clip(dot, -1.0, 1.0)
        return float(2.0 * np.arccos(dot))

    def _read_target():
        left_pos = np.ndarray(left_target_pos_shared_shape, dtype=dtype, buffer=left_target_pos_shm.buf).copy()
        left_quat = np.ndarray(left_target_quat_shared_shape, dtype=dtype, buffer=left_target_quat_shm.buf).copy()
        right_pos = np.ndarray(right_target_pos_shared_shape, dtype=dtype, buffer=right_target_pos_shm.buf).copy()
        right_quat = np.ndarray(right_target_quat_shared_shape, dtype=dtype, buffer=right_target_quat_shm.buf).copy()
        return left_pos, left_quat, right_pos, right_quat

    def _state_to_kinematics_input(x_raw: np.ndarray) -> np.ndarray:
        x_lib = np.zeros((model.nx,), dtype=np.float64)
        n = min(model.nx, x_raw.shape[0])
        x_lib[:n] = x_raw[:n]
        return x_lib

    def _grid_shape(n: int, ncols: int = 4) -> tuple[int, int]:
        ncols = min(ncols, max(1, n))
        nrows = int(np.ceil(n / ncols))
        return nrows, ncols

    def _set_axis_limits(ax, t_list, y_list):
        if len(t_list) == 0 or len(y_list) == 0:
            return

        if len(t_list) == 1:
            ax.set_xlim(max(0.0, t_list[0] - 1.0), t_list[0] + 1.0)
        else:
            ax.set_xlim(t_list[0], t_list[-1])

        y_min = float(np.min(y_list))
        y_max = float(np.max(y_list))
        span = y_max - y_min

        if span < 1e-9:
            pad = 1e-3 if abs(y_max) < 1e-9 else 0.1 * abs(y_max)
        else:
            pad = 0.1 * span

        ax.set_ylim(y_min - pad, y_max + pad)

    def _make_series_figure(window_name: str, prefix: str, n_series: int):
        nrows, ncols = _grid_shape(n_series, ncols=4)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            num=window_name,
            figsize=(18, max(8, 2.1 * nrows)),
            squeeze=False,
        )
        fig.subplots_adjust(left=0.05, right=0.70, top=0.95, bottom=0.05, hspace=0.55, wspace=0.35)

        axes_flat = axes.ravel()
        lines = []

        for i in range(n_series):
            ax = axes_flat[i]
            line = ax.plot([], [], linewidth=1.5)[0]
            ax.set_title(f'{prefix}[{i}]', fontsize=10)
            ax.set_xlabel('time [s]', fontsize=8)
            ax.set_ylabel(prefix, fontsize=8)
            ax.grid(True)
            lines.append(line)

        for i in range(n_series, len(axes_flat)):
            axes_flat[i].axis('off')

        text_left = fig.text(
            0.73, 0.98, '',
            ha='left', va='top',
            family='monospace', fontsize=8,
        )
        text_right = fig.text(
            0.86, 0.98, '',
            ha='left', va='top',
            family='monospace', fontsize=8,
        )

        return fig, axes_flat[:n_series], lines, text_left, text_right

    def _make_value_text(title: str, prefix: str, values: np.ndarray) -> tuple[str, str]:
        lines = [f'{title}']
        lines.extend([f'{prefix}[{i:02d}] = {float(values[i]): .6f}' for i in range(values.shape[0])])

        split_idx = (len(lines) + 1) // 2
        left_text = '\n'.join(lines[:split_idx])
        right_text = '\n'.join(lines[split_idx:])
        return left_text, right_text

    plt.ion()

    fig_pos, pos_axes, pos_lines, pos_text_l, pos_text_r = _make_series_figure('Position', 'x', x_shared_shape[0])
    fig_vel, vel_axes, vel_lines, vel_text_l, vel_text_r = _make_series_figure('Velocity', 'u', u_shared_shape[0])

    fig_err, axs_err = plt.subplots(5, 1, sharex=True, num='Error', figsize=(14, 14))
    fig_err.subplots_adjust(left=0.08, right=0.76, top=0.96, bottom=0.06, hspace=0.55)

    err_titles = [
        ('Position Distance Error', 'distance [m]'),
        ('Position Error X', 'x [m]'),
        ('Position Error Y', 'y [m]'),
        ('Position Error Z', 'z [m]'),
        ('Rotation Error', 'rot [rad]'),
    ]
    for ax, (title, ylabel) in zip(axs_err, err_titles):
        ax.set_title(title)
        ax.set_xlabel('time [s]')
        ax.set_ylabel(ylabel)
        ax.grid(True)

    err_text = fig_err.text(
        0.79, 0.98, '',
        ha='left', va='top',
        family='monospace', fontsize=9,
    )

    try:
        x0 = np.ndarray(x_shared_shape, dtype=dtype, buffer=x_shm.buf).copy()
        u0 = np.ndarray(u_shared_shape, dtype=dtype, buffer=u_shm.buf).copy()

        q_dim = x0.shape[0]
        u_dim = u0.shape[0]

        t_hist = deque(maxlen=max_samples)
        q_hist = [deque(maxlen=max_samples) for _ in range(q_dim)]
        v_hist = [deque(maxlen=max_samples) for _ in range(u_dim)]

        err_dist_l_hist = deque(maxlen=max_samples)
        err_dist_r_hist = deque(maxlen=max_samples)
        err_x_l_hist = deque(maxlen=max_samples)
        err_x_r_hist = deque(maxlen=max_samples)
        err_y_l_hist = deque(maxlen=max_samples)
        err_y_r_hist = deque(maxlen=max_samples)
        err_z_l_hist = deque(maxlen=max_samples)
        err_z_r_hist = deque(maxlen=max_samples)
        err_rot_l_hist = deque(maxlen=max_samples)
        err_rot_r_hist = deque(maxlen=max_samples)

        line_err_dist_l = axs_err[0].plot([], [], label='left', linewidth=1.5)[0]
        line_err_dist_r = axs_err[0].plot([], [], label='right', linewidth=1.5)[0]
        line_err_x_l = axs_err[1].plot([], [], label='left', linewidth=1.5)[0]
        line_err_x_r = axs_err[1].plot([], [], label='right', linewidth=1.5)[0]
        line_err_y_l = axs_err[2].plot([], [], label='left', linewidth=1.5)[0]
        line_err_y_r = axs_err[2].plot([], [], label='right', linewidth=1.5)[0]
        line_err_z_l = axs_err[3].plot([], [], label='left', linewidth=1.5)[0]
        line_err_z_r = axs_err[3].plot([], [], label='right', linewidth=1.5)[0]
        line_err_rot_l = axs_err[4].plot([], [], label='left', linewidth=1.5)[0]
        line_err_rot_r = axs_err[4].plot([], [], label='right', linewidth=1.5)[0]
        

        for ax in axs_err:
            ax.legend(loc='upper right')

        start_time = time.perf_counter()

        while (
            plt.fignum_exists(fig_pos.number)
            and plt.fignum_exists(fig_vel.number)
            and plt.fignum_exists(fig_err.number)
        ):
            now = time.perf_counter()

            x = np.ndarray(x_shared_shape, dtype=dtype, buffer=x_shm.buf).copy()
            u = np.ndarray(u_shared_shape, dtype=dtype, buffer=u_shm.buf).copy()

            left_target_pos, left_target_quat, right_target_pos, right_target_quat = _read_target()

            kin = kinematics(model, _state_to_kinematics_input(x))

            left_pos = kin[OPENARM_LEFT_HAND_POS].copy()
            left_quat = kin[OPENARM_LEFT_HAND_QUAT].copy()
            right_pos = kin[OPENARM_RIGHT_HAND_POS].copy()
            right_quat = kin[OPENARM_RIGHT_HAND_QUAT].copy()
            
            left_target_matrix = quat_error_matrix_from_target(left_target_quat.astype(np.float64))
            right_target_matrix = quat_error_matrix_from_target(right_target_quat.astype(np.float64))

            e_r_left_raw = quat_orientation_error_from_matrix(left_quat, left_target_matrix)
            e_r_right_raw = quat_orientation_error_from_matrix(right_quat, right_target_matrix)

            e_p_left = left_pos - left_target_pos
            e_p_right = right_pos - right_target_pos
            e_r_left = _quat_rot_error(left_quat, left_target_quat)
            e_r_right = _quat_rot_error(right_quat, right_target_quat)

            t_now = now - start_time
            t_hist.append(t_now)

            for i in range(q_dim):
                q_hist[i].append(float(x[i]))
            for i in range(u_dim):
                v_hist[i].append(float(u[i]))

            err_dist_l_hist.append(float(np.linalg.norm(e_p_left)))
            err_dist_r_hist.append(float(np.linalg.norm(e_p_right)))
            err_x_l_hist.append(float(e_p_left[0]))
            err_x_r_hist.append(float(e_p_right[0]))
            err_y_l_hist.append(float(e_p_left[1]))
            err_y_r_hist.append(float(e_p_right[1]))
            err_z_l_hist.append(float(e_p_left[2]))
            err_z_r_hist.append(float(e_p_right[2]))
            err_rot_l_hist.append(float(e_r_left))
            err_rot_r_hist.append(float(e_r_right))

            t_list = list(t_hist)

            for i in range(q_dim):
                y_list = list(q_hist[i])
                pos_lines[i].set_data(t_list, y_list)
                _set_axis_limits(pos_axes[i], t_list, y_list)

            for i in range(u_dim):
                y_list = list(v_hist[i])
                vel_lines[i].set_data(t_list, y_list)
                _set_axis_limits(vel_axes[i], t_list, y_list)

            line_err_dist_l.set_data(t_list, list(err_dist_l_hist))
            line_err_dist_r.set_data(t_list, list(err_dist_r_hist))
            line_err_x_l.set_data(t_list, list(err_x_l_hist))
            line_err_x_r.set_data(t_list, list(err_x_r_hist))
            line_err_y_l.set_data(t_list, list(err_y_l_hist))
            line_err_y_r.set_data(t_list, list(err_y_r_hist))
            line_err_z_l.set_data(t_list, list(err_z_l_hist))
            line_err_z_r.set_data(t_list, list(err_z_r_hist))
            line_err_rot_l.set_data(t_list, list(err_rot_l_hist))
            line_err_rot_r.set_data(t_list, list(err_rot_r_hist))

            _set_axis_limits(axs_err[0], t_list, list(err_dist_l_hist) + list(err_dist_r_hist))
            _set_axis_limits(axs_err[1], t_list, list(err_x_l_hist) + list(err_x_r_hist))
            _set_axis_limits(axs_err[2], t_list, list(err_y_l_hist) + list(err_y_r_hist))
            _set_axis_limits(axs_err[3], t_list, list(err_z_l_hist) + list(err_z_r_hist))
            _set_axis_limits(axs_err[4], t_list, list(err_rot_l_hist) + list(err_rot_r_hist))

            pos_left_text, pos_right_text = _make_value_text('Position', 'x', x)
            vel_left_text, vel_right_text = _make_value_text('Velocity', 'u', u)

            pos_text_l.set_text(pos_left_text)
            pos_text_r.set_text(pos_right_text)
            vel_text_l.set_text(vel_left_text)
            vel_text_r.set_text(vel_right_text)

            err_text.set_text(
                '\n'.join([
                    'Error',
                    f'L dist = {np.linalg.norm(e_p_left): .6f}',
                    f'R dist = {np.linalg.norm(e_p_right): .6f}',
                    f'L ex   = {e_p_left[0]: .6f}',
                    f'R ex   = {e_p_right[0]: .6f}',
                    f'L ey   = {e_p_left[1]: .6f}',
                    f'R ey   = {e_p_right[1]: .6f}',
                    f'L ez   = {e_p_left[2]: .6f}',
                    f'R ez   = {e_p_right[2]: .6f}',
                    f'L rot  = {e_r_left: .6f}',
                    f'R rot  = {e_r_right: .6f}',
                    '',
                    'MPC solver rot err raw',
                    'Euler Angle, ndarray shape=(3,), unit=rad',
                    f'L rot raw = {np.array2string(e_r_left_raw, precision=6, suppress_small=False)}',
                    f'R rot raw = {np.array2string(e_r_right_raw, precision=6, suppress_small=False)}',
                    '',
                    'Definition',
                    'rot_err[0] = theta * u_x [rad]',
                    'rot_err[1] = theta * u_y [rad]',
                    'rot_err[2] = theta * u_z [rad]',
                    '||rot_err|| = theta [rad]',
                    '',
                    'Current EE',
                    f'L pos  = [{left_pos[0]: .4f}, {left_pos[1]: .4f}, {left_pos[2]: .4f}]',
                    f'R pos  = [{right_pos[0]: .4f}, {right_pos[1]: .4f}, {right_pos[2]: .4f}]',
                    '',
                    'Target EE',
                    f'L pos  = [{left_target_pos[0]: .4f}, {left_target_pos[1]: .4f}, {left_target_pos[2]: .4f}]',
                    f'R pos  = [{right_target_pos[0]: .4f}, {right_target_pos[1]: .4f}, {right_target_pos[2]: .4f}]',
                ])
            )

            fig_pos.canvas.draw_idle()
            fig_vel.canvas.draw_idle()
            fig_err.canvas.draw_idle()

            plt.pause(poll_dt)
    except KeyboardInterrupt:
        print('saving figures and exiting...')
        if 'fig_pos' in locals():
            fig_pos.savefig(position_png_path, dpi=150, bbox_inches='tight')
        if 'fig_vel' in locals():
            fig_vel.savefig(velocity_png_path, dpi=150, bbox_inches='tight')
        if 'fig_err' in locals():
            fig_err.savefig(error_png_path, dpi=150, bbox_inches='tight')
        plt.ioff()