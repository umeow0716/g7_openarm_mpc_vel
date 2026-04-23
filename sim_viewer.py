
import time
import atexit
import mujoco
import mujoco.viewer
import numpy as np

from src.mpc_solver import OpenArmMPCSolver
from plotter import plot_shared_memory_x_target_realtime
from multiprocessing import Process, Event, shared_memory


LEFT_EE_BODY = 'L_gripper_tcp_link'
RIGHT_EE_BODY = 'R_gripper_tcp_link'

X_SHARED_SHAPE = (33,)
U_SHARED_SHAPE = (21,)
LEFT_TARGET_POS_SHARED_SHAPE = (3,)
LEFT_TARGET_QUAT_SHARED_SHAPE = (4,)
RIGHT_TARGET_POS_SHARED_SHAPE = (3,)
RIGHT_TARGET_QUAT_SHARED_SHAPE = (4,)
DTYPE = np.float64

def control_to_mj_qvel(u: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """
    u = [vx, vy, wz, left_arm(9), right_arm(9)]  -> total 21 dims

    MuJoCo qvel:
    [base_vx, base_vy, base_vz, base_wx, base_wy, base_wz,
     8 wheel joint vels,
     18 arm joint vels]
    """
    
    u[-1] = 0.0
    u[-2] = 0.0
    u[-10] = 0.0
    u[-11] = 0.0
    
    qvel = np.zeros(model.nv, dtype=np.float64)

    qvel[0] = u[0]   # vx
    qvel[1] = u[1]   # vy
    qvel[5] = u[2]   # wz

    qvel[14:32] = u[3:21]
    return qvel

def simulation_loop(x_shm_name: str, u_shm_name: str, left_target_pos_shm_name: str, left_target_quat_shm_name: str, right_target_pos_shm_name: str, right_target_quat_shm_name: str):
    x_shm = shared_memory.SharedMemory(name=x_shm_name)
    u_shm = shared_memory.SharedMemory(name=u_shm_name)

    u = np.ndarray(U_SHARED_SHAPE, dtype=DTYPE, buffer=u_shm.buf)

    left_target_pos_shm = shared_memory.SharedMemory(name=left_target_pos_shm_name)
    left_target_quat_shm = shared_memory.SharedMemory(name=left_target_quat_shm_name)
    right_target_pos_shm = shared_memory.SharedMemory(name=right_target_pos_shm_name)
    right_target_quat_shm = shared_memory.SharedMemory(name=right_target_quat_shm_name)

    spec = mujoco.MjSpec.from_file('g7_openarm_mujoco/scene.xml')
    
    left_target = spec.worldbody.add_body(
        name='left_target',
        mocap=True,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    left_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.05],
        rgba=[1, 0, 0, 0.3],
        contype=0,
        conaffinity=0,
    )

    right_target = spec.worldbody.add_body(
        name='right_target',
        mocap=True,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    right_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.05],
        rgba=[0, 0, 1, 0.3],
        contype=0,
        conaffinity=0,
    )
    
    model = spec.compile()
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    left_hand_pos = data.body(LEFT_EE_BODY).xpos.copy()
    right_hand_pos = data.body(RIGHT_EE_BODY).xpos.copy()
    left_hand_quat = data.body(LEFT_EE_BODY).xquat.copy()
    right_hand_quat = data.body(RIGHT_EE_BODY).xquat.copy()
    
    left_target_mocap_id = model.body_mocapid[model.body('left_target').id]
    right_target_mocap_id = model.body_mocapid[model.body('right_target').id]

    data.mocap_pos[left_target_mocap_id] = left_hand_pos
    data.mocap_quat[left_target_mocap_id] = left_hand_quat
    data.mocap_pos[right_target_mocap_id] = right_hand_pos
    data.mocap_quat[right_target_mocap_id] = right_hand_quat


    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            cycle_end_time = time.time() + model.opt.timestep
            
            qvel_cmd = control_to_mj_qvel(u, model)
            data.qvel[:] = qvel_cmd
            mujoco.mj_step(model, data)
            x_shm.buf[:] = data.qpos.copy().astype(DTYPE).tobytes() # type: ignore
            left_target_pos_shm.buf[:] = data.mocap_pos[left_target_mocap_id].astype(DTYPE).tobytes() # type: ignore
            left_target_quat_shm.buf[:] = data.mocap_quat[left_target_mocap_id].astype(DTYPE).tobytes() # type: ignore
            right_target_pos_shm.buf[:] = data.mocap_pos[right_target_mocap_id].astype(DTYPE).tobytes() # type: ignore
            right_target_quat_shm.buf[:] = data.mocap_quat[right_target_mocap_id].astype(DTYPE).tobytes() # type: ignore
            viewer.sync()
            
            now = time.time()
            sleep_time = cycle_end_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    x_shm = shared_memory.SharedMemory(create=True, size=X_SHARED_SHAPE[0] * DTYPE().nbytes)
    u_shm = shared_memory.SharedMemory(create=True, size=U_SHARED_SHAPE[0] * DTYPE().nbytes)
    
    left_pos_target_shm = shared_memory.SharedMemory(create=True, size=LEFT_TARGET_POS_SHARED_SHAPE[0] * DTYPE().nbytes)
    left_quat_target_shm = shared_memory.SharedMemory(create=True, size=LEFT_TARGET_QUAT_SHARED_SHAPE[0] * DTYPE().nbytes)
    right_pos_target_shm = shared_memory.SharedMemory(create=True, size=RIGHT_TARGET_POS_SHARED_SHAPE[0] * DTYPE().nbytes)
    right_quat_target_shm = shared_memory.SharedMemory(create=True, size=RIGHT_TARGET_QUAT_SHARED_SHAPE[0] * DTYPE().nbytes)

    mpc = OpenArmMPCSolver()
    U_init = np.zeros((mpc.N, mpc.nu), dtype=np.float64)
    max_iter = 10

    solve_every = 3
    step_count = 0
    u_cmd = np.zeros((mpc.nu,), dtype=np.float64)
    
    process1 = Process(target=simulation_loop, args=(x_shm.name, u_shm.name, left_pos_target_shm.name, left_quat_target_shm.name, right_pos_target_shm.name, right_quat_target_shm.name))
    process1.start()
    process2 = Process(target=plot_shared_memory_x_target_realtime, args=(x_shm.name, u_shm.name, left_pos_target_shm.name, left_quat_target_shm.name, right_pos_target_shm.name, right_quat_target_shm.name))
    process2.start()
    
    def onexit():
        process1.terminate()
        process2.terminate()
        x_shm.close()
        u_shm.close()
        left_pos_target_shm.close()
        left_quat_target_shm.close()
        right_pos_target_shm.close()
        right_quat_target_shm.close()

    atexit.register(onexit)

    while True:
        step_start = time.time()

        if step_count % solve_every == 0:
            left_pos = np.ndarray(LEFT_TARGET_POS_SHARED_SHAPE, dtype=DTYPE, buffer=left_pos_target_shm.buf)
            left_quat = np.ndarray(LEFT_TARGET_QUAT_SHARED_SHAPE, dtype=DTYPE, buffer=left_quat_target_shm.buf)
            right_pos = np.ndarray(RIGHT_TARGET_POS_SHARED_SHAPE, dtype=DTYPE, buffer=right_pos_target_shm.buf)
            right_quat = np.ndarray(RIGHT_TARGET_QUAT_SHARED_SHAPE, dtype=DTYPE, buffer=right_quat_target_shm.buf)

            target = np.concatenate([
                left_pos, left_quat,
                right_pos, right_quat
            ]).astype(np.float64)

            x = np.ndarray(X_SHARED_SHAPE, dtype=DTYPE, buffer=x_shm.buf)

            _, U_sol, success = mpc.solve_slq(
                x0=x,
                target=target,
                U_init=U_init,
                max_iter=max_iter,
                shift_warm_start=True,
            )
            U_init = U_sol.copy()

            u_cmd = U_sol[0].copy()
            if not success:
                u_cmd[:] = 0.0
            
            step_end = time.time()
            dt_solve = step_end - step_start
            print(f'SLQ step spent {dt_solve:.6f}s ({1.0 / max(dt_solve, 1e-9):.2f} Hz)')

        u_shm.buf[:] = u_cmd.astype(DTYPE).tobytes() # type: ignore

        step_count += 1


if __name__ == "__main__":
    main()