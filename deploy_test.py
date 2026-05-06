import os
import time
import atexit
import signal
import mujoco
import mujoco.viewer
import numpy as np
import openarm_can as oa
from openarm_can import MITParam, PosVelParam

from src.low_level_controller_wbc import LowLevelController
from src.mpc_solver import OpenArmMPCSolver
from plotter import plot_shared_memory_x_target_realtime
from multiprocessing import Process, shared_memory


LEFT_EE_BODY = 'L_gripper_tcp_link'
RIGHT_EE_BODY = 'R_gripper_tcp_link'

X_SHARED_SHAPE = (33,)
Q_SHARED_SHAPE = (33,)
V_SHARED_SHAPE = (32,)
U_SHARED_SHAPE = (21,)
LEFT_TARGET_POS_SHARED_SHAPE = (3,)
LEFT_TARGET_QUAT_SHARED_SHAPE = (4,)
RIGHT_TARGET_POS_SHARED_SHAPE = (3,)
RIGHT_TARGET_QUAT_SHARED_SHAPE = (4,)
DTYPE = np.float64

MIT_KP_SHAPE = (26,)
MIT_KD_SHAPE = (26,)
MIT_POS_DES_SHAPE = (26,)
MIT_VEL_DES_SHAPE = (26,)
MIT_TAU_DES_SHAPE = (26,)

# MITCommand actuator order is not the same as raw MuJoCo qpos[7:] / qvel[6:].
# Controller actuator order:
#   [FL_steer, FR_steer, RL_steer, RR_steer,
#    FL_wheel, FR_wheel, RL_wheel, RR_wheel, arms...]
# MuJoCo joint order after the floating base is interleaved:
#   [FL_steer, FL_wheel, FR_steer, FR_wheel,
#    RL_steer, RL_wheel, RR_steer, RR_wheel, arms...]
ACTUATOR_QPOS_IDX = np.array([7, 9, 11, 13, 8, 10, 12, 14, *range(15, 33)], dtype=np.int32)
ACTUATOR_QVEL_IDX = np.array([6, 8, 10, 12, 7, 9, 11, 13, *range(14, 32)], dtype=np.int32)

def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

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



def simulation_loop(q_shm_name: str, left_target_pos_shm_name: str, left_target_quat_shm_name: str, right_target_pos_shm_name: str, right_target_quat_shm_name: str):
    q_shm = shared_memory.SharedMemory(name=q_shm_name)

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
    
    left_target_mocap_id = model.body_mocapid[model.body('left_target').id]
    right_target_mocap_id = model.body_mocapid[model.body('right_target').id]
    
    left_hand_pos = data.body(LEFT_EE_BODY).xpos.copy()
    right_hand_pos = data.body(RIGHT_EE_BODY).xpos.copy()
    left_hand_quat = data.body(LEFT_EE_BODY).xquat.copy()
    right_hand_quat = data.body(RIGHT_EE_BODY).xquat.copy()

    data.mocap_pos[left_target_mocap_id] = left_hand_pos
    data.mocap_quat[left_target_mocap_id] = left_hand_quat
    data.mocap_pos[right_target_mocap_id] = right_hand_pos
    data.mocap_quat[right_target_mocap_id] = right_hand_quat
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            cycle_end_time = time.time() + model.opt.timestep
            
            qpos = np.ndarray(Q_SHARED_SHAPE, dtype=DTYPE, buffer=q_shm.buf)
            data.qpos[:] = qpos.copy() # type: ignore
            mujoco.mj_step(model, data)

            left_target_pos_shm.buf[:] = data.mocap_pos[left_target_mocap_id].astype(DTYPE).tobytes() # type: ignore
            left_target_quat_shm.buf[:] = data.mocap_quat[left_target_mocap_id].astype(DTYPE).tobytes() # type: ignore
            right_target_pos_shm.buf[:] = data.mocap_pos[right_target_mocap_id].astype(DTYPE).tobytes() # type: ignore
            right_target_quat_shm.buf[:] = data.mocap_quat[right_target_mocap_id].astype(DTYPE).tobytes() # type: ignore
            viewer.sync()
            
            now = time.time()
            sleep_time = cycle_end_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)

def mid_level_control_loop(x_shm_name: str, u_shm_name: str, left_target_pos_shm_name: str, left_target_quat_shm_name: str, right_target_pos_shm_name: str, right_target_quat_shm_name: str):
    x_shm = shared_memory.SharedMemory(name=x_shm_name)
    u_shm = shared_memory.SharedMemory(name=u_shm_name)
    
    left_target_pos_shm = shared_memory.SharedMemory(name=left_target_pos_shm_name)
    left_target_quat_shm = shared_memory.SharedMemory(name=left_target_quat_shm_name)
    right_target_pos_shm = shared_memory.SharedMemory(name=right_target_pos_shm_name)
    right_target_quat_shm = shared_memory.SharedMemory(name=right_target_quat_shm_name)

    mpc = OpenArmMPCSolver()
    
    U_init = np.zeros((mpc.N, mpc.nu), dtype=np.float64)
    max_iter = 10

    solve_every = 3
    step_count = 0
    u_cmd = np.zeros((mpc.nu,), dtype=np.float64)
    
    while True:
        step_start = time.time()

        if step_count % solve_every == 0:
            left_pos = np.ndarray(LEFT_TARGET_POS_SHARED_SHAPE, dtype=DTYPE, buffer=left_target_pos_shm.buf)
            left_quat = np.ndarray(LEFT_TARGET_QUAT_SHARED_SHAPE, dtype=DTYPE, buffer=left_target_quat_shm.buf)
            right_pos = np.ndarray(RIGHT_TARGET_POS_SHARED_SHAPE, dtype=DTYPE, buffer=right_target_pos_shm.buf)
            right_quat = np.ndarray(RIGHT_TARGET_QUAT_SHARED_SHAPE, dtype=DTYPE, buffer=right_target_quat_shm.buf)

            target = np.concatenate([
                left_pos, left_quat,
                right_pos, right_quat
            ]).astype(np.float64)

            x = np.ndarray(X_SHARED_SHAPE, dtype=DTYPE, buffer=x_shm.buf)
            print(x)

            _, U_sol, success = mpc.solve_slq(
                x0=x,
                target=target,
                U_init=U_init,
                max_iter=max_iter,
                shift_warm_start=True,
            )

            # if not success:
            #     U_init[:] = 0.0
            #     u_cmd[:] = 0.0
            # else:
            U_init = U_sol.copy()
            u_cmd = U_sol[0].copy()
            
            step_end = time.time()
            dt_solve = step_end - step_start
            # print(f'SLQ step spent {dt_solve:.6f}s ({1.0 / max(dt_solve, 1e-9):.2f} Hz)')

        u_shm.buf[:] = u_cmd.astype(DTYPE).tobytes() # type: ignore

        step_count += 1
    
def low_level_control_loop(q_shm_name: str, v_shm_name: str, u_shm_name: str, mit_kp_shm_name: str, mit_kd_shm_name: str, mit_pos_des_shm_name: str, mit_vel_des_shm_name: str, mit_tau_des_shm_name: str):
    q_shm = shared_memory.SharedMemory(name=q_shm_name)
    v_shm = shared_memory.SharedMemory(name=v_shm_name)
    u_shm = shared_memory.SharedMemory(name=u_shm_name)
    
    mit_kp_shm = shared_memory.SharedMemory(name=mit_kp_shm_name)
    mit_kd_shm = shared_memory.SharedMemory(name=mit_kd_shm_name)
    
    mit_pos_des_shm = shared_memory.SharedMemory(name=mit_pos_des_shm_name)
    mit_vel_des_shm = shared_memory.SharedMemory(name=mit_vel_des_shm_name)
    mit_tau_des_shm = shared_memory.SharedMemory(name=mit_tau_des_shm_name)

    llc = LowLevelController()

    while True:
        qpos = np.ndarray(Q_SHARED_SHAPE, dtype=DTYPE, buffer=q_shm.buf)
        qvel = np.ndarray(V_SHARED_SHAPE, dtype=DTYPE, buffer=v_shm.buf)
        u = np.ndarray(U_SHARED_SHAPE, dtype=DTYPE, buffer=u_shm.buf)
        cmd = llc.update(qpos, qvel, u)
        mit_kp_shm.buf[:] = cmd.kp.astype(DTYPE).tobytes() # type: ignore
        mit_kd_shm.buf[:] = cmd.kd.astype(DTYPE).tobytes() # type: ignore
        mit_pos_des_shm.buf[:] = cmd.pos_des.astype(DTYPE).tobytes() # type: ignore
        mit_vel_des_shm.buf[:] = cmd.vel_des.astype(DTYPE).tobytes() # type: ignore
        mit_tau_des_shm.buf[:] = cmd.tau_ff.astype(DTYPE).tobytes() # type: ignore
        time.sleep(0.01)

def control_loop(x_shm_name: str, q_shm_name: str, v_shm_name: str, mit_kp_shm_name: str, mit_kd_shm_name: str, mit_pos_des_shm_name: str, mit_vel_des_shm_name: str, mit_tau_des_shm_name: str):
    x_shm = shared_memory.SharedMemory(name=x_shm_name)
    q_shm = shared_memory.SharedMemory(name=q_shm_name)
    v_shm = shared_memory.SharedMemory(name=v_shm_name)
    
    mit_kp_shm = shared_memory.SharedMemory(name=mit_kp_shm_name)
    mit_kd_shm = shared_memory.SharedMemory(name=mit_kd_shm_name)
    
    mit_pos_des_shm = shared_memory.SharedMemory(name=mit_pos_des_shm_name)
    mit_vel_des_shm = shared_memory.SharedMemory(name=mit_vel_des_shm_name)
    mit_tau_des_shm = shared_memory.SharedMemory(name=mit_tau_des_shm_name)
    
    right_arm = oa.OpenArm("can0", True)
    left_arm  = oa.OpenArm("can1", True)
    
    motor_types = [
        oa.MotorType.DM8009, oa.MotorType.DM8009,
        oa.MotorType.DM4340, oa.MotorType.DM4340,
        oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310
    ]
    send_ids = [ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 ]
    recv_ids = [ 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17 ]
    
    right_arm.init_arm_motors(motor_types, send_ids, recv_ids)
    left_arm.init_arm_motors(motor_types, send_ids, recv_ids)
    
    right_arm.enable_all()
    left_arm.enable_all()
    time.sleep(5 * 1e-3)
    right_arm.recv_all()
    left_arm.recv_all()
    
    right_arm.set_callback_mode_all(oa.CallbackMode.STATE)
    left_arm.set_callback_mode_all(oa.CallbackMode.STATE)
    time.sleep(5 * 1e-3)
    right_arm.recv_all()
    left_arm.recv_all()
    
    right_arm.get_arm().set_control_mode_all(oa.ControlMode.MIT)
    left_arm.get_arm().set_control_mode_all(oa.ControlMode.MIT)
    time.sleep(5 * 1e-3)
    right_arm.recv_all()
    left_arm.recv_all()
    
    RIGHT_KPS = [300.0, 240.0, 150.0, 150.0, 30.0, 50.0, 30.0]
    RIGHT_KDS = [5.0, 2.5, 1.2, 1.2, 0.5, 0.3, 0.5]
    
    LEFT_KPS = [300.0, 240.0, 150.0, 150.0, 30.0, 50.0, 30.0]
    LEFT_KDS = [5.0, 2.5, 1.2, 1.2, 0.5, 0.3, 0.5]
    
    left_reversed_ids = [0, 1, 3, 5, 6]
    right_reversed_ids = [1, 3, 5]
    
    start_time = time.time()

    while True:
        mit_kp = np.ndarray(MIT_KP_SHAPE, dtype=DTYPE, buffer=mit_kp_shm.buf)
        mit_kd = np.ndarray(MIT_KD_SHAPE, dtype=DTYPE, buffer=mit_kd_shm.buf)
        mit_pos_des = np.ndarray(MIT_POS_DES_SHAPE, dtype=DTYPE, buffer=mit_pos_des_shm.buf)
        mit_vel_des = np.ndarray(MIT_VEL_DES_SHAPE, dtype=DTYPE, buffer=mit_vel_des_shm.buf)
        mit_tau_des = np.ndarray(MIT_TAU_DES_SHAPE, dtype=DTYPE, buffer=mit_tau_des_shm.buf)
        
        q = np.ndarray(Q_SHARED_SHAPE, dtype=DTYPE, buffer=q_shm.buf)
        v = np.ndarray(V_SHARED_SHAPE, dtype=DTYPE, buffer=v_shm.buf)
        
        mit_pos_des = np.zeros(MIT_POS_DES_SHAPE, dtype=DTYPE)
        mit_pos_des[8:8+7] = q[7+8:7+8+7] + mit_vel_des[8:8+7] * 0.05
        mit_pos_des[8+9:8+9+7] = q[7+8+9:7+8+9+7] + mit_vel_des[8+9:8+9+7] * 0.05
        
        for idx in left_reversed_ids:
            mit_pos_des[8+idx] *= -1.0
            mit_vel_des[8+idx] *= -1.0
            mit_tau_des[8+idx] *= -1.0
        
        for idx in right_reversed_ids:
            mit_pos_des[8+9+idx] *= -1.0
            mit_vel_des[8+9+idx] *= -1.0
            mit_tau_des[8+9+idx] *= -1.0

        left_cmds = [
            MITParam(q=float(mit_pos_des[8+i]), dq=float(mit_vel_des[8+i]), kp=float(LEFT_KPS[i]), kd=float(LEFT_KDS[i]), tau=float(mit_tau_des[8+i]) * 0.5)
            for i in range(7)
        ]
        if time.time() - start_time > 3.0:
            left_arm.get_arm().mit_control_all(left_cmds)
        
        right_cmds = [
            MITParam(q=float(mit_pos_des[8+9+i]), dq=float(mit_vel_des[8+9+i]), kp=float(RIGHT_KPS[i]), kd=float(RIGHT_KDS[i]), tau=float(mit_tau_des[8+9+i]) * 0.5)
            for i in range(7)
        ]
        if time.time() - start_time > 3.0:
            right_arm.get_arm().mit_control_all(right_cmds)
        
        time.sleep(10 * 1e-3)
        right_arm.refresh_all()
        left_arm.refresh_all()
        time.sleep(10 * 1e-3)
        right_arm.recv_all()
        left_arm.recv_all()
        
        q = np.zeros(Q_SHARED_SHAPE, dtype=DTYPE)
        v = np.zeros(V_SHARED_SHAPE, dtype=DTYPE)
        
        q[2] = 0.2
        q[3] = 1.0
        
        motors = left_arm.get_arm().get_motors()
        for i in range(7):
            q[7+8+i] = motors[i].get_position()
            v[6+8+i] = motors[i].get_velocity()
            if i in left_reversed_ids:
                q[7+8+i] *= -1.0
                v[6+8+i] *= -1.0
        
        motors = right_arm.get_arm().get_motors()
        for i in range(7):
            q[7+8+9+i] = motors[i].get_position()
            v[6+8+9+i] = motors[i].get_velocity()
            if i in right_reversed_ids:
                q[7+8+9+i] *= -1.0
                v[6+8+9+i] *= -1.0
        
        x_shm.buf[:] = q.copy().tobytes() # type: ignore
        q_shm.buf[:] = q.copy().tobytes() # type: ignore
        v_shm.buf[:] = v.copy().tobytes() # type: ignore
    

def main():
    x_shm = shared_memory.SharedMemory(create=True, size=X_SHARED_SHAPE[0] * DTYPE().nbytes)
    q_shm = shared_memory.SharedMemory(create=True, size=Q_SHARED_SHAPE[0] * DTYPE().nbytes)
    v_shm = shared_memory.SharedMemory(create=True, size=V_SHARED_SHAPE[0] * DTYPE().nbytes)
    u_shm = shared_memory.SharedMemory(create=True, size=U_SHARED_SHAPE[0] * DTYPE().nbytes)

    left_pos_target_shm = shared_memory.SharedMemory(create=True, size=LEFT_TARGET_POS_SHARED_SHAPE[0] * DTYPE().nbytes)
    left_quat_target_shm = shared_memory.SharedMemory(create=True, size=LEFT_TARGET_QUAT_SHARED_SHAPE[0] * DTYPE().nbytes)
    right_pos_target_shm = shared_memory.SharedMemory(create=True, size=RIGHT_TARGET_POS_SHARED_SHAPE[0] * DTYPE().nbytes)
    right_quat_target_shm = shared_memory.SharedMemory(create=True, size=RIGHT_TARGET_QUAT_SHARED_SHAPE[0] * DTYPE().nbytes)
    
    mit_kp_shm = shared_memory.SharedMemory(create=True, size=MIT_KP_SHAPE[0] * DTYPE().nbytes)
    mit_kd_shm = shared_memory.SharedMemory(create=True, size=MIT_KD_SHAPE[0] * DTYPE().nbytes) 
    
    mit_pos_des_shm = shared_memory.SharedMemory(create=True, size=MIT_POS_DES_SHAPE[0] * DTYPE().nbytes)
    mit_vel_des_shm = shared_memory.SharedMemory(create=True, size=MIT_VEL_DES_SHAPE[0] * DTYPE().nbytes)
    mit_tau_des_shm = shared_memory.SharedMemory(create=True, size=MIT_TAU_DES_SHAPE[0] * DTYPE().nbytes)
    
    process1 = Process(target=simulation_loop, args=(q_shm.name, left_pos_target_shm.name, left_quat_target_shm.name, right_pos_target_shm.name, right_quat_target_shm.name))
    process1.start()
    process2 = Process(target=plot_shared_memory_x_target_realtime, args=(x_shm.name, u_shm.name, left_pos_target_shm.name, left_quat_target_shm.name, right_pos_target_shm.name, right_quat_target_shm.name, mit_tau_des_shm.name))
    process2.start()
    process3 = Process(target=mid_level_control_loop, args=(x_shm.name, u_shm.name, left_pos_target_shm.name, left_quat_target_shm.name, right_pos_target_shm.name, right_quat_target_shm.name))
    process3.start()
    process4 = Process(target=low_level_control_loop, args=(q_shm.name, v_shm.name, u_shm.name, mit_kp_shm.name, mit_kd_shm.name, mit_pos_des_shm.name, mit_vel_des_shm.name, mit_tau_des_shm.name))
    process4.start()
    process5 = Process(target=control_loop, args=(x_shm.name, q_shm.name, v_shm.name, mit_kp_shm.name, mit_kd_shm.name, mit_pos_des_shm.name, mit_vel_des_shm.name, mit_tau_des_shm.name))
    process5.start()
    
    def onexit():
        process1.terminate()
        os.kill(process2.pid, signal.SIGINT)
        process2.join()
        process3.terminate()
        process4.terminate()
        process5.terminate()
        x_shm.close()
        q_shm.close()
        v_shm.close()
        u_shm.close()
        left_pos_target_shm.close()
        left_quat_target_shm.close()
        right_pos_target_shm.close()
        right_quat_target_shm.close()
        mit_pos_des_shm.close()
        mit_vel_des_shm.close()
        mit_tau_des_shm.close()

    atexit.register(onexit)

    try:
        while True:
            time.sleep(10 ** 6)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()