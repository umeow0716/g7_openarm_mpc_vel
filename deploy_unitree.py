import time
import atexit
from multiprocessing import Process

from src.nodes import (
    control_node,
    imu_node,
    low_controller_node,
    mid_controller_node,
    simulation_node,
    sport_node,
)


def start_process(target, *args):
    process = Process(target=target, args=args)
    process.start()
    return process


def terminate_process(process: Process):
    if process.is_alive():
        process.terminate()
        process.join(timeout=3.0)

    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def main():
    processes: list[Process] = []
    
    processes.append(start_process(control_node.main))
    processes.append(start_process(imu_node.main))
    processes.append(start_process(sport_node.main))
    processes.append(start_process(simulation_node.main, "real"))
    processes.append(start_process(mid_controller_node.main))
    processes.append(start_process(low_controller_node.main))

    def atexit_func():
        for process in reversed(processes):
            terminate_process(process)

    atexit.register(atexit_func)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        atexit_func()


if __name__ == "__main__":
    main()