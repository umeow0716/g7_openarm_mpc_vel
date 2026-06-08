import time
import atexit

from multiprocessing import Process
from src.nodes import low_controller_node, mid_controller_node, simulation_node

def main():
    process1 = Process(target=low_controller_node.main)
    process1.start()
    
    process2 = Process(target=mid_controller_node.main)
    process2.start()
    
    process3 = Process(target=simulation_node.main, args=('sim',))
    process3.start()
    
    def atexit_func():
        process1.terminate()
        process2.terminate()
        process3.terminate()
    
    atexit.register(atexit_func)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()