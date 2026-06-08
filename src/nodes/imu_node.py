import time
import math
import xspublic
import threading
from collections import deque

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__IMUState_ as IMUState_default


def deg_to_rad(value: float) -> float:
    return float(value) * math.pi / 180.0

class CallbackHandler(xspublic.XsCallback):
    def __init__(self, max_buffer=5):
        super().__init__()
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=max_buffer)

    def packet_available(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0

    def get_next_packet(self) -> xspublic.XsDataPacket:
        with self._lock:
            return self._buffer.popleft()

    def onLiveDataAvailable(self, dev, packet):
        with self._lock:
            self._buffer.append(xspublic.XsDataPacket(packet))

class IMUNode:
    def __init__(self, callback_handler: CallbackHandler):
        self.imu_state_pub = ChannelPublisher("rt/imustate", IMUState_)
        self.imu_state_pub.Init()
        self.state = IMUState_default()
        self.callback_handler = callback_handler

    def control_loop(self):
        while True:
            if not self.callback_handler.packet_available():
                time.sleep(0.001)
                continue

            packet = self.callback_handler.get_next_packet()

            if packet.contains_calibrated_acc():
                acc = packet.calibrated_acc()
                self.state.accelerometer[0] = acc[0] # type: ignore
                self.state.accelerometer[1] = acc[1] # type: ignore
                self.state.accelerometer[2] = acc[2] # type: ignore

            if packet.contains_calibrated_gyr():
                gyr = packet.calibrated_gyr()
                self.state.gyroscope[0] = gyr[0] # type: ignore
                self.state.gyroscope[1] = gyr[1] # type: ignore
                self.state.gyroscope[2] = gyr[2] # type: ignore

            if packet.contains_orientation():
                quat = packet.orientation_quaternion()
                self.state.quaternion[0] = quat[0] # type: ignore
                self.state.quaternion[1] = quat[1] # type: ignore
                self.state.quaternion[2] = quat[2] # type: ignore
                self.state.quaternion[3] = quat[3] # type: ignore
                
                # Xsens orientation_euler() returns degrees.
                # Unitree IMUState.rpy is consumed by EKF as radians.
                rpy = packet.orientation_euler()
                self.state.rpy[0] = deg_to_rad(rpy.roll)   # type: ignore
                self.state.rpy[1] = deg_to_rad(rpy.pitch)  # type: ignore
                self.state.rpy[2] = deg_to_rad(rpy.yaw)    # type: ignore
                
            self.imu_state_pub.Write(self.state)    
    

def main():
    ChannelFactoryInitialize()
    
    control = xspublic.XsControl()
    ports = xspublic.XsScanner.scan_ports()

    mt_port = None
    for port in ports:
        if port.device_id().is_mti() or port.device_id().is_mtig():
            mt_port = port
            break

    if mt_port is None:
        control.close()
        raise RuntimeError("No MTi device found. Aborting.")

    if not control.open_port(mt_port.port_name(), mt_port.baud_rate()):
        control.close()
        raise RuntimeError("Could not open port. Aborting.")

    device = control.device(mt_port.device_id())
    print(f"Device: {device.product_code()}  ID: {device.device_id().to_string()} opened.")

    callback = CallbackHandler()
    device.add_callback_handler(callback)

    if not device.goto_config():
        control.close()
        raise RuntimeError("Could not put device into configuration mode. Aborting.")
    
    device.read_emts_and_device_configuration()
    device_id = device.device_id()

    config = [
        xspublic.XsOutputConfiguration(xspublic.PacketCounter,   0),
        xspublic.XsOutputConfiguration(xspublic.SampleTimeFine,  0),
        xspublic.XsOutputConfiguration(xspublic.StatusWord,      0),
        xspublic.XsOutputConfiguration(xspublic.Acceleration,      100),
        xspublic.XsOutputConfiguration(xspublic.FreeAcceleration,  100),
        xspublic.XsOutputConfiguration(xspublic.RateOfTurn,        100),
        xspublic.XsOutputConfiguration(xspublic.MagneticField,     100),
        xspublic.XsOutputConfiguration(xspublic.Quaternion,        100),
    ]
    
    assert device_id.is_imu()

    if not device.set_output_configuration(config):
        control.close()
        raise RuntimeError("Could not configure MTi device. Aborting.")

    if not device.goto_measurement():
        control.close()
        raise RuntimeError("Could not put device into measurement mode. Aborting.")
    
    node = IMUNode(callback)
    node.control_loop()

if __name__ == "__main__":
    main()
