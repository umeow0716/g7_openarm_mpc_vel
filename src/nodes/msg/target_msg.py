import cyclonedds.idl as idl

from dataclasses import dataclass
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Pose_ as Pose_default

@dataclass
class TargetMsg(idl.IdlStruct, typename="TargetMsg"):
    left: Pose_
    right: Pose_

def TargetMsg_default():
    return TargetMsg(Pose_default(), Pose_default())