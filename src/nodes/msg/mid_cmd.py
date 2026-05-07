import cyclonedds.idl.types as types

from dataclasses import dataclass
import cyclonedds.idl as idl
import cyclonedds.idl.types as types

@dataclass
class MidCmd(idl.IdlStruct, typename="MidCmd"):
    u: types.array[types.float64, 21]

def MidCmd_default():
    return MidCmd([0.0] * 21) # type: ignore