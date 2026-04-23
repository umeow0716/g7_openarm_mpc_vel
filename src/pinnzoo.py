import typing
import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from .pinnzoo_binding import PinnZooModel

def kinematics(model: 'PinnZooModel', x: npt.NDArray[np.float64]):
    locs = np.empty(model.kinematics_size, dtype=np.float64)
    
    p_x = model.ffi.cast("double*", x.ctypes.data)
    p_locs = model.ffi.cast("double*", locs.ctypes.data)
    
    model.lib.kinematics_wrapper(p_x, p_locs) # type: ignore
    
    return locs

def kinematics_jacobian(model: 'PinnZooModel', x: npt.NDArray[np.float64]):
    buf = np.zeros(model.kinematics_size * model.nx, dtype=np.float64)

    p_x = model.ffi.cast("double*", x.ctypes.data)
    p_buf = model.ffi.cast("double*", buf.ctypes.data)

    model.lib.kinematics_jacobian_wrapper(p_x, p_buf)  # type: ignore

    J = buf.reshape((model.kinematics_size, model.nx), order="F")
    return J

def forward_dynamics(model: 'PinnZooModel', x: npt.NDArray[np.float64], tau: npt.NDArray[np.float64]):
    vdot = np.empty(model.nv, dtype=np.float64)
    
    p_x    = model.ffi.cast("double*", x.ctypes.data)
    p_tau  = model.ffi.cast("double*", tau.ctypes.data)
    p_vdot = model.ffi.cast("double*", vdot.ctypes.data)
    
    model.lib.forward_dynamics_wrapper(p_x, p_tau, p_vdot) # type: ignore
    
    return vdot

def forward_dynamics_deriv(
    model: 'PinnZooModel',
    x: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
):
    dvdot_dx = np.empty((model.nv, model.nx), dtype=np.float64)
    dvdot_dtau = np.empty((model.nv, model.nv), dtype=np.float64)

    p_x = model.ffi.cast("double*", x.ctypes.data)
    p_tau = model.ffi.cast("double*", tau.ctypes.data)
    p_dvdot_dx = model.ffi.cast("double*", dvdot_dx.ctypes.data)
    p_dvdot_dtau = model.ffi.cast("double*", dvdot_dtau.ctypes.data)

    model.lib.forward_dynamics_deriv_wrapper(  # type: ignore
        p_x,
        p_tau,
        p_dvdot_dx,
        p_dvdot_dtau,
    )

    return dvdot_dx, dvdot_dtau

def dynamics_deriv(
    model: 'PinnZooModel',
    x: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
):
    dxdot_dx = np.empty((model.nx, model.nx), dtype=np.float64)
    dxdot_dtau = np.empty((model.nx, model.nv), dtype=np.float64)

    p_x = model.ffi.cast("double*", x.ctypes.data)
    p_tau = model.ffi.cast("double*", tau.ctypes.data)
    p_dxdot_dx = model.ffi.cast("double*", dxdot_dx.ctypes.data)
    p_dxdot_dtau = model.ffi.cast("double*", dxdot_dtau.ctypes.data)

    model.lib.dynamics_deriv_wrapper(  # type: ignore
        p_x,
        p_tau,
        p_dxdot_dx,
        p_dxdot_dtau,
    )

    return dxdot_dx, dxdot_dtau
    
def zero_state(model: 'PinnZooModel'):
    return np.zeros(model.nx, dtype=np.float64)