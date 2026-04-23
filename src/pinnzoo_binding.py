import os
import numpy as np
import numpy.typing as npt

from cffi import FFI
from functools import cached_property

class PinnZooModel:
    def __init__(self, lib_path: str, nx: int|None = None):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f'file `{lib_path}` not found!')
        
        self.lib_path = lib_path
        self.ffi = FFI()
        
        self.ffi.cdef("""
            extern const char* config_names[];
            extern const char* vel_names[];
            extern const char* torque_names[];
            extern const char* kinematics_bodies[];
            void kinematics_wrapper(double* x, double* locs);
            void kinematics_jacobian_wrapper(double* x, double* J);
            void forward_dynamics_wrapper(double* x_in, double* tau_in, double* vdot_out);
            void forward_dynamics_deriv_wrapper(double* x_in, double* tau_in, double* dvdot_dx_out, double* dvdout_dtau_out);
            void dynamics_deriv_wrapper(double* x_in, double* tau_in, double* dxdot_dx_out, double* dxdout_dtau_out);
        """)
        
        self.lib = self.ffi.dlopen(os.path.abspath(self.lib_path))
        
        self.nq = self._get_c_array_len(self.lib.config_names) # type: ignore
        self.nv = self._get_c_array_len(self.lib.vel_names) # type: ignore
        self.nx = (self.nq + self.nv) if nx is None else nx
        self.nu = self.nv
        
        self.bodies_count = self._get_c_array_len(self.lib.kinematics_bodies) # type: ignore

    @cached_property
    def kinematics_size(self):
        if 'quat' in self.lib_path:
            return 7 * self.bodies_count
        else:
            return 3 * self.bodies_count
    
    def _get_c_array_len(self, ptr):
        count = 0
        while ptr[count] != self.ffi.NULL:
            count += 1
        return count