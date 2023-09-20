#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

### IMPORTS = none?
include "parameters.pxi"

cdef class PostProcessing:
    cdef:
        str out_path
        str fields_path
        list gridsize
        list gridspacing

    cpdef initialize(self, namelist)
    cpdef combine3d(self)
    cpdef to_3d(self, double[:] f_data, int nl_0, int nl_1, int nl_2, int indx_lo_0,
                int indx_lo_1, int indx_lo_2, double[:, :, :] f_data_3d)
    cpdef save_timestep(self, fname, variables, time)
