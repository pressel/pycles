#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport ParallelMPI
cimport ReferenceState

cdef class PostProcessing:
    cdef:
        str out_dir
        str fields_dir
        str stats_file
        list gridsize
        list gridspacing
        bint collapse_y
        bint half_x
        bint only_T_anomaly

    cpdef initialize(self, namelist)
    cpdef combine3d(self, ParallelMPI.ParallelMPI Pa, ReferenceState.ReferenceState Ref)
    cpdef to_3d(self, double[:] f_data, int nl_0, int nl_1, int nl_2, int indx_lo_0,
                int indx_lo_1, int indx_lo_2, double[:, :, :] f_data_3d)
    cpdef save_timestep(self, fname, variables, time, ReferenceState.ReferenceState Ref)
