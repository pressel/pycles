cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from NetCDFIO cimport NetCDFIO_Stats

cdef class Kinematics:
    cdef:
        float[:] vgrad
        float[:] strain_rate
        float[:] strain_rate_mag
        float[:] wind_speed
        float[:] wind_angle
        Py_ssize_t get_grad_shift(self, Grid.Grid Gr, Py_ssize_t vel_i, Py_ssize_t dx_j)
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef stats_io(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
