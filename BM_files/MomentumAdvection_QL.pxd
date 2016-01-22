cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from NetCDFIO cimport NetCDFIO_Stats

cdef class MomentumAdvection:
    cdef:
        Py_ssize_t order

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef double[:, :, :] get_flux(self, Py_ssize_t i_advected, Py_ssize_t i_advecting, Grid.Grid Gr)
