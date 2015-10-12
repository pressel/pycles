cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from NetCDFIO cimport NetCDFIO_Stats


cdef class ScalarAdvection:

    cdef:
        double [:] flux
        Py_ssize_t order
        Py_ssize_t mp

    cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs,PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)