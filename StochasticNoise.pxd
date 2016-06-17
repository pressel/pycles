cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from NetCDFIO cimport NetCDFIO_Stats

cdef class StochasticNoise:
    cdef:
        Py_ssize_t order

    cpdef initialize(self)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV, Th, ParallelMPI.ParallelMPI Pa)
