cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState

cdef class ScalarAdvection:

    cdef:
        double [:] flux
        long order

    cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update_cython(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs,PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)