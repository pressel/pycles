cimport ParallelMPI
cimport Grid
cimport ReferenceState


cdef class PressureSparseSciPy:

    cdef:
        object A

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)

    cpdef solve(self,Grid.Grid Gr, ReferenceState.ReferenceState RS, double [:] divergence, ParallelMPI.ParallelMPI PM)