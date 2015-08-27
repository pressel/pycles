cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState

cdef class MomentumAdvection:
    cdef:
        double[:] flux
        Py_ssize_t order

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef double[:, :, :] get_flux(self, Py_ssize_t i_advected, Py_ssize_t i_advecting, Grid.Grid Gr)
