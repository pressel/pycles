cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState

cdef class MomentumAdvection:
    cdef:
        double [:] flux
        long order

    cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update(self, Grid.Grid Gr,ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa)
    cpdef double [:,:,:] get_flux(self, int i_advected, int i_advecting, Grid.Grid Gr)