cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from NetCDFIO cimport NetCDFIO_Stats

cdef class StochasticNoise:
    cdef:
        bint stoch_noise

    cpdef initialize(self,namelist)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV, Th, ParallelMPI.ParallelMPI Pa)
    cpdef add_theta_noise(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV, Th, ParallelMPI.ParallelMPI Pa)
