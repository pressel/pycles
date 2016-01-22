cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
cimport DiagnosticVariables
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from Thermodynamics cimport LatentHeat

cdef class ScalarAdvection:

    cdef:
        double [:] flux
        Py_ssize_t order
        Py_ssize_t order_sedimentation
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil

    cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs,PrognosticVariables.PrognosticVariables PV,  DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)