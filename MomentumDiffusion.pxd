cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Kinematics
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats

cdef class MomentumDiffusion:
    cdef:
        double[:] flux
        Py_ssize_t n_fluxes
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, Kinematics.Kinematics Ke)
    cpdef stats_io(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
