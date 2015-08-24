cimport ParallelMPI
cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables

cdef class PressureSolver:
    cdef:
        #Here we define the poisson solver to be an arbitrary python object (there is likely a better way to do this)
        object poisson_solver
        double [:] divergence
    cpdef initialize(self,namelist, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI PM)
    cpdef update(self,Grid.Grid Gr, ReferenceState.ReferenceState RS,   DiagnosticVariables.DiagnosticVariables DV, PrognosticVariables.PrognosticVariables PV,
                 ParallelMPI.ParallelMPI PM)