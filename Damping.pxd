cimport ParallelMPI as ParallelMPI
cimport PrognosticVariables as PrognosticVariables
cimport Grid as Grid
cimport ReferenceState
cimport DiagnosticVariables

cdef class Damping:
    cdef:
        object scheme
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
cdef class Dummy:
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
cdef class Rayleigh:
    cdef:
        double z_d  # Depth of damping layer
        double gamma_r  # Inverse damping timescale
        double[:] gamma_zhalf
        double[:] gamma_z
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)

cdef class NudgeCGILS:
    cdef:
        double z_relax
        double z_relax_plus
        double tau_inverse
        double tau_vel_inverse
        double qt_floor
        Py_ssize_t floor_index
        double[:] gamma_zhalf
        double[:] gamma_z
        bint is_p2
        Py_ssize_t loc
        double [:] nudge_qt
        double [:] nudge_s
        double [:] nudge_u
        double [:] nudge_v

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)