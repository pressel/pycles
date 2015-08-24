cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI


cdef class Surface:
    cdef:
        object scheme

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)

cdef class SurfaceNone:
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)

cdef class SurfaceSullivanPatton:
    cdef:
        double theta_flux
        double z0
        double gustiness
        double buoyancy_flux
        double [:] ustar
        double [:] windspeed
        double [:] u_flux
        double [:] v_flux
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)

cdef class SurfaceBomex:
    cdef:
        double theta_flux
        double qt_flux
        double ustar
        double [:] windspeed
        double [:] u_flux
        double [:] v_flux
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)

cdef inline double compute_z0(double z1, double windspeed) nogil

