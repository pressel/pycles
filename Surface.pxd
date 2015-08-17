cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables


cdef class Surface:
    cdef:
        object scheme

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV)

cdef class SurfaceNone:
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV)

cdef class SurfaceSullivanPatton:
    cdef:
        double theta_flux
        double shf
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV)


