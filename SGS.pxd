cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ReferenceState

cdef class SGS:
    cdef:
        object scheme

    cpdef initialize(self, Grid.Grid Gr)


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV)


cdef class UniformViscosity:

    cpdef initialize(self, Grid.Grid Gr)


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV)

