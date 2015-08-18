cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables

cdef class Forcing:
    cdef:
        object scheme
    cpdef initialize(self)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

cdef class ForcingNone:
    cpdef initialize(self)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

cdef class ForcingSullivanPatton:
    cdef:
        double ug
        double vg
        double coriolis_param

    cpdef initialize(self)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)