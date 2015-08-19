cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
import numpy as np

cdef class Forcing:
    cdef:
        object scheme
    cpdef initialize(self, Grid.Grid Gr)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

cdef class ForcingNone:
    cpdef initialize(self, Grid.Grid Gr)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)

cdef class ForcingSullivanPatton:
    cdef:
        double [:] ug
        double [:] vg
        double coriolis_param

    cpdef initialize(self, Grid.Grid Gr)

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)