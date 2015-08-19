cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
import numpy as np

cdef class Forcing:
    cdef:
        object scheme
    cpdef initialize(self, Grid.Grid Gr)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)

cdef class ForcingNone:
    cpdef initialize(self, Grid.Grid Gr)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)

cdef class ForcingBomex:
    cdef:
        double [:] ug
        double [:] vg
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double coriolis_param
    cpdef initialize(self, Grid.Grid Gr)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)

cdef class ForcingSullivanPatton:
    cdef:
        double [:] ug
        double [:] vg
        double coriolis_param

    cpdef initialize(self, Grid.Grid Gr)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)