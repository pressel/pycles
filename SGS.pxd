cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ReferenceState
cimport Kinematics
from libc.math cimport  fmax

cdef class SGS:
    cdef:
        object scheme

    cpdef initialize(self, Grid.Grid Gr)


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV,Kinematics.Kinematics Ke)


cdef class UniformViscosity:

    cpdef initialize(self, Grid.Grid Gr)


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke)

cdef class Smagorinsky:
    cdef:
        double cs
        double prt

    cpdef initialize(self, Grid.Grid Gr)


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke)

