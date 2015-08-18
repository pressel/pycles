cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables

cdef class Forcing:
    cdef:
        object scheme
    cpdef initialize(self)

    cpdef update(self)

cdef class ForcingNone:
    cpdef initialize(self)

    cpdef update(self)

cdef class ForcingSullivanPatton:
    cdef:
        double ug
        double vg
        double coriolis_param

    cpdef initialize(self)

    cpdef update(self)