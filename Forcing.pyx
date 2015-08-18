cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables

cdef class Forcing:
    def __init__(self, namelist):
        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            self.scheme = ForcingSullivanPatton()
        else:
            self.scheme= ForcingNone()

    cpdef initialize(self):
        self.scheme.initialize()

    cpdef update(self):
        self.scheme.update()


cdef class ForcingNone:
    def __init__(self):
        pass
    cpdef initialize(self):
        return

    cpdef update(self):
        return


cdef class ForcingSullivanPatton:
    def __init__(self):
        self.ug = 1.0 #m/s
        self.vg = 0.0 #m/s
        self.coriolis_param = 1.0e-4 #s^{-1}
        return
    cpdef initialize(self):
        return
    cpdef update(self):
        return

