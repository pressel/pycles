cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables


cdef class SGS:
    def __init__(self):
        self.scheme = UniformViscosity()

    cpdef initialize(self, Grid.Grid Gr):
        self.scheme.initialize(Gr)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV):

        self.scheme.update(Gr,RS,DV,PV)

        return


cdef class UniformViscosity:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr):

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV):

        cdef:
            long diff_shift = DV.get_varshift(Gr,'diffusivity')
            long visc_shift = DV.get_varshift(Gr,'viscosity')
            long i


        with nogil:
            for i in xrange(Gr.dims.npg):
                DV.values[diff_shift + i] = 75.0
                DV.values[visc_shift + i] = 75.0

        return