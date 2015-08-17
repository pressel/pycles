cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
include "parameters.pxi"



cdef extern from "thermodynamic_functions.h":
    inline double pd_c(double p0, double qt, double qv) nogil
    inline double pv_c(double p0, double qt, double qv) nogil

cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil
    inline double sc_c(double L, double T) nogil



cdef class Surface:
    def __init__(self,namelist, LatentHeat LH):
        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            self.scheme = SurfaceSullivanPatton(LH)
        else:
            self.scheme= SurfaceNone()

    cpdef initialize(self, Grid.Grid Gr):
        self.scheme.initialize(Gr)
        return

    cpdef update(self):
        self.scheme.update()
        return


cdef class SurfaceNone:
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr):
        pass

    cpdef update():
        pass


cdef class SurfaceSullivanPatton():
    def __init__(self):
        self.theta_flux = 0.24 # K m/s
        self.L_fp = LH.L_fp
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        self.shf = self.theta_flux * exner(RS.p0[Gr.dims.gw-1]) * cpd / RS.alpha0[Gr.dims.gw-1]
        self.lhf = 0.0
        return

    # @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr DiagnosticVariables.DiagnosticVariables DV):
        cdef:
            long i
            long j
            long k = Gr.dims.gw
            long imin = 0
            long jmin = 0
            long imax = Gr.dims.nlg[0]
            long jmax = Gr.dims.nlg[1]
            long temp_shift = DV.get_varshift(Gr, 'temperature')


        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):








        return
