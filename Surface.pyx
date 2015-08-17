cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from Thermodynamics cimport LatentHeat
import cython
from thermodynamic_functions import exner
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
            self.scheme = SurfaceSullivanPatton()
        else:
            self.scheme= SurfaceNone()

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        self.scheme.initialize(Gr, RS)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        self.scheme.update(Gr, RS, PV, DV)
        return


cdef class SurfaceNone:
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        return


cdef class SurfaceSullivanPatton:
    def __init__(self):
        self.theta_flux = 0.24 # K m/s
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        # should theta_flux be adjusted to sensible heat flux using half or whole RS values?
        self.shf = self.theta_flux * exner(RS.p0[Gr.dims.gw-1]) * cpd / RS.alpha0[Gr.dims.gw-1]
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple
        cdef:
            long i
            long j
            long gw = Gr.dims.gw
            long ijk
            long imax = Gr.dims.nlg[0]
            long jmax = Gr.dims.nlg[1]
            long istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            long jstride = Gr.dims.nlg[2]
            long temp_shift = DV.get_varshift(Gr, 'temperature')
            long s_shift = PV.get_varshift(Gr, 's')
            double entropy_flux
            double shf = self.shf
            double lhf = self.lhf
            double alpha0_b = RS.alpha0_half[gw]
            double dzi = 1.0/Gr.dims.dx[2]


        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    entropy_flux = alpha0_b*shf/DV.values[temp_shift+ijk]
                    PV.tendencies[s_shift + ijk] = PV.tendencies[s_shift + ijk] + entropy_flux*alpha0_b/RS.alpha0[gw-1]*dzi

        return
