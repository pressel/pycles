cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
from Thermodynamics cimport LatentHeat
import cython
from thermodynamic_functions import exner
from libc.math cimport sqrt, log, fabs,atan, exp, fmax
cimport numpy as np
import numpy as np
include "parameters.pxi"

import cython

cdef extern from "advection_interpolation.h":
    double interp_2(double phi, double phip1) nogil


cdef extern from "thermodynamic_functions.h":
    inline double pd_c(double p0, double qt, double qv) nogil
    inline double pv_c(double p0, double qt, double qv) nogil
    inline double exner_c(const double p0) nogil





cdef extern from "surface.h":
    inline double compute_ustar_c(double windspeed, double buoyancy_flux, double z0, double z1) nogil
    inline double entropyflux_from_thetaflux_qtflux(double thetaflux, double qtflux, double p0_b, double T_b, double qt_b, double qv_b) nogil


cdef class Surface:
    def __init__(self,namelist, LatentHeat LH):
        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            self.scheme = SurfaceSullivanPatton()
        elif casename == 'Bomex':
            self.scheme = SurfaceBomex()
        else:
            self.scheme= SurfaceNone()

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        self.scheme.initialize(Gr, RS)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        self.scheme.update(Gr, RS, PV, DV, Pa)
        return


cdef class SurfaceNone:
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        return


cdef class SurfaceSullivanPatton:
    def __init__(self):
        self.theta_flux = 0.24 # K m/s
        self.z0 = 0.1 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):
        T0 = RS.p0_half[Gr.dims.gw] * RS.alpha0_half[Gr.dims.gw]/Rd
        self.buoyancy_flux = self.theta_flux * exner(RS.p0[Gr.dims.gw-1]) * g /T0
        self.ustar = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.u_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.v_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple

        if Pa.sub_z_rank != 0:
            return

        cdef:
            long i
            long j
            long gw = Gr.dims.gw
            long ijk, ij
            long imax = Gr.dims.nlg[0]
            long jmax = Gr.dims.nlg[1]
            long istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            long jstride = Gr.dims.nlg[2]
            long istride_2d = Gr.dims.nlg[1]
            long temp_shift = DV.get_varshift(Gr, 'temperature')
            long s_shift = PV.get_varshift(Gr, 's')
            double dzi = 1.0/Gr.dims.dx[2]
            double entropy_flux

        #Get the scalar flux (dry entropy only)
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    entropy_flux = cpd * self.theta_flux*exner_c(RS.p0_half[gw])/DV.values[temp_shift+ijk]
                    PV.tendencies[s_shift + ijk] = PV.tendencies[s_shift + ijk] + entropy_flux*RS.alpha0_half[gw]/RS.alpha0[gw-1]*dzi

        cdef:
            long u_shift = PV.get_varshift(Gr,'u')
            long v_shift = PV.get_varshift(Gr, 'v')

        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax):
                for j in xrange(1,jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.windspeed[ij] = fmax(sqrt((interp_2(PV.values[u_shift+ijk-istride],PV.values[u_shift+ijk])+RS.u0)**2
                                                    + (interp_2(PV.values[v_shift+ijk-jstride],PV.values[v_shift+ijk]) + RS.v0)**2), self.gustiness)
                    self.ustar[ij] = compute_ustar_c(self.windspeed[ij],self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(self.ustar[ij], self.ustar[ij+istride_2d])**2/interp_2(self.windspeed[ij], self.windspeed[ij+istride_2d]) * PV.values[u_shift + ijk]
                    self.v_flux[ij] = -interp_2(self.ustar[ij], self.ustar[ij+1])**2/interp_2(self.windspeed[ij], self.windspeed[ij+1]) * PV.values[v_shift + ijk]
                    PV.tendencies[u_shift + ijk] = PV.tendencies[u_shift + ijk] + self.u_flux[ij]/RS.alpha0[gw-1]*RS.alpha0_half[gw]*dzi
                    PV.tendencies[v_shift + ijk] = PV.tendencies[v_shift + ijk] + self.v_flux[ij]/RS.alpha0[gw-1]*RS.alpha0_half[gw]*dzi

        return

cdef class SurfaceBomex:
    def __init__(self):
        self.theta_flux = 8.0e-3 # K m/s
        self.qt_flux = 5.2e-5 # m/s
        self.ustar = 0.28 #m/s

        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):

        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.u_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.v_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            long i
            long j
            long gw = Gr.dims.gw
            long ijk, ij
            long imax = Gr.dims.nlg[0]
            long jmax = Gr.dims.nlg[1]
            long istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            long jstride = Gr.dims.nlg[2]
            long istride_2d = Gr.dims.nlg[1]
            long temp_shift = DV.get_varshift(Gr, 'temperature')
            long s_shift = PV.get_varshift(Gr, 's')
            long qt_shift = PV.get_varshift(Gr, 'qt')
            long qv_shift = DV.get_varshift(Gr,'qv')
            double dzi = 1.0/Gr.dims.dx[2]
            double entropy_flux

        # Get the scalar flux
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j

                    entropy_flux = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux, RS.p0_half[gw], DV.values[temp_shift+ijk], PV.values[qt_shift+ijk], DV.values[qv_shift+ijk])
                    PV.tendencies[s_shift + ijk] = PV.tendencies[s_shift + ijk] + entropy_flux*RS.alpha0_half[gw]/RS.alpha0[gw-1]*dzi
                    PV.tendencies[qt_shift + ijk] = PV.tendencies[qt_shift + ijk] + self.qt_flux*RS.alpha0_half[gw]/RS.alpha0[gw-1]*dzi

        cdef:
            long u_shift = PV.get_varshift(Gr,'u')
            long v_shift = PV.get_varshift(Gr, 'v')


        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax):
                for j in xrange(1,jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.windspeed[ij] = sqrt((interp_2(PV.values[u_shift+ijk-istride],PV.values[u_shift+ijk])+RS.u0)**2
                                              + (interp_2(PV.values[v_shift+ijk-jstride],PV.values[v_shift+ijk])+RS.v0)**2)

            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -self.ustar**2/interp_2(self.windspeed[ij], self.windspeed[ij+istride_2d]) * PV.values[u_shift + ijk]
                    self.v_flux[ij] = -self.ustar**2/interp_2(self.windspeed[ij], self.windspeed[ij+1]) * PV.values[v_shift + ijk]
                    PV.tendencies[u_shift + ijk] = PV.tendencies[u_shift + ijk] + self.u_flux[ij]/RS.alpha0[gw-1]*RS.alpha0_half[gw]*dzi
                    PV.tendencies[v_shift + ijk] = PV.tendencies[v_shift + ijk] + self.v_flux[ij]/RS.alpha0[gw-1]*RS.alpha0_half[gw]*dzi

        return




# Anderson, R. J., 1993: A Study of Wind Stress and Heat Flux over the Open
# Ocean by the Inertial-Dissipation Method. J. Phys. Oceanogr., 23, 2153--“2161.
# See also: ARPS documentation
cdef inline double compute_z0(double z1, double windspeed) nogil:
    cdef double z0 =z1*exp(-kappa/sqrt((0.4 + 0.079*windspeed)*1e-3))
    return z0

@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
cdef inline double compute_ustar(double windspeed, double buoyancy_flux, double z0, double z1) nogil:
    cdef:
        double lnz = log(z1/fabs(z0))
        double ustar = windspeed * kappa/lnz
        int i
        double lmo
        double zeta
        double x
        double psi1
        double am = 4.8
        double bm = 19.3
        double c1 = -0.50864521488493919 # = pi/2 - 3*log(2)

    if fabs(buoyancy_flux) > 1.0e-10:
        for i in xrange(6):
            lmo = -(ustar * ustar * ustar)/(buoyancy_flux * kappa)
            zeta = z1/lmo
            if zeta > 0.0:
                ustar = kappa*windspeed/(lnz + am*zeta)
            else:
                x = sqrt(sqrt(1.0 - bm * zeta))
                psi1 = 2.0 * log(1.0 + x) + log(1.0 + x*x) - 2.0 * atan(x) + c1
                ustar = windspeed * kappa/(lnz-psi1)

    return  ustar

