#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
from Thermodynamics cimport LatentHeat,ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Stats
import cython
from thermodynamic_functions import exner, theta_rho
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
    inline double theta_rho_c(double p0, double T,double qt, double qv) nogil


cdef extern from "surface.h":
    inline double compute_ustar_c(double windspeed, double buoyancy_flux, double z0, double z1) nogil
    inline double entropyflux_from_thetaflux_qtflux(double thetaflux, double qtflux, double p0_b, double T_b, double qt_b, double qv_b) nogil
    void compute_windspeed(Grid.DimStruct *dims, double* u, double*  v, double*  speed, double u0, double v0, double gustiness ) nogil
    void exchange_coefficients_byun(double Ri, double zb, double z0, double* cm, double* ch, double* lmo) nogil



cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil


cdef class Surface:
    def __init__(self,namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            self.scheme = SurfaceSullivanPatton()
        elif casename == 'Bomex':
            self.scheme = SurfaceBomex()
        elif casename == 'Gabls':
            self.scheme = SurfaceGabls(namelist,LH, Par)
        elif casename == 'DYCOMS_RF01':
            self.scheme = SurfaceDYCOMS_RF01(namelist, LH)
        else:
            self.scheme= SurfaceNone()
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        DV.add_variables_2d('obukhov_length', 'm')
        DV.add_variables_2d('friction_velocity', 'm/s')
        self.scheme.initialize(Gr, Ref, NS, Pa)
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        self.scheme.update(Gr, Ref, PV, DV, Pa, TS)
        return
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.scheme.stats_io(Gr, NS, Pa)
        return


cdef class SurfaceNone:
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        return
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class SurfaceSullivanPatton:
    def __init__(self):
        self.theta_flux = 0.24 # K m/s
        self.z0 = 0.1 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        T0 = Ref.p0_half[Gr.dims.gw] * Ref.alpha0_half[Gr.dims.gw]/Rd
        self.buoyancy_flux = self.theta_flux * exner(Ref.p0[Gr.dims.gw-1]) * g /T0
        self.s_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.u_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.v_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')



        NS.add_ts('uw_surface_mean',Gr, Pa)
        NS.add_ts('vw_surface_mean',Gr, Pa)
        NS.add_ts('s_flux_surface_mean', Gr, Pa)


        return


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t i
            Py_ssize_t j
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t ijk, ij
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            double dzi = 1.0/Gr.dims.dx[2]
            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]*dzi


        #Get the scalar flux (dry entropy only)
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.s_flux[ij] = cpd * self.theta_flux*exner_c(Ref.p0_half[gw])/DV.values[temp_shift+ijk]
                    PV.tendencies[s_shift + ijk] = PV.tendencies[s_shift + ijk] + self.s_flux[ij] * tendency_factor

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr,'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        cdef :
            Py_ssize_t lmo_shift = DV.get_varshift_2d(Gr, 'obukhov_length')
            Py_ssize_t ustar_shift = DV.get_varshift_2d(Gr, 'friction_velocity')

        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax):
                for j in xrange(1,jmax):
                    ij = i * istride_2d + j
                    DV.values_2d[ustar_shift + ij] = compute_ustar_c(windspeed[ij],self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)
                    DV.values_2d[lmo_shift + ij] = -DV.values_2d[ustar_shift + ij]*DV.values_2d[ustar_shift + ij]*DV.values_2d[ustar_shift + ij]/self.buoyancy_flux/vkb
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(DV.values_2d[ustar_shift + ij], DV.values_2d[ustar_shift+ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(DV.values_2d[ustar_shift + ij], DV.values_2d[ustar_shift+ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
                    PV.tendencies[u_shift + ijk] = PV.tendencies[u_shift + ijk] + self.u_flux[ij] * tendency_factor
                    PV.tendencies[v_shift + ijk] = PV.tendencies[v_shift + ijk] + self.v_flux[ij] * tendency_factor

        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef double tmp

        tmp = Pa.HorizontalMeanSurface(Gr, &self.u_flux[0])
        NS.write_ts('uw_surface_mean',tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.v_flux[0])
        NS.write_ts('vw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.s_flux[0])
        NS.write_ts('s_flux_surface_mean', tmp, Pa)


        return


cdef class SurfaceBomex:
    def __init__(self):
        self.theta_flux = 8.0e-3 # K m/s
        self.qt_flux = 5.2e-5 # m/s
        self.ustar_ = 0.28 #m/s
        self.theta_surface = 299.1 #K
        self.qt_surface = 22.45e-3 # kg/kg
        self.buoyancy_flux = g * ((self.theta_flux + (eps_vi-1.0)*(self.theta_surface*self.qt_flux + self.qt_surface *self.theta_flux))
                              /(self.theta_surface*(1.0 + (eps_vi-1)*self.qt_surface)))


        pass

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):


        self.u_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.v_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.s_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')



        NS.add_ts('uw_surface_mean',Gr, Pa)
        NS.add_ts('vw_surface_mean',Gr, Pa)
        NS.add_ts('s_flux_surface_mean', Gr, Pa)


        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return


        cdef :
            Py_ssize_t lmo_shift = DV.get_varshift_2d(Gr, 'obukhov_length')
            Py_ssize_t ustar_shift = DV.get_varshift_2d(Gr, 'friction_velocity')



            Py_ssize_t i
            Py_ssize_t j
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t ijk, ij
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            double dzi = 1.0/Gr.dims.dx[2]
            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]*dzi


        # Get the scalar flux
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    DV.values_2d[ustar_shift + ij] = self.ustar_
                    DV.values_2d[lmo_shift + ij] = -self.ustar_*self.ustar_*self.ustar_/self.buoyancy_flux/vkb
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux, Ref.p0_half[gw], DV.values[temp_shift+ijk], PV.values[qt_shift+ijk], DV.values[qv_shift+ijk])
                    PV.tendencies[s_shift + ijk] = PV.tendencies[s_shift + ijk] + self.s_flux[ij] * tendency_factor
                    PV.tendencies[qt_shift + ijk] = PV.tendencies[qt_shift + ijk] + self.qt_flux * tendency_factor

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr,'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -self.ustar_**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.ustar_**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
                    PV.tendencies[u_shift + ijk] = PV.tendencies[u_shift + ijk] + self.u_flux[ij] * tendency_factor
                    PV.tendencies[v_shift + ijk] = PV.tendencies[v_shift + ijk] + self.v_flux[ij] * tendency_factor

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef double tmp

        tmp = Pa.HorizontalMeanSurface(Gr, &self.u_flux[0])
        NS.write_ts('uw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.v_flux[0])
        NS.write_ts('vw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.s_flux[0])
        NS.write_ts('s_flux_surface_mean', tmp, Pa)

        return


cdef class SurfaceGabls:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
        self.gustiness = 0.001
        self.z0 = 0.1
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Par)
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.u_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.v_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        # self.qt_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.s_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')


        NS.add_ts('uw_surface_mean',Gr, Pa)
        NS.add_ts('vw_surface_mean',Gr, Pa)
        NS.add_ts('s_flux_surface_mean', Gr, Pa)


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr,'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr,'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            # Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            # Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t th_shift = DV.get_varshift(Gr,'theta')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double theta_rho_b, Nb2, Ri
            double zb = Gr.dims.dx[2] * 0.5
            double cm=0.0
            double ch=0.0

            double sst = 265.0 - 0.25 * TS.t/3600.0

            # double pv_star = self.CC.LT.fast_lookup(sst)
            # double qv_star = eps_v*pv_star/((eps_v-1)*pv_star-Ref.Pg)
            # double theta_rho_g = theta_rho_c(Ref.Pg,sst,qv_star,qv_star)
            # double s_star = (1.0 - qv_star) * sd_c(Ref.Pg-pv_star,sst) + qv_star * sv_c(pv_star,sst)

            double theta_rho_g = theta_rho_c(Ref.Pg,sst,0.0,0.0)
            double s_star = sd_c(Ref.Pg,sst)
            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]/Gr.dims.dx[2]


        cdef :
            Py_ssize_t lmo_shift = DV.get_varshift_2d(Gr, 'obukhov_length')
            Py_ssize_t ustar_shift = DV.get_varshift_2d(Gr, 'friction_velocity')

        with nogil:
            for i in xrange(gw-1,imax-gw+1):
                for j in xrange(gw-1,jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    # theta_rho_b=theta_rho_c(Ref.p0_half[gw],DV.values[t_shift+ijk],PV.values[qt_shift+ijk],DV.values[qv_shift+ijk])
                    theta_rho_b=DV.values[th_shift + ijk]
                    Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
                    Ri = Nb2 * zb* zb/(windspeed[ij] * windspeed[ij])
                    exchange_coefficients_byun(Ri,zb,self.z0, &cm, &ch, &DV.values_2d[lmo_shift + ij])
                    # self.qt_flux[ij] = -ch * self.windspeed[ij] * (PV.values[qt_shift+ijk] - qv_star)
                    self.s_flux[ij] = -ch * windspeed[ij] * (PV.values[s_shift+ijk] - s_star)
                    DV.values_2d[ustar_shift + ij] = sqrt(cm) * windspeed[ij]
            for i in xrange(gw,imax-gw):
                for j in xrange(gw,jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(DV.values_2d[ustar_shift + ij], DV.values_2d[ustar_shift+ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(DV.values_2d[ustar_shift + ij], DV.values_2d[ustar_shift+ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
                    PV.tendencies[u_shift  + ijk] = PV.tendencies[u_shift  + ijk] + self.u_flux[ij] * tendency_factor
                    PV.tendencies[v_shift  + ijk] = PV.tendencies[v_shift  + ijk] + self.v_flux[ij] * tendency_factor
                    PV.tendencies[s_shift  + ijk] = PV.tendencies[s_shift  + ijk] + self.s_flux[ij] * tendency_factor
                    # PV.tendencies[qt_shift + ijk] = PV.tendencies[qt_shift + ijk] + self.qt_flux[ij]* tendency_factor
        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef double tmp


        tmp = Pa.HorizontalMeanSurface(Gr, &self.u_flux[0])
        NS.write_ts('uw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.v_flux[0])
        NS.write_ts('vw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.s_flux[0])
        NS.write_ts('s_flux_surface_mean', tmp, Pa)


        return


cdef class SurfaceDYCOMS_RF01:
    def __init__(self,namelist, LatentHeat LH):
        self.ft = 15.0
        self.fq = 115.0
        self.gustiness = 0.0
        self.cm = 0.0011
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp



    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.u_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.v_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.qt_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.s_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        self.ustar = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')

        NS.add_ts('friction_velocity_mean', Gr, Pa)
        NS.add_ts('uw_surface_mean',Gr, Pa)
        NS.add_ts('vw_surface_mean',Gr, Pa)
        NS.add_ts('s_flux_surface_mean', Gr, Pa)

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr,'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr,'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')



        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &self.windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]/Gr.dims.dx[2]
            double lam
            double lv
            double pv
            double pd
            double sv
            double sd

            double [:] windspeed = self.windspeed

        with nogil:
            for i in xrange(gw-1,imax-gw+1):
                for j in xrange(gw-1,jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j

                    self.ustar[ij] = sqrt(self.cm) * self.windspeed[ij]
                    lam = self.Lambda_fp(DV.values[t_shift+ij])
                    lv = self.L_fp(lam,DV.values[t_shift+ij])
                    pv = pv_c(Ref.p0_half[gw],PV.values[ij + qt_shift],PV.values[ij + qt_shift] - DV.values[ij + ql_shift])
                    pd = pd_c(Ref.p0_half[gw],PV.values[ij + qt_shift],PV.values[ij + qt_shift] - DV.values[ij + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ij])
                    sd = sd_c(pd,DV.values[t_shift+ij])
                    self.qt_flux[ij] = self.fq / lv
                    self.s_flux[ij] = Ref.alpha0_half[gw] * (self.ft/DV.values[t_shift+ij] + self.qt_flux[ij]*(sv - sd))
            for i in xrange(gw,imax-gw):
                for j in xrange(gw,jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(self.ustar[ij], self.ustar[ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(self.ustar[ij], self.ustar[ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
                    PV.tendencies[u_shift  + ijk] = PV.tendencies[u_shift  + ijk] + self.u_flux[ij] * tendency_factor
                    PV.tendencies[v_shift  + ijk] = PV.tendencies[v_shift  + ijk] + self.v_flux[ij] * tendency_factor
                    PV.tendencies[s_shift  + ijk] = PV.tendencies[s_shift  + ijk] + self.s_flux[ij] * tendency_factor
                    PV.tendencies[qt_shift + ijk] = PV.tendencies[qt_shift + ijk] + self.qt_flux[ij]* tendency_factor

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef double tmp

        tmp = Pa.HorizontalMeanSurface(Gr, &self.ustar[0])
        NS.write_ts('friction_velocity_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr, &self.u_flux[0])
        NS.write_ts('uw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.v_flux[0])
        NS.write_ts('vw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.s_flux[0])
        NS.write_ts('s_flux_surface_mean', tmp, Pa)

        return



# Anderson, R. J., 1993: A Study of Wind Stress and Heat Flux over the Open
# Ocean by the Inertial-Dissipation Method. J. Phys. Oceanogr., 23, 2153--“2161.
# See also: ARPS documentation
cdef inline double compute_z0(double z1, double windspeed) nogil:
    cdef double z0 =z1*exp(-kappa/sqrt((0.4 + 0.079*windspeed)*1e-3))
    return z0



