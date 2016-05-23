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
from thermodynamic_functions import exner, cpm
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
    inline double cpm_c(double qt) nogil
cdef extern from "surface.h":
    double compute_ustar(double windspeed, double buoyancy_flux, double z0, double z1) nogil
    inline double entropyflux_from_thetaflux_qtflux(double thetaflux, double qtflux, double p0_b, double T_b, double qt_b, double qv_b) nogil
    void compute_windspeed(Grid.DimStruct *dims, double* u, double*  v, double*  speed, double u0, double v0, double gustiness ) nogil
    void exchange_coefficients_byun(double Ri, double zb, double z0, double* cm, double* ch, double* lmo) nogil
cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil

def SurfaceFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):

        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
           return SurfaceSullivanPatton(LH)
        elif casename == 'Bomex':
            return SurfaceBomex(LH)
        elif casename == 'Gabls':
            return SurfaceGabls(namelist,LH)
        elif casename == 'DYCOMS_RF01':
            return SurfaceDYCOMS_RF01(namelist, LH)
        elif casename == 'DYCOMS_RF02':
            return SurfaceDYCOMS_RF02(namelist, LH)
        elif casename == 'Rico':
            return SurfaceRico(LH)
        else:
            return SurfaceNone()



cdef class SurfaceBase:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.u_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.v_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.qt_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.s_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        self.obukhov_length = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.friction_velocity = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.shf = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.lhf = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.b_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        NS.add_ts('uw_surface_mean', Gr, Pa)
        NS.add_ts('vw_surface_mean', Gr, Pa)
        NS.add_ts('s_flux_surface_mean', Gr, Pa)
        NS.add_ts('shf_surface_mean', Gr, Pa)
        NS.add_ts('lhf_surface_mean', Gr, Pa)
        NS.add_ts('obukhov_length_mean', Gr, Pa)
        NS.add_ts('friction_velocity_mean', Gr, Pa)
        NS.add_ts('buoyancy_flux_surface_mean', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        cdef :
            Py_ssize_t i, j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]-gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t ql_shift, qt_shift
            double [:] t_mean =  Pa.HorizontalMean(Gr, &DV.values[t_shift])
            double cp_, lam, lv, pv, pd, sv, sd
            double dzi = 1.0/Gr.dims.dx[2]
            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]*dzi


        if self.dry_case:
            with nogil:
                for i in xrange(gw, imax):
                    for j in xrange(gw, jmax):
                        ijk = i * istride + j * jstride + gw
                        ij = i * istride_2d + j
                        self.shf[ij] = self.s_flux[ij] * Ref.rho0_half[gw] * DV.values[t_shift+ijk]
                        self.b_flux[ij] = self.shf[ij] * g * Ref.alpha0_half[gw]/cpd/t_mean[gw]
                        self.obukhov_length[ij] = -self.friction_velocity[ij] *self.friction_velocity[ij] *self.friction_velocity[ij] /self.b_flux[ij]/vkb

                        PV.tendencies[u_shift  + ijk] +=  self.u_flux[ij] * tendency_factor
                        PV.tendencies[v_shift  + ijk] +=  self.v_flux[ij] * tendency_factor
                        PV.tendencies[s_shift  + ijk] +=  self.s_flux[ij] * tendency_factor

        else:
            ql_shift = DV.get_varshift(Gr,'ql')
            qt_shift = PV.get_varshift(Gr, 'qt')
            with nogil:
                for i in xrange(gw, imax):
                    for j in xrange(gw, jmax):
                        ijk = i * istride + j * jstride + gw
                        ij = i * istride_2d + j
                        lam = self.Lambda_fp(DV.values[t_shift+ijk])
                        lv = self.L_fp(DV.values[t_shift+ijk],lam)
                        self.lhf[ij] = self.qt_flux[ij] * Ref.rho0_half[gw] * lv
                        pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                        pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                        sv = sv_c(pv,DV.values[t_shift+ijk])
                        sd = sd_c(pd,DV.values[t_shift+ijk])
                        self.shf[ij] = (self.s_flux[ij] * Ref.rho0_half[gw] - self.lhf[ij]/lv * (sv-sd)) * DV.values[t_shift+ijk]
                        cp_ = cpm_c(PV.values[qt_shift+ijk])
                        self.b_flux[ij] = g * Ref.alpha0_half[gw]/cp_/t_mean[gw] * \
                                          (self.shf[ij] + (eps_vi-1.0)*cp_*t_mean[gw]*self.lhf[ij]/lv)
                        self.obukhov_length[ij] = -self.friction_velocity[ij] *self.friction_velocity[ij] *self.friction_velocity[ij] /self.b_flux[ij]/vkb


                        PV.tendencies[u_shift  + ijk] +=  self.u_flux[ij] * tendency_factor
                        PV.tendencies[v_shift  + ijk] +=  self.v_flux[ij] * tendency_factor
                        PV.tendencies[s_shift  + ijk] +=  self.s_flux[ij] * tendency_factor
                        PV.tendencies[qt_shift + ijk] +=  self.qt_flux[ij] * tendency_factor

        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef double tmp

        tmp = Pa.HorizontalMeanSurface(Gr, &self.u_flux[0])
        NS.write_ts('uw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr, &self.v_flux[0])
        NS.write_ts('vw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr, &self.s_flux[0])
        NS.write_ts('s_flux_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.b_flux[0])
        NS.write_ts('buoyancy_flux_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.shf[0])
        NS.write_ts('shf_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.lhf[0])
        NS.write_ts('lhf_surface_mean', tmp, Pa)

        tmp = Pa.HorizontalMeanSurface(Gr,&self.friction_velocity[0])
        NS.write_ts('friction_velocity_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.obukhov_length[0])
        NS.write_ts('obukhov_length_mean', tmp, Pa)
        return


cdef class SurfaceNone(SurfaceBase):
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        return
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class SurfaceSullivanPatton(SurfaceBase):
    def __init__(self, LatentHeat LH):
        self.theta_flux = 0.24 # K m/s
        self.z0 = 0.1 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp

        self.dry_case = True
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):


        T0 = Ref.p0_half[Gr.dims.gw] * Ref.alpha0_half[Gr.dims.gw]/Rd
        self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.dims.gw]) * g /T0
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t i, j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')


        #Get the scalar flux (dry entropy only)
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.s_flux[ij] = cpd * self.theta_flux*exner_c(Ref.p0_half[gw])/DV.values[temp_shift+ijk]

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)


        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax):
                for j in xrange(1,jmax):
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = compute_ustar(windspeed[ij],self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) \
                                      * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) \
                                      * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)
        return



cdef class SurfaceBomex(SurfaceBase):
    def __init__(self,  LatentHeat LH):
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.dry_case = False
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.qt_flux = np.add(self.qt_flux,5.2e-5) # m/s

        self.theta_flux = 8.0e-3 # K m/s
        self.ustar_ = 0.28 #m/s
        self.theta_surface = 299.1 #K
        self.qt_surface = 22.45e-3 # kg/kg
        self.buoyancy_flux = g * ((self.theta_flux + (eps_vi-1.0)*(self.theta_surface*self.qt_flux[0]
                                                                   + self.qt_surface *self.theta_flux))
                              /(self.theta_surface*(1.0 + (eps_vi-1)*self.qt_surface)))


        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return


        cdef :
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


        # Get the scalar flux
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = self.ustar_
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux[ij], Ref.p0_half[gw],
                                                                        DV.values[temp_shift+ijk], PV.values[qt_shift+ijk], DV.values[qv_shift+ijk])

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)

        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -self.ustar_**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.ustar_**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)


        return


cdef class SurfaceGabls(SurfaceBase):
    def __init__(self, namelist,  LatentHeat LH):
        self.gustiness = 0.001
        self.z0 = 0.1
        # Rate of change of surface temperature, in K/hour
        # GABLS1 IC (Beare et al) value is 0.25 (given as default)
        try:
            self.cooling_rate = namelist['surface']['cooling_rate']
        except:
            self.cooling_rate = 0.25

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp

        self.dry_case = True


        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_shift = DV.get_varshift(Gr, 'theta')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)

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
            double [:] cm= np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double ch=0.0


            double sst = 265.0 - self.cooling_rate * TS.t/3600.0 # sst = theta_surface also


            double theta_rho_g = theta_rho_c(Ref.Pg, sst, 0.0, 0.0)
            double s_star = sd_c(Ref.Pg,sst)
            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]/Gr.dims.dx[2]


        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1,jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    theta_rho_b = DV.values[th_shift + ijk]
                    Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
                    Ri = Nb2 * zb* zb/(windspeed[ij] * windspeed[ij])
                    exchange_coefficients_byun(Ri,zb,self.z0, &cm[ij], &ch, &self.obukhov_length[ij])
                    self.s_flux[ij] = -ch * windspeed[ij] * (PV.values[s_shift+ijk] - s_star)
                    self.friction_velocity[ij] = sqrt(cm[ij]) * windspeed[ij]
            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return


cdef class SurfaceDYCOMS_RF01(SurfaceBase):
    def __init__(self,namelist, LatentHeat LH):
        self.ft = 15.0
        self.fq = 115.0
        self.gustiness = 0.0
        self.cm = 0.0011
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        sst = 292.5 # K
        psurface = 1017.8e2 # Pa
        theta_surface = sst/exner(psurface)
        qt_surface = 13.84e-3 # qs(sst) using Teten's formula
        density_surface = 1.22 #kg/m^3
        theta_flux = self.ft/(density_surface*cpm(qt_surface)*exner(psurface))
        qt_flux_ = self.fq/self.L_fp(sst,self.Lambda_fp(sst))
        self.buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*qt_flux_ + qt_surface * theta_flux))
                              /(theta_surface*(1.0 + (eps_vi-1)*qt_surface)))

        self.dry_case = False

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')



        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &self.windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double lam, lv, pv, pd, sv, sd

            double [:] windspeed = self.windspeed


        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = sqrt(self.cm) * self.windspeed[ij]
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    self.qt_flux[ij] = self.fq / lv / 1.22
                    self.s_flux[ij] = Ref.alpha0_half[gw] * (self.ft/DV.values[t_shift+ijk] + self.fq*(sv - sd)/lv)
            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa,TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return


cdef class SurfaceDYCOMS_RF02(SurfaceBase):
    def __init__(self,namelist, LatentHeat LH):
        self.ft = 16.0
        self.fq = 93.0
        self.gustiness = 0.0
        self.ustar = 0.25
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        sst = 292.5 # K
        psurface = 1017.8e2 # Pa
        theta_surface = sst/exner(psurface)
        qt_surface = 13.84e-3 # qs(sst) using Teten's formula
        density_surface = 1.22 #kg/m^3
        theta_flux = self.ft/(density_surface*cpm(qt_surface)*exner(psurface))
        qt_flux_ = self.fq/self.L_fp(sst,self.Lambda_fp(sst))
        self.buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*qt_flux_ + qt_surface * theta_flux))
                              /(theta_surface*(1.0 + (eps_vi-1)*qt_surface)))

        self.dry_case = False




    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')



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
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = self.ustar
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    self.qt_flux[ij] = self.fq / lv / 1.21
                    self.s_flux[ij] = Ref.alpha0_half[gw] * (self.ft/DV.values[t_shift+ijk] + self.fq*(sv - sd)/lv)
            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -self.ustar*self.ustar / interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.ustar*self.ustar / interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)


        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return




cdef class SurfaceRico(SurfaceBase):
    def __init__(self, LatentHeat LH):
        self.cm =0.001229
        self.ch = 0.001094
        self.cq = 0.001133
        self.z0 = 0.00015
        self.gustiness = 0.0
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.dry_case = False
        return


    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)

        self.cm = self.cm*(log(20.0/self.z0)/log(Gr.zl_half[Gr.dims.gw]/self.z0))**2
        self.ch = self.ch*(log(20.0/self.z0)/log(Gr.zl_half[Gr.dims.gw]/self.z0))**2
        self.cq = self.cq*(log(20.0/self.z0)/log(Gr.zl_half[Gr.dims.gw]/self.z0))**2


        cdef double pv_star = pv_c(Ref.Pg, Ref.qtg, Ref.qtg)
        cdef double  pd_star = Ref.Pg - pv_star
        self.s_star = (1.0-Ref.qtg) * sd_c(pd_star, Ref.Tg) + Ref.qtg * sv_c(pv_star,Ref.Tg)



        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')

            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double ustar_
            double buoyancy_flux, theta_flux
            double theta_surface = Ref.Tg * exner_c(Ref.Pg)

            double cm_sqrt = sqrt(self.cm)

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        with nogil:
            for i in xrange(gw, imax-gw):
                for j in xrange(gw,jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    theta_flux = -self.ch * windspeed[ij] * (DV.values[t_shift + ijk]*exner_c(Ref.p0_half[gw]) - theta_surface)

                    self.s_flux[ij]  = -self.ch * windspeed[ij] * (PV.values[s_shift + ijk] - self.s_star)
                    self.qt_flux[ij] = -self.cq * windspeed[ij] * (PV.values[qt_shift + ijk] - Ref.qtg)
                    buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*self.qt_flux[ij] + Ref.qtg * theta_flux))/(theta_surface*(1.0 + (eps_vi-1)*Ref.qtg)))
                    self.u_flux[ij]  = -self.cm * interp_2(windspeed[ij], windspeed[ij + istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij + 1])* (PV.values[v_shift + ijk] + Ref.v0)
                    ustar_ = cm_sqrt * windspeed[ij]
                    self.friction_velocity[ij] = ustar_

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)

        return




    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)
        return






# Anderson, R. J., 1993: A Study of Wind Stress and Heat Flux over the Open
# Ocean by the Inertial-Dissipation Method. J. Phys. Oceanogr., 23, 2153--“2161.
# See also: ARPS documentation
cdef inline double compute_z0(double z1, double windspeed) nogil:
    cdef double z0 =z1*exp(-kappa/sqrt((0.4 + 0.079*windspeed)*1e-3))
    return z0

