#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport TimeStepping
cimport Lookup
cimport Thermodynamics
import cython
from Thermodynamics cimport LatentHeat, ClausiusClapeyron

include 'parameters_micro.pxi'
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI

from libc.math cimport fmax, fmin, fabs

cdef extern from "microphysics_arctic_1m.h":
    inline double entropy_src_precipitation_c(double p0, double T, double qt, double qv,
                                              double L, double precip_rate) nogil
    inline double entropy_src_evaporation_c(double p0, double T, double Tw, double qt, double qv,
                                            double L, double evap_rate) nogil
    void sedimentation_velocity_rain(Grid.DimStruct *dims, double* density, double* nrain, double* qrain,
                                     double* qrain_velocity) nogil
    void sedimentation_velocity_snow(Grid.DimStruct *dims, double* density, double* nsnow, double* qsnow,
                                     double* qsnow_velocity) nogil
    void entropy_source_heating_rain(Grid.DimStruct *dims, double* T, double* Twet, double* qrain,
                                     double* w_qrain, double* w,  double* entropy_tendency) nogil
    void entropy_source_heating_snow(Grid.DimStruct *dims, double* T, double* Twet, double* qsnow,
                                     double* w_qsnow, double* w,  double* entropy_tendency) nogil
    void entropy_source_drag(Grid.DimStruct *dims, double* T, double* qprec, double* w_qprec,
                             double* entropy_tendency) nogil
    void get_virtual_potential_temperature(Grid.DimStruct *dims, double* p0, double* T, double* qv,
                                       double* ql, double* qi, double* thetav) nogil
    void microphysics_sources(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double),
                             double (*L_fp)(double, double), double* density, double* p0,
                             double* temperature, double* qt, double ccn, double n0_ice,
                             double* ql, double* qi, double* qrain, double* nrain,
                             double* qsnow, double* nsnow, double dt,
                             double* qrain_tendency_micro, double* qrain_tendency,
                             double* qsnow_tendency_micro, double* qsnow_tendency,
                             double* precip_rate, double* evap_rate) nogil
    void qt_source_formation(Grid.DimStruct *dims, double* qt_tendency, double* precip_rate, double* evap_rate) nogil
    void evaporation_snow_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double),
                              double (*L_fp)(double, double), double* density, double* p0, double* temperature,
                              double* qt, double* qsnow, double* nsnow, double* qsnow_tendency) nogil
    void accretion_all_wrapper(Grid.DimStruct *dims, double* density, double* p0, double* temperature, double n0_ice, double ccn,
                           double* ql, double* qi, double* qrain, double* nrain, double* qsnow, double* nsnow,
                           double* ql_tendency, double* qi_tendency, double* qrain_tendency, double* qsnow_tendency) nogil
    void autoconversion_snow_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double),
                                 double (*L_fp)(double, double), double n0_ice, double* density, double* p0, double* temperature,
                                 double* qt, double* qi, double* qsnow_tendency) nogil
    void melt_snow_wrapper(Grid.DimStruct *dims, double* density, double* temperature, double* qsnow, double* nsnow,
                           double* qsnow_tendency) nogil
    void autoconversion_rain_wrapper(Grid.DimStruct *dims, double* density, double ccn, double* ql, double* qrain,
                                     double* nrain, double* qrain_tendency) nogil
    void evaporation_rain_wrapper(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double (*lam_fp)(double),
                                  double (*L_fp)(double, double), double* density, double* p0, double* temperature,
                                  double* qt, double* qrain, double* nrain, double* qrain_tendency) nogil
    void get_rain_n0(Grid.DimStruct *dims, double* density, double* qrain, double* nrain) nogil
    void get_snow_n0(Grid.DimStruct *dims, double* density, double* qsnow, double* nsnow) nogil


cdef extern from "microphysics.h":
    void microphysics_wetbulb_temperature(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double* p0, double* s,
                                          double* qt,  double* T, double* Twet )nogil

cdef extern from "advection_interpolation.h":
    double interp_2(double phi, double phip1) nogil

cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *velocity,
                                    double *scalar, double* flux, int d, int scheme) nogil

cdef class Microphysics_Arctic_1M:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):

        self.thermodynamics_type = 'SA'

        #Get namelist variables
        try:
            self.ccn = namelist['microphysics']['ccn']
        except:
            self.ccn = 100.0e6

        try:
            self.n0_ice_input = namelist['microphysics']['n0_ice']
            Par.root_print('set n0_ice to be '+self.n0_ice_input)
        except:
            self.n0_ice_input = 1.0e7
            Par.root_print('default n0_ice value 1.0e7')

        try:
            self.order = namelist['scalar_transport']['order_sedimentation']
        except:
            self.order = namelist['scalar_transport']['order']

        try:
            if namelist['microphysics']['phase_partitioning'] == 'liquid_only':
                self.Lambda_fp = lambda_constant_Arctic
                LH.Lambda_fp = lambda_constant_Arctic
                Par.root_print('liquid only microphysics set Lambda = 1.0 ')
            else:
                self.Lambda_fp = lambda_Arctic
                LH.Lambda_fp = lambda_Arctic
        except:
            self.Lambda_fp = lambda_Arctic
            LH.Lambda_fp = lambda_Arctic


        LH.L_fp = latent_heat_Arctic

        self.L_fp = latent_heat_Arctic
        # self.Lambda_fp = LH.Lambda_fp

        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Par)

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.precip_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.evap_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        #Add precipitation variables
        PV.add_variable('qr', 'kg/kg', r'q_r', 'rain water specific humidity','sym','scalar',Pa)
        PV.add_variable('qs', 'kg/kg', r'q_r', 'snow water specific humidity','sym','scalar',Pa)

        # add sedimentation velocities as diagnostic variables (the format has to be w_q)
        DV.add_variables('w_qr', 'm/s', r'w_{qr}', r'rain mass sedimentation velocity', 'sym', Pa)
        DV.add_variables('w_qs', 'm/s', r'w_{qs}', r'snow mass sedimentation velocity', 'sym', Pa)

        # add number concentrations as DV
        DV.add_variables('nr', '1/kg', r'n_r', r'rain droplet number concentration','sym', Pa)
        DV.add_variables('ns', '1/kg', r'n_s', r'snow number concentration','sym', Pa)

        # add wet bulb temperature
        DV.add_variables('temperature_wb', 'K', r'T_{wb}','wet bulb temperature','sym', Pa)

        NS.add_profile('evap_rate', Gr, Pa)
        NS.add_profile('precip_rate', Gr, Pa)
        NS.add_profile('rain_auto_mass', Gr, Pa)
        NS.add_profile('snow_auto_mass', Gr, Pa)
        NS.add_profile('rain_accr_mass', Gr, Pa)
        NS.add_profile('snow_accr_mass', Gr, Pa)
        NS.add_profile('rain_evap_mass', Gr, Pa)
        NS.add_profile('snow_depo_mass', Gr, Pa)
        NS.add_profile('snow_melt_mass', Gr, Pa)

        NS.add_profile('rain_sedimentation_flux', Gr, Pa)
        NS.add_profile('snow_sedimentation_flux', Gr, Pa)

        NS.add_profile('micro_s_source_precipitation', Gr, Pa)
        NS.add_profile('micro_s_source_evaporation', Gr, Pa)

        NS.add_profile('thetav_mean', Gr, Pa)
        NS.add_profile('thetav_flux_z', Gr, Pa)

        NS.add_ts('iwp', Gr, Pa)
        NS.add_ts('rwp', Gr, Pa)
        NS.add_ts('swp', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        #Get parameters
        cdef:

            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t qrain_shift = PV.get_varshift(Gr, 'qr')
            Py_ssize_t qsnow_shift = PV.get_varshift(Gr, 'qs')
            Py_ssize_t nrain_shift = DV.get_varshift(Gr, 'nr')
            Py_ssize_t nsnow_shift = DV.get_varshift(Gr, 'ns')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t tw_shift = DV.get_varshift(Gr, 'temperature_wb')
            Py_ssize_t wqrain_shift = DV.get_varshift(Gr, 'w_qr')
            Py_ssize_t wqsnow_shift = DV.get_varshift(Gr, 'w_qs')

            double [:] qrain_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] qsnow_tend_micro = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')


        # Calculate sedimentation before anything else to get N0

        get_rain_n0(&Gr.dims, &Ref.rho0_half[0], &PV.values[qrain_shift], &DV.values[nrain_shift])
        get_snow_n0(&Gr.dims, &Ref.rho0_half[0], &PV.values[qsnow_shift], &DV.values[nsnow_shift])

        # Microphysics source terms

        microphysics_sources(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &Ref.rho0_half[0],
                             &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], self.ccn, self.n0_ice_input,
                             &DV.values[ql_shift], &DV.values[qi_shift], &PV.values[qrain_shift], &DV.values[nrain_shift],
                             &PV.values[qsnow_shift], &DV.values[nsnow_shift], TS.dt,
                             &qrain_tend_micro[0], &PV.tendencies[qrain_shift],
                             &qsnow_tend_micro[0], &PV.tendencies[qsnow_shift], &self.precip_rate[0], &self.evap_rate[0])

        sedimentation_velocity_rain(&Gr.dims, &Ref.rho0_half[0], &DV.values[nrain_shift], &PV.values[qrain_shift],
                                     &DV.values[wqrain_shift])

        sedimentation_velocity_snow(&Gr.dims, &Ref.rho0_half[0], &DV.values[nsnow_shift], &PV.values[qsnow_shift],
                                     &DV.values[wqsnow_shift])

        qt_source_formation(&Gr.dims, &PV.tendencies[qt_shift], &self.precip_rate[0], &self.evap_rate[0])

        # #Add entropy tendency due to microphysics (precipitation and evaporation only)
        microphysics_wetbulb_temperature(&Gr.dims, &self.CC.LT.LookupStructC, &Ref.p0_half[0], &PV.values[s_shift],
                                          &PV.values[qt_shift], &DV.values[t_shift], &DV.values[tw_shift])

        get_s_source_precip(&Gr.dims, Th, &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift],
                               &self.precip_rate[0], &PV.tendencies[s_shift])

        get_s_source_evap(&Gr.dims, Th, &Ref.p0_half[0], &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qt_shift], &DV.values[qv_shift],
                             &self.evap_rate[0], &PV.tendencies[s_shift])

        entropy_source_heating_rain(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qrain_shift],
                                   &DV.values[wqrain_shift],  &PV.values[w_shift], &PV.tendencies[s_shift])

        entropy_source_heating_snow(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qsnow_shift],
                                   &DV.values[wqsnow_shift],  &PV.values[w_shift], &PV.tendencies[s_shift])

        entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qrain_shift], &DV.values[wqrain_shift],
                             &PV.tendencies[s_shift])

        entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qsnow_shift], &DV.values[wqsnow_shift],
                             &PV.tendencies[s_shift])



        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t tw_shift = DV.get_varshift(Gr, 'temperature_wb')
            double [:] tmp = np.zeros((Gr.dims.npg), dtype=np.double, order='c')
            double [:] tmp_tendency = np.zeros((Gr.dims.npg), dtype=np.double, order='c')
            Py_ssize_t qrain_shift = PV.get_varshift(Gr, 'qr')
            Py_ssize_t qsnow_shift = PV.get_varshift(Gr, 'qs')
            Py_ssize_t wqrain_shift = DV.get_varshift(Gr, 'w_qr')
            Py_ssize_t wqsnow_shift = DV.get_varshift(Gr, 'w_qs')
            Py_ssize_t nsnow_shift = DV.get_varshift(Gr, 'ns')
            Py_ssize_t nrain_shift = DV.get_varshift(Gr, 'nr')
            double [:] dummy =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] dummy2 =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] dummy3 =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] dummy4 =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')


        tmp = Pa.HorizontalMean(Gr, &self.precip_rate[0])
        NS.write_profile('precip_rate', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.evap_rate[0])
        NS.write_profile('evap_rate', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        autoconversion_rain_wrapper(&Gr.dims, &RS.rho0_half[0], self.ccn, &DV.values[ql_shift], &PV.values[qrain_shift],
                                     &DV.values[nrain_shift], &dummy[0])
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('rain_auto_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        dummy[:] = 0.0
        autoconversion_snow_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, self.n0_ice_input,
                                    &RS.rho0_half[0], &RS.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift],
                                    &DV.values[qi_shift], &dummy[0])
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('snow_auto_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        dummy[:] = 0.0
        evaporation_rain_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.rho0_half[0],
                                 &RS.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], &PV.values[qrain_shift],
                                 &DV.values[nrain_shift], &dummy[0])
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('rain_evap_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        dummy[:] = 0.0
        evaporation_snow_wrapper(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.rho0_half[0],
                                 &RS.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], &PV.values[qsnow_shift],
                                 &DV.values[nsnow_shift], &dummy[0])
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('snow_depo_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        dummy[:] = 0.0
        accretion_all_wrapper(&Gr.dims, &RS.rho0_half[0], &RS.p0_half[0], &DV.values[t_shift], self.n0_ice_input, self.ccn,
                              &DV.values[ql_shift], &DV.values[qi_shift], &PV.values[qrain_shift], &DV.values[nrain_shift],
                              &PV.values[qsnow_shift], &DV.values[nsnow_shift], &dummy2[0], &dummy3[0],
                              &dummy4[0], &dummy[0])
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('snow_accr_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &dummy4[0])
        NS.write_profile('rain_accr_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        dummy[:] = 0.0
        melt_snow_wrapper(&Gr.dims, &RS.rho0_half[0], &DV.values[t_shift], &PV.values[qsnow_shift],
                          &DV.values[nsnow_shift], &dummy[0])
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('snow_melt_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        #compute sedimentation flux only of qrain and qsnow
        compute_advective_fluxes_a(&Gr.dims, &RS.rho0[0], &RS.rho0_half[0], &DV.values[wqrain_shift],
                                   &PV.values[qrain_shift], &dummy[0], 2, self.order)
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('rain_sedimentation_flux', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        compute_advective_fluxes_a(&Gr.dims, &RS.rho0[0], &RS.rho0_half[0], &DV.values[wqsnow_shift],
                                   &PV.values[qsnow_shift], &dummy[0], 2, self.order)
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('snow_sedimentation_flux', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        get_virtual_potential_temperature(&Gr.dims, &RS.p0_half[0], &DV.values[t_shift], &DV.values[qv_shift],
                                          &DV.values[ql_shift], &DV.values[qi_shift], &dummy[0])
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('thetav_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanofSquares(Gr, &PV.values[w_shift], &dummy[0])
        NS.write_profile('thetav_flux_z', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        #Output entropy source terms due to microphysics

        get_s_source_precip(&Gr.dims, Th, &RS.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift],
                            &self.precip_rate[0], &tmp_tendency[0])
        tmp = Pa.HorizontalMean(Gr, &tmp_tendency[0])
        NS.write_profile('micro_s_source_precipitation', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp_tendency[:] = 0.0
        get_s_source_evap(&Gr.dims, Th, &RS.p0_half[0], &DV.values[t_shift], &DV.values[tw_shift],&PV.values[qt_shift], &DV.values[qv_shift],
                            &self.evap_rate[0], &tmp_tendency[0])
        tmp = Pa.HorizontalMean(Gr, &tmp_tendency[0])
        NS.write_profile('micro_s_source_evaporation', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        self.ice_stats(Gr, RS, PV, DV, NS, Pa)

        return

    cpdef ice_stats(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t kmin = 0
            Py_ssize_t kmax = Gr.dims.n[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t pi, k
            ParallelMPI.Pencil z_pencil = ParallelMPI.Pencil()
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi')
            Py_ssize_t qrain_shift = PV.get_varshift(Gr, 'qr')
            Py_ssize_t qsnow_shift = PV.get_varshift(Gr, 'qs')
            double[:, :] qi_pencils
            double[:, :] qrain_pencils
            double[:, :] qsnow_pencils
            # Cloud indicator
            double[:] ci
            double cb
            double ct
            # Weighted sum of local cloud indicator
            double ci_weighted_sum = 0.0
            double mean_divisor = np.double(Gr.dims.n[0] * Gr.dims.n[1])

            double dz = Gr.dims.dx[2]
            double[:] iwp
            double[:] rwp
            double[:] swp
            double iwp_weighted_sum = 0.0
            double rwp_weighted_sum = 0.0
            double swp_weighted_sum = 0.0

            double[:] cf_profile = np.zeros((Gr.dims.n[2]), dtype=np.double, order='c')

        # Initialize the z-pencil
        z_pencil.initialize(Gr, Pa, 2)
        qi_pencils =  z_pencil.forward_double( &Gr.dims, Pa, &DV.values[qi_shift])
        qrain_pencils =  z_pencil.forward_double( &Gr.dims, Pa, &PV.values[qrain_shift])
        qsnow_pencils =  z_pencil.forward_double( &Gr.dims, Pa, &PV.values[qsnow_shift])


        # Compute liquid, ice, rain, and snow water paths
        iwp = np.empty((z_pencil.n_local_pencils), dtype=np.double, order='c')
        rwp = np.empty((z_pencil.n_local_pencils), dtype=np.double, order='c')
        swp = np.empty((z_pencil.n_local_pencils), dtype=np.double, order='c')
        with nogil:
            for pi in xrange(z_pencil.n_local_pencils):
                iwp[pi] = 0.0
                rwp[pi] = 0.0
                swp[pi] = 0.0
                for k in xrange(kmin, kmax):
                    iwp[pi] += RS.rho0_half[k] * qi_pencils[pi, k] * dz
                    rwp[pi] += RS.rho0_half[k] * qrain_pencils[pi, k] * dz
                    swp[pi] += RS.rho0_half[k] * qsnow_pencils[pi, k] * dz

            for pi in xrange(z_pencil.n_local_pencils):
                iwp_weighted_sum += iwp[pi]
                rwp_weighted_sum += rwp[pi]
                swp_weighted_sum += swp[pi]

            iwp_weighted_sum /= mean_divisor
            rwp_weighted_sum /= mean_divisor
            swp_weighted_sum /= mean_divisor

        iwp_weighted_sum = Pa.domain_scalar_sum(iwp_weighted_sum)
        NS.write_ts('iwp', iwp_weighted_sum, Pa)

        rwp_weighted_sum = Pa.domain_scalar_sum(rwp_weighted_sum)
        NS.write_ts('rwp', rwp_weighted_sum, Pa)

        swp_weighted_sum = Pa.domain_scalar_sum(swp_weighted_sum)
        NS.write_ts('swp', swp_weighted_sum, Pa)

        return

cdef get_s_source_precip(Grid.DimStruct *dims, Th, double *p0_half, double *t, double *qt, double *qv, double *precip_rate, double *s_tendency):
    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double L

    for i in xrange(imin,imax):
        ishift = i * istride
        for j in xrange(jmin,jmax):
            jshift = j * jstride
            for k in xrange(kmin,kmax):
                ijk = ishift + jshift + k
                L = Th.get_lh(t[ijk])
                s_tendency[ijk] += entropy_src_precipitation_c(p0_half[k], t[ijk], qt[ijk], qv[ijk], L, precip_rate[ijk])

    return

cdef get_s_source_evap(Grid.DimStruct *dims, Th, double *p0_half, double *t, double *tw, double *qt, double *qv, double *evap_rate, double *s_tendency):
    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double L

    for i in xrange(imin,imax):
        ishift = i * istride
        for j in xrange(jmin,jmax):
            jshift = j * jstride
            for k in xrange(kmin,kmax):
                ijk = ishift + jshift + k
                L = Th.get_lh(t[ijk])
                s_tendency[ijk] += entropy_src_evaporation_c(p0_half[k], t[ijk], tw[ijk], qt[ijk], qv[ijk], L, evap_rate[ijk])

    return
