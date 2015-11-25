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


from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI

from libc.math cimport fmax, fmin, fabs

include 'micro_parameters.pxi'

cdef extern from "microphysics_arctic_1m.h":
    void micro_substep_c(Lookup.LookupStruct *LT, double alpha, double p0, double qt, double qi, double T, double cnn, double n0_ice,
                         hm_parameters *rain_param, hm_parameters *snow_param, hm_parameters *liquid_param, hm_parameters *ice_param,
                         hm_properties *rain_prop, hm_properties *snow_prop, hm_properties *liquid_prop, hm_properties *ice_prop,
                         double* aut_rain, double* aut_snow, ret_acc *src_acc, double* evp_rain,
                         double* evp_snow, double* melt_snow) nogil

    inline double get_rain_vel_c(double alpha, double qrain, hm_parameters *rain_param, hm_properties *rain_prop) nogil
    inline double get_snow_vel_c(double alpha, double qsnow, hm_parameters *snow_param, hm_properties *snow_prop) nogil
    inline double get_n0_snow_c(double alpha, double qsnow, hm_parameters *snow_param) nogil
    inline double get_n0_rain_c(double alpha, double qrain, hm_parameters *rain_param) nogil
    inline double get_n0_ice_c(double alpha, double qi, double ice_n0, hm_parameters *ice_param) nogil
    inline double get_lambda_c(double alpha, hm_properties *_prop, hm_parameters *_param) nogil
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
    void entropy_source_drag(Grid.DimStruct *dims, double* T,  double* qprec, double* w_qprec,
                             double* entropy_tendency) nogil


cdef extern from "microphysics.h":
    void microphysics_wetbulb_temperature(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double* p0, double* s,
                                          double* qt,  double* T, double* Twet )nogil

cdef extern from "advection_interpolation.h":
    double interp_2(double phi, double phip1) nogil

cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *velocity, double *scalar, double* flux, int d, int scheme) nogil

cdef class Microphysics_Arctic_1M:
    def __init__(self, ParallelMPI.ParallelMPI Par, LatentHeat LH, namelist):

        LH.Lambda_fp = lambda_Arctic
        LH.L_fp = latent_heat_Arctic

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

        # self.L_fp = LH.L_fp
        # self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Par)

        self.z_pencil = ParallelMPI.Pencil()

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.z_pencil.initialize(Gr, Pa, 2)

        #Define all source terms that need to be stored
        #Ghosted or not???
        self.autoconversion = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.evaporation = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.accretion = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.melting = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        # self.qrain_flux = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.qsnow_flux = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.qrain_tendency = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.qsnow_tendency = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.qrain_vel = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.qsnow_vel = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        # self.rain_number_density = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.snow_number_density = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.ice_number_density = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.ice_lambda = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.snow_lambda = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.n0_snow = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        # self.n0_ice = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        self.precip_rate = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.evap_rate = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        #Add precipitation variables
        PV.add_variable('qrain', 'kg kg^-1', "sym", "scalar", Pa)
        PV.add_variable('qsnow', 'kg kg^-1', "sym", "scalar", Pa)

        # add sedimentation velocities as diagnostic variables (the format has to be w_q)
        DV.add_variables('w_qrain', 'm/s', 'sym', Pa)
        DV.add_variables('w_qsnow', 'm/s', 'sym', Pa)

        # add number concentrations as DV
        DV.add_variables('nrain', '1/kg', 'sym', Pa)
        DV.add_variables('nsnow', '1/kg', 'sym', Pa)

        # add wet bulb temperature
        DV.add_variables('temperature_wb', 'K', 'sym', Pa)

        # #Initialize Statistical Output
        # NS.add_profile('n_rain_mean', Gr, Pa)
        # NS.add_profile('n_snow_mean', Gr, Pa)
        # NS.add_profile('n_ice_mean', Gr, Pa)
        # NS.add_profile('snow_lambda', Gr, Pa)
        # NS.add_profile('ice_lambda', Gr, Pa)
        # NS.add_profile('n0_snow', Gr, Pa)
        # NS.add_profile('n0_ice', Gr, Pa)


        NS.add_profile('rain_auto_mass', Gr, Pa)
        NS.add_profile('snow_auto_mass', Gr, Pa)
        NS.add_profile('rain_accr_mass', Gr, Pa)
        NS.add_profile('snow_accr_mass', Gr, Pa)
        NS.add_profile('rain_evap_mass', Gr, Pa)
        NS.add_profile('snow_depo_mass', Gr, Pa)
        NS.add_profile('snow_melt_mass', Gr, Pa)

        # NS.add_profile('rain_sedimentation_velocity', Gr, Pa)
        # NS.add_profile('snow_sedimentation_velocity', Gr, Pa)
        NS.add_profile('rain_sedimentation_flux', Gr, Pa)
        NS.add_profile('snow_sedimentation_flux', Gr, Pa)
        # NS.add_profile('rain_sedimentation_tendency', Gr, Pa)
        # NS.add_profile('snow_sedimentation_tendency', Gr, Pa)

        NS.add_profile('micro_s_source_precipitation', Gr, Pa)
        NS.add_profile('micro_s_source_evaporation', Gr, Pa)

        NS.add_ts('iwp', Gr, Pa)
        NS.add_ts('rwp', Gr, Pa)
        NS.add_ts('swp', Gr, Pa)


        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        #Get parameters
        cdef:
            #struct pointers???
            hm_properties rain_prop
            hm_properties snow_prop
            hm_properties ice_prop
            hm_properties liquid_prop

            ret_acc src_acc

            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk, pi
            Py_ssize_t allshift = Gr.dims.npg

            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t qrain_shift = PV.get_varshift(Gr, 'qrain')
            Py_ssize_t qsnow_shift = PV.get_varshift(Gr, 'qsnow')
            Py_ssize_t nrain_shift = DV.get_varshift(Gr, 'nrain')
            Py_ssize_t nsnow_shift = DV.get_varshift(Gr, 'nsnow')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t tw_shift = DV.get_varshift(Gr, 'temperature_wb')
            Py_ssize_t wqrain_shift = DV.get_varshift(Gr, 'w_qrain')
            Py_ssize_t wqsnow_shift = DV.get_varshift(Gr, 'w_qsnow')

            double [:] aut = self.autoconversion
            double [:] evp = self.evaporation
            double [:] acc = self.accretion
            double [:] melt = self.melting

            double aut_rain, aut_snow, evp_rain, evp_snow, melt_snow
            double qt_micro

            double iter_count, time_added, dt_, rate, rate1, rate2, rate3, rate4
            double machine_eps = np.finfo(np.float64).eps
            double Tw, L

            double [:] precip_rate = self.precip_rate
            double [:] evap_rate = self.evap_rate




        #Start the Main loop
        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k

                        #Initialize all source terms to be zeros
                        aut[ijk] = 0.0
                        aut[ijk+allshift] = 0.0
                        evp[ijk] = 0.0
                        evp[ijk+allshift] = 0.0
                        acc[ijk] = 0.0
                        acc[ijk+allshift] = 0.0
                        melt[ijk] = 0.0

                        #Prepare for substepping
                        #Assign mass fraction of each species to property structs
                        rain_prop.mf = PV.values[qrain_shift + ijk]
                        snow_prop.mf = PV.values[qsnow_shift + ijk]
                        liquid_prop.mf = DV.values[ql_shift + ijk]
                        ice_prop.mf = DV.values[qi_shift + ijk]
                        qt_micro = PV.values[qt_shift + ijk]

                        #Do substepping (iterations < 20)

                        iter_count = 0
                        time_added = 0.0

                        while time_added < TS.dt and iter_count < 1:
                            if (liquid_prop.mf+ice_prop.mf) < small and (rain_prop.mf+snow_prop.mf) < small:
                                break

                            micro_substep_c(&self.CC.LT.LookupStructC, Ref.alpha0_half[k], Ref.p0_half[k], qt_micro, DV.values[qi_shift + ijk], DV.values[t_shift + ijk],
                                            self.ccn, self.n0_ice_input, &rain_param, &snow_param, &liquid_param, &ice_param,
                                            &rain_prop, &snow_prop, &liquid_prop, &ice_prop, &aut_rain, &aut_snow,
                                            &src_acc, &evp_rain, &evp_snow, &melt_snow)

                            dt_ = TS.dt - time_added
                            rate1 = 1.05 * (aut_rain + src_acc.dyr +  evp_rain - melt_snow)/(-fmax(rain_prop.mf,machine_eps)/dt_)
                            rate2 = 1.05 * (aut_snow + src_acc.dys +  evp_snow + melt_snow)/(-fmax(snow_prop.mf,machine_eps)/dt_)
                            rate3 = 1.05 * (-aut_rain + src_acc.dyl)/(-fmax(liquid_prop.mf,machine_eps)/dt_)
                            rate4 = 1.05 * (-aut_snow + src_acc.dyi)/(-fmax(ice_prop.mf,machine_eps)/dt_)

                            rate = fmax(fmax(fmax(rate1,rate2),rate3),rate4)
                            if rate > 1.0:
                                # Limit the timestep, but don't let it get too small
                                dt_ = fmax(dt_/rate,1e-10)


                            # Integrate forward in time
                            rain_prop.mf = fmax(rain_prop.mf + (aut_rain + src_acc.dyr + evp_rain - melt_snow)* dt_,0.0)
                            snow_prop.mf = fmax(snow_prop.mf + (aut_snow + src_acc.dys + evp_snow + melt_snow)* dt_,0.0)
                            liquid_prop.mf = fmax(liquid_prop.mf + (-aut_rain + src_acc.dyl) * dt_,0.0)
                            ice_prop.mf = fmax(ice_prop.mf + (-aut_snow + src_acc.dyi) * dt_,0.0)
                            #vapor_star = fmax(vapor_star + (-evp_rain - evp_snow) * dt_,0.0)

                            precip_rate[ijk] = -aut_rain + src_acc.dyl - aut_snow + src_acc.dyi
                            evap_rate[ijk] = evp_rain + evp_snow
                            qt_micro = fmax(qt_micro + (precip_rate[ijk] - evap_rate[ijk])*dt_, 0.0)
                            #qt_micro = fmax(qt_micro + (-aut_rain + src_acc.dyl - aut_snow + src_acc.dyi - evp_rain - evp_snow)*dt_, 0.0)

                            # Update the contributions of each source term
                            aut[ijk] = aut[ijk] + aut_rain * dt_/TS.dt
                            acc[ijk] = acc[ijk] + src_acc.dyr * dt_/TS.dt
                            evp[ijk] = evp[ijk] + evp_rain * dt_/TS.dt
                            aut[ijk+allshift] = aut[ijk+allshift] + aut_snow * dt_/TS.dt
                            acc[ijk+allshift] = acc[ijk+allshift] + src_acc.dys * dt_/TS.dt
                            evp[ijk+allshift] = evp[ijk+allshift] + evp_snow * dt_/TS.dt
                            melt[ijk] = melt[ijk] + melt_snow * dt_/TS.dt

                            # Increment the local time variables
                            time_added = time_added + dt_
                            iter_count += 1

                            # if iter_count > 19:
                            #     with gil:
                            #         print " ******  "
                            #         print "Substeps: ", iter_count, dt_, rain_prop.mf, snow_prop.mf

                        PV.tendencies[qrain_shift + ijk] = PV.tendencies[qrain_shift + ijk] + (rain_prop.mf - PV.values[qrain_shift + ijk])/TS.dt
                        PV.tendencies[qsnow_shift + ijk] = PV.tendencies[qsnow_shift + ijk] + (snow_prop.mf - PV.values[qsnow_shift + ijk])/TS.dt

                        #Add tendency of qt due to microphysics
                        PV.tendencies[qt_shift + ijk] += (qt_micro - PV.values[qt_shift + ijk])/TS.dt

        #Add entropy tendency due to microphysics (precipitation and evaporation only)
        microphysics_wetbulb_temperature(&Gr.dims, &self.CC.LT.LookupStructC, &Ref.p0_half[0], &PV.values[s_shift],
                                         &PV.values[qt_shift], &DV.values[t_shift], &DV.values[tw_shift])

        get_s_source_precip(&Gr.dims, Th, &Ref.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift],
                            &precip_rate[0], &PV.tendencies[s_shift])
        get_s_source_evap(&Gr.dims, Th, &Ref.p0_half[0], &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qt_shift], &DV.values[qv_shift],
                            &evap_rate[0], &PV.tendencies[s_shift])


        #*************************** Now add sedimentation **************************

        sedimentation_velocity_rain(&Gr.dims, &Ref.rho0_half[0], &DV.values[nrain_shift], &PV.values[qrain_shift],
                                    &DV.values[wqrain_shift])
        sedimentation_velocity_snow(&Gr.dims, &Ref.rho0_half[0], &DV.values[nsnow_shift], &PV.values[qsnow_shift],
                                    &DV.values[wqsnow_shift])

        entropy_source_heating_rain(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qrain_shift],
                                  &DV.values[wqrain_shift],  &PV.values[w_shift], &PV.tendencies[s_shift])

        entropy_source_heating_snow(&Gr.dims, &DV.values[t_shift], &DV.values[tw_shift], &PV.values[qsnow_shift],
                                  &DV.values[wqsnow_shift],  &PV.values[w_shift], &PV.tendencies[s_shift])

        entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qrain_shift], &DV.values[wqrain_shift],
                            &PV.tendencies[s_shift])

        entropy_source_drag(&Gr.dims, &DV.values[t_shift], &PV.values[qsnow_shift], &DV.values[wqsnow_shift],
                            &PV.tendencies[s_shift])

        # #Initialize pencils
        # cdef:
        #     Py_ssize_t nz = Gr.dims.n[2]
        #     Py_ssize_t nlz = Gr.dims.nl[2]
        #     double [:, :] qrain_pencils = self.z_pencil.forward_double(& Gr.dims, Pa, & PV.values[qrain_shift])
        #     double [:, :] qsnow_pencils = self.z_pencil.forward_double(& Gr.dims, Pa, & PV.values[qsnow_shift])
        #     double [:, :] qrain_pencils_ghosted = np.zeros((self.z_pencil.n_local_pencils, nz + 2*kmin), dtype=np.double, order='c')
        #     double [:, :] qsnow_pencils_ghosted = np.zeros((self.z_pencil.n_local_pencils, nz + 2*kmin), dtype=np.double, order='c')
        #
        # #Fill the ghost points
        # with nogil:
        #     for pi in xrange(self.z_pencil.n_local_pencils):
        #         for k in xrange(nz):
        #             qrain_pencils_ghosted[pi, k+kmin] = qrain_pencils[pi, k]
        #             qsnow_pencils_ghosted[pi, k+kmin] = qsnow_pencils[pi, k]
        #
        #         for k in xrange(kmin):
        #             qrain_pencils_ghosted[pi, kmin-1-k] = qrain_pencils_ghosted[pi, kmin+k]
        #             qsnow_pencils_ghosted[pi, kmin-1-k] = qsnow_pencils_ghosted[pi, kmin+k]
        #             qrain_pencils_ghosted[pi, nz+kmin+k] = qrain_pencils_ghosted[pi, nz+kmin-k-1]
        #             qsnow_pencils_ghosted[pi, nz+kmin+k] = qsnow_pencils_ghosted[pi, nz+kmin-k-1]
        #
        # cdef:
        #     double [:] vel_cols_r = np.zeros((nz + 2*kmin), dtype=np.double, order='c')
        #     double [:] vel_cols_s = np.zeros((nz + 2*kmin), dtype=np.double, order='c')
        #
        #     double [:,:] qrain_flux_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
        #     double [:,:] qsnow_flux_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
        #     double [:,:] qrain_tendency_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
        #     double [:,:] qsnow_tendency_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
        #     double [:,:] qrain_vel_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
        #     double [:,:] qsnow_vel_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
        #
        #     double [:] qrain_tmp = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        #     double [:] qsnow_tmp = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        #
        #     double [:] a = np.zeros((nz + 2*kmin), dtype=np.double, order='c')
        #     double [:] a_bar_i = np.zeros((nz + 2*kmin), dtype=np.double, order='c')
        #
        #     double xx, vel_i_m, vel_i_p
        #
        # with nogil:
        #     for pi in xrange(self.z_pencil.n_local_pencils):
        #         time_added = 0.0
        #         iter_count = 0
        #         while time_added < TS.dt:
        #             dt_ = TS.dt - time_added
        #             #First fill/update ghost cells
        #             for k in xrange(kmin):
        #                 qrain_pencils_ghosted[pi, kmin-1-k] = qrain_pencils_ghosted[pi, kmin+k]
        #                 qsnow_pencils_ghosted[pi, kmin-1-k] = qsnow_pencils_ghosted[pi, kmin+k]
        #                 qrain_pencils_ghosted[pi, nz+kmin+k] = qrain_pencils_ghosted[pi, nz+kmin-k-1]
        #                 qsnow_pencils_ghosted[pi, nz+kmin+k] = qsnow_pencils_ghosted[pi, nz+kmin-k-1]
        #             #Then compute velocities
        #             for k in xrange(nz+2*kmin):
        #                 rain_prop.mf = qrain_pencils_ghosted[pi, k]
        #                 vel_cols_r[k] = get_rain_vel_c(Ref.alpha0_half[k], qrain_pencils_ghosted[pi, k], &rain_param, &rain_prop)
        #                 snow_prop.mf = qsnow_pencils_ghosted[pi, k]
        #                 vel_cols_s[k] = get_snow_vel_c(Ref.alpha0_half[k], qsnow_pencils_ghosted[pi, k], &snow_param, &snow_prop)
        #                 dt_ = fmin(dt_, 0.5 * Gr.dims.dx[2] / fmax( fmax(vel_cols_r[k], vel_cols_s[k]), 1.0e-10) )
        #
        #             #Increment the local time variable
        #             time_added += dt_
        #             iter_count += 1
        #
        #
        # #Fill in the velocities (not working!!)
        # with nogil:
        #     for i in xrange(imin,imax):
        #         ishift = i * istride
        #         for j in xrange(jmin,jmax):
        #             jshift = j * jstride
        #             for k in xrange(kmin,kmax):
        #                 ijk = ishift + jshift + k
        #                 DV.values[wqrain_shift+ijk] = interp_2(vel_cols_r[k], vel_cols_r[k+1])
        #                 DV.values[wqsnow_shift+ijk] = interp_2(vel_cols_s[k], vel_cols_s[k+1])


                    #Now do advection (first order upwind scheme)

                    # #First for RAIN
                    # for k in xrange(nz+2*kmin):
                    #     a[k] = qrain_pencils_ghosted[pi, k]
                    #     vel_cols_r[k] = -vel_cols_r[k]
                    #
                    # # for k in xrange(2, nz+2*kmin-2):
                    # #     a_bar_i[k] = a[k+1]
                    #
                    # for k in xrange(kmin, kmax):
                    #     vel_i_m = 0.5*(vel_cols_r[k-1] + vel_cols_r[k])
                    #     vel_i_p = 0.5*(vel_cols_r[k] + vel_cols_r[k+1])
                    #     xx = Gr.dims.dxi[2] * (vel_i_m*a[k]*Ref.rho0[k-1] - vel_i_p*a[k+1]*Ref.rho0[k]) * Ref.alpha0_half[k]
                    #
                    #     qrain_tendency_pencils[pi, k-kmin] += xx*dt_/TS.dt
                    #     qrain_flux_pencils[pi, k-kmin] += Ref.alpha0_half[k]*0.5\
                    #                                     *(vel_i_m*a[k]*Ref.rho0[k-1] +
                    #                                       vel_i_p*a[k+1]*Ref.rho0[k])*dt_/TS.dt
                    #     qrain_pencils_ghosted[pi, k] = fmax(qrain_pencils_ghosted[pi, k] + xx*dt_, 0.0)
                    #     qrain_pencils[pi, k-kmin] = qrain_pencils_ghosted[pi, k]
                    #
                    #     qrain_vel_pencils[pi, k-kmin] = -vel_cols_r[k]
                    #
                    #
                    # #Then for SNOW
                    # for k in xrange(nz+2*kmin):
                    #     a[k] = qsnow_pencils_ghosted[pi, k]
                    #     vel_cols_s[k] = -vel_cols_s[k]
                    #
                    # # for k in xrange(2, nz+2*kmin-2):
                    # #     a_bar_i[k] = a[k+1]
                    #
                    # for k in xrange(kmin, kmax):
                    #     vel_i_m = 0.5*(vel_cols_s[k-1] + vel_cols_s[k])
                    #     vel_i_p = 0.5*(vel_cols_s[k] + vel_cols_s[k+1])
                    #     xx = Gr.dims.dxi[2] * (vel_i_m*a[k]*Ref.rho0[k-1] - vel_i_p*a[k+1]*Ref.rho0[k]) * Ref.alpha0_half[k]
                    #
                    #     qsnow_tendency_pencils[pi, k-kmin] += xx*dt_/TS.dt
                    #     qsnow_flux_pencils[pi, k-kmin] += Ref.alpha0_half[k]*0.5\
                    #                                     *(vel_i_m*a[k]*Ref.rho0[k-1] +
                    #                                       vel_i_p*a[k+1]*Ref.rho0[k])*dt_/TS.dt
                    #     qsnow_pencils_ghosted[pi, k] = fmax(qsnow_pencils_ghosted[pi, k] + xx*dt_, 0.0)
                    #     qsnow_pencils[pi, k-kmin] = qsnow_pencils_ghosted[pi, k]
                    #
                    #     qsnow_vel_pencils[pi, k-kmin] = -vel_cols_s[k]
                    #

                    # if iter_count > 20 and (TS.dt - time_added) > 0.0:
                    #     with gil:
                    #         print " ******  "
                    #         print "Substeps: ", iter_count, (TS.dt - time_added), snow_prop.mf, qsnow_pencils


        # self.z_pencil.reverse_double(&Gr.dims, Pa, qrain_pencils, &qrain_tmp[0])
        # self.z_pencil.reverse_double(&Gr.dims, Pa, qsnow_pencils, &qsnow_tmp[0])
        #
        # cdef:
        #     double rain_dt, snow_dt
        #
        # with nogil:
        #     for i in xrange(imin,imax):
        #         ishift = i * istride
        #         for j in xrange(jmin,jmax):
        #             jshift = j * jstride
        #             for k in xrange(kmin,kmax):
        #                 ijk = ishift + jshift + k
        #
        #                 #Sedimentation only affect precipitation tendencies
        #                 rain_dt = (qrain_tmp[ijk] - PV.values[qrain_shift + ijk])/TS.dt
        #                 PV.tendencies[qrain_shift + ijk] += rain_dt
        #
        #                 snow_dt = (qsnow_tmp[ijk] - PV.values[qsnow_shift + ijk])/TS.dt
        #                 PV.tendencies[qsnow_shift + ijk] += snow_dt
        #
        #                 #For DEBUG: 07/27/2015
        #                 # if fabs(snow_dt) > 1.0e-3:
        #                 #     with gil:
        #                 #         print(i, j, k, TS.dt, qsnow_tmp[ijk], snow_dt, PV.tendencies[qsnow_shift+ijk])
        #                 # #

        # #Now prepare for output
        #
        # self.z_pencil.reverse_double(&Gr.dims, Pa, qrain_flux_pencils, &self.qrain_flux[0])
        # self.z_pencil.reverse_double(&Gr.dims, Pa, qsnow_flux_pencils, &self.qsnow_flux[0])
        # self.z_pencil.reverse_double(&Gr.dims, Pa, qrain_tendency_pencils, &self.qrain_tendency[0])
        # self.z_pencil.reverse_double(&Gr.dims, Pa, qsnow_tendency_pencils, &self.qsnow_tendency[0])
        # self.z_pencil.reverse_double(&Gr.dims, Pa, qrain_vel_pencils, &self.qrain_vel[0])
        # self.z_pencil.reverse_double(&Gr.dims, Pa, qsnow_vel_pencils, &self.qsnow_vel[0])

        # #Get number density for output
        # cdef:
        #     double [:] rain_number = self.rain_number_density
        #     double [:] snow_number = self.snow_number_density
        #     double [:] ice_number = self.ice_number_density
        #     double [:] snow_lambda = self.snow_lambda
        #     double [:] ice_lambda = self.ice_lambda
        #     double [:] n0_snow = self.n0_snow
        #     double [:] n0_ice = self.n0_ice
        #
        # #with nogil:
        # for i in xrange(imin,imax):
        #     ishift = i * istride
        #     for j in xrange(jmin,jmax):
        #         jshift = j * jstride
        #         for k in xrange(kmin,kmax):
        #             ijk = ishift + jshift + k
        #
        #             snow_prop.mf = PV.values[qsnow_shift + ijk]
        #             snow_prop.n0 = get_n0_snow_c(Ref.alpha0_half[k], PV.values[qsnow_shift+ijk], &snow_param)
        #             snow_prop.lam = get_lambda_c(Ref.alpha0_half[k], &snow_prop, &snow_param)
        #             snow_number[ijk] = snow_prop.n0/snow_prop.lam
        #             snow_lambda[ijk] = snow_prop.lam
        #             n0_snow[ijk] = snow_prop.n0
        #
        #             rain_prop.mf = PV.values[qrain_shift + ijk]
        #             rain_prop.n0 = get_n0_rain_c(Ref.alpha0_half[k], PV.values[qrain_shift+ijk], &rain_param)
        #             rain_prop.lam = get_lambda_c(Ref.alpha0_half[k], &rain_prop, &rain_param)
        #             rain_number[ijk] = rain_prop.n0/rain_prop.lam
        #
        #             ice_prop.mf = DV.values[qi_shift + ijk]
        #             ice_prop.n0 = get_n0_ice_c(Ref.alpha0_half[k], DV.values[qi_shift+ijk], self.n0_ice_input, &ice_param)
        #             ice_prop.lam = get_lambda_c(Ref.alpha0_half[k], &ice_prop, &ice_param)
        #             ice_number[ijk] = ice_prop.n0/ice_prop.lam
        #             ice_lambda[ijk] = ice_prop.lam
        #             n0_ice[ijk] = ice_prop.n0



        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t tw_shift = DV.get_varshift(Gr, 'temperature_wb')
            double [:] tmp = np.zeros((Gr.dims.npg), dtype=np.double, order='c')
            double [:] tmp_tendency = np.zeros((Gr.dims.npg), dtype=np.double, order='c')
            Py_ssize_t qrain_shift = PV.get_varshift(Gr, 'qrain')
            Py_ssize_t qsnow_shift = PV.get_varshift(Gr, 'qsnow')
            Py_ssize_t wqrain_shift = DV.get_varshift(Gr, 'w_qrain')
            Py_ssize_t wqsnow_shift = DV.get_varshift(Gr, 'w_qsnow')
            double [:] dummy =  np.zeros((Gr.dims.npg,), dtype=np.double, order='c')



        tmp = Pa.HorizontalMean(Gr, &self.autoconversion[0])
        NS.write_profile('rain_auto_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.autoconversion[Gr.dims.npg])
        NS.write_profile('snow_auto_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.evaporation[0])
        NS.write_profile('rain_evap_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.evaporation[Gr.dims.npg])
        NS.write_profile('snow_depo_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.accretion[0])
        NS.write_profile('rain_accr_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.accretion[Gr.dims.npg])
        NS.write_profile('snow_accr_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.melting[0])
        NS.write_profile('snow_melt_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # tmp = Pa.HorizontalMean(Gr, &self.ice_number_density[0])
        # NS.write_profile('n_ice_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.rain_number_density[0])
        # NS.write_profile('n_rain_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.snow_number_density[0])
        # NS.write_profile('n_snow_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.snow_lambda[0])
        # NS.write_profile('snow_lambda', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.ice_lambda[0])
        # NS.write_profile('ice_lambda', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.n0_snow[0])
        # NS.write_profile('n0_snow', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.n0_ice[0])
        # NS.write_profile('n0_ice', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        #compute sedimentation flux only of qrain and qsnow
        compute_advective_fluxes_a(&Gr.dims, &RS.rho0[0], &RS.rho0_half[0], &DV.values[wqrain_shift], &PV.values[qrain_shift], &dummy[0], 2, self.order)
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('rain_sedimentation_flux', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        compute_advective_fluxes_a(&Gr.dims, &RS.rho0[0], &RS.rho0_half[0], &DV.values[wqsnow_shift], &PV.values[qsnow_shift], &dummy[0], 2, self.order)
        tmp = Pa.HorizontalMean(Gr, &dummy[0])
        NS.write_profile('snow_sedimentation_flux', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # tmp = Pa.HorizontalMean(Gr, &self.qrain_flux[0])
        # NS.write_profile('rain_sedimentation_flux', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.qrain_tendency[0])
        # NS.write_profile('rain_sedimentation_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.qrain_vel[0])
        # NS.write_profile('rain_sedimentation_velocity', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.qsnow_flux[0])
        # NS.write_profile('snow_sedimentation_flux', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.qsnow_tendency[0])
        # NS.write_profile('snow_sedimentation_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # tmp = Pa.HorizontalMean(Gr, &self.qsnow_vel[0])
        # NS.write_profile('snow_sedimentation_velocity', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

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
            Py_ssize_t qrain_shift = PV.get_varshift(Gr, 'qrain')
            Py_ssize_t qsnow_shift = PV.get_varshift(Gr, 'qsnow')
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
