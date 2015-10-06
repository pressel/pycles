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

cdef extern from "microphysics_functions.h":
    void micro_substep_c(Lookup.LookupStruct *LT, double alpha, double p0, double qt, double T, double cnn, double n0_ice,
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
    inline double get_wet_bulb_c(double T) nogil
    inline double entropy_src_precipitation_c(double p0, double T, double qt, double qv,
                                              double L, double precip_rate) nogil
    inline double entropy_src_evaporation_c(double p0, double T, double Tw, double qt, double qv,
                                            double L, double evap_rate) nogil

cdef class MicrophysicsArctic:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):

        LH.Lambda_fp = lambda_Arctic
        LH.L_fp = latent_heat_Arctic

        self.thermodynamics_type = 'SA'

        #Get namelist variables
        try:
            self.ccn = namelist['microphysics']['ccn']
        except:
            self.ccn = 100.0e6

        try:
            self.n0_ice = namelist['microphysics']['n0_ice']
        except:
            self.n0_ice = 1.0e7

        # self.L_fp = LH.L_fp
        # self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Par)

        self.z_pencil = ParallelMPI.Pencil()

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.z_pencil.initialize(Gr, Pa, 2)

        #Define all source terms that need to be stored
        #Ghosted or not???
        self.autoconversion = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.evaporation = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.accretion = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.melting = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        self.qrain_flux = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.qsnow_flux = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.qrain_tendency = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.qsnow_tendency = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.qrain_vel = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.qsnow_vel = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        self.rain_number_density = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.snow_number_density = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.ice_number_density = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        #Add precipitation variables
        PV.add_variable('qrain', 'kg kg^-1', "sym", "scalar", Pa)
        PV.add_variable('qsnow', 'kg kg^-1', "sym", "scalar", Pa)

        #Initialize Statistical Output
        NS.add_profile('n_rain_mean', Gr, Pa)
        # NS.add_profile('n_rain_mean2', Gr, Pa)
        NS.add_profile('n_snow_mean', Gr, Pa)
        # NS.add_profile('n_snow_mean2', Gr, Pa)
        NS.add_profile('n_ice_mean', Gr, Pa)
        # NS.add_profile('n_ice_mean2', Gr, Pa)

        NS.add_profile('rain_auto_mass', Gr, Pa)
        NS.add_profile('snow_auto_mass', Gr, Pa)
        NS.add_profile('rain_accr_mass', Gr, Pa)
        NS.add_profile('snow_accr_mass', Gr, Pa)
        NS.add_profile('rain_evap_mass', Gr, Pa)
        NS.add_profile('snow_depo_mass', Gr, Pa)
        NS.add_profile('snow_melt_mass', Gr, Pa)

        NS.add_profile('rain_sedimentation_velocity', Gr, Pa)
        NS.add_profile('snow_sedimentation_velocity', Gr, Pa)
        NS.add_profile('rain_sedimentation_flux', Gr, Pa)
        NS.add_profile('snow_sedimentation_flux', Gr, Pa)
        NS.add_profile('rain_sedimentation_tendency', Gr, Pa)
        NS.add_profile('snow_sedimentation_tendency', Gr, Pa)


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

            Py_ssize_t alpha_shift = DV.get_varshift(Gr, 'alpha')
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t qrain_shift = PV.get_varshift(Gr, 'qrain')
            Py_ssize_t qsnow_shift = PV.get_varshift(Gr, 'qsnow')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')

            double [:] aut = self.autoconversion
            double [:] evp = self.evaporation
            double [:] acc = self.accretion
            double [:] melt = self.melting

            double aut_rain, aut_snow, evp_rain, evp_snow, melt_snow
            double qt_micro

            double iter_count, time_added, dt_, rate, rate1, rate2, rate3, rate4
            double machine_eps = np.finfo(np.float64).eps
            double Tw, L, precip_rate, evap_rate

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
                        #Get liquid fraction ???
                        # lf[ijk] = Th.compute_liquid_fraction_c()
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

                            micro_substep_c(&self.CC.LT.LookupStructC, Ref.alpha0_half[k], Ref.p0_half[k], qt_micro, DV.values[t_shift + ijk],
                                            self.ccn, self.n0_ice, &rain_param, &snow_param, &liquid_param, &ice_param,
                                            &rain_prop, &snow_prop, &liquid_prop, &ice_prop, &aut_rain, &aut_snow,
                                            &src_acc, &evp_rain, &evp_snow, &melt_snow)

                            dt_ = TS.dt - time_added
                            rate1 = 1.05 * (aut_rain + src_acc.dyr +  evp_rain - melt_snow)/(-(rain_prop.mf+machine_eps)/dt_)
                            rate2 = 1.05 * (aut_snow + src_acc.dys +  evp_snow + melt_snow)/(-(snow_prop.mf+machine_eps)/dt_)
                            rate3 = 1.05 * (-aut_rain + src_acc.dyl)/(-(liquid_prop.mf+machine_eps)/dt_)
                            rate4 = 1.05 * (-aut_snow + src_acc.dyi)/(-(ice_prop.mf+machine_eps)/dt_)

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

                            precip_rate = -aut_rain + src_acc.dyl - aut_snow + src_acc.dyi
                            evap_rate = evp_rain + evp_snow
                            qt_micro = fmax(qt_micro + (precip_rate - evap_rate)*dt_, 0.0)
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

        #Get entropy tendency
        for i in xrange(imin,imax):
            ishift = i * istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    L = Th.get_lh(DV.values[t_shift + ijk])
                    Tw = get_wet_bulb_c(DV.values[t_shift + ijk])
                    PV.tendencies[s_shift + ijk] += entropy_src_precipitation_c(Ref.p0_half[k], DV.values[t_shift + ijk],
                                                    PV.values[qt_shift + ijk], DV.values[qv_shift + ijk], L, precip_rate) + \
                                                    entropy_src_evaporation_c(Ref.p0_half[k], DV.values[t_shift + ijk], Tw,
                                                    PV.values[qt_shift + ijk], DV.values[qv_shift + ijk], L, evap_rate)

        #*************************** Now add sedimentation **************************

        #Initialize pencils
        cdef:
            Py_ssize_t nz = Gr.dims.n[2]
            Py_ssize_t nlz = Gr.dims.nl[2]
            double [:, :] qrain_pencils = self.z_pencil.forward_double(& Gr.dims, Pa, & PV.values[qrain_shift])
            double [:, :] qsnow_pencils = self.z_pencil.forward_double(& Gr.dims, Pa, & PV.values[qsnow_shift])
            double [:, :] qrain_pencils_ghosted = np.zeros((self.z_pencil.n_local_pencils, nz + 2*kmin), dtype=np.double, order='c')
            double [:, :] qsnow_pencils_ghosted = np.zeros((self.z_pencil.n_local_pencils, nz + 2*kmin), dtype=np.double, order='c')

        #Fill the ghost points
        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):
                for k in xrange(nz):
                    qrain_pencils_ghosted[pi, k+kmin] = qrain_pencils[pi, k]
                    qsnow_pencils_ghosted[pi, k+kmin] = qsnow_pencils[pi, k]

                for k in xrange(kmin):
                    qrain_pencils_ghosted[pi, kmin-1-k] = qrain_pencils_ghosted[pi, kmin+k]
                    qsnow_pencils_ghosted[pi, kmin-1-k] = qsnow_pencils_ghosted[pi, kmin+k]
                    qrain_pencils_ghosted[pi, nz+kmin+k] = qrain_pencils_ghosted[pi, nz+kmin-k-1]
                    qsnow_pencils_ghosted[pi, nz+kmin+k] = qsnow_pencils_ghosted[pi, nz+kmin-k-1]

        cdef:
            double [:] vel_cols_r = np.zeros((nz + 2*kmin), dtype=np.double, order='c')
            double [:] vel_cols_s = np.zeros((nz + 2*kmin), dtype=np.double, order='c')

            double [:,:] qrain_flux_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
            double [:,:] qsnow_flux_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
            double [:,:] qrain_tendency_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
            double [:,:] qsnow_tendency_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
            double [:,:] qrain_vel_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')
            double [:,:] qsnow_vel_pencils = np.zeros((self.z_pencil.n_local_pencils, nz),dtype=np.double, order='c')

            double [:] qrain_tmp = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
            double [:] qsnow_tmp = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

            double [:] a = np.zeros((nz + 2*kmin), dtype=np.double, order='c')
            double [:] a_bar_i = np.zeros((nz + 2*kmin), dtype=np.double, order='c')

            double xx, vel_i_m, vel_i_p

        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):
                time_added = 0.0
                iter_count = 0
                while time_added < TS.dt:
                    dt_ = TS.dt - time_added
                    #First fill/update ghost cells
                    for k in xrange(kmin):
                        qrain_pencils_ghosted[pi, kmin-1-k] = qrain_pencils_ghosted[pi, kmin+k]
                        qsnow_pencils_ghosted[pi, kmin-1-k] = qsnow_pencils_ghosted[pi, kmin+k]
                        qrain_pencils_ghosted[pi, nz+kmin+k] = qrain_pencils_ghosted[pi, nz+kmin-k-1]
                        qsnow_pencils_ghosted[pi, nz+kmin+k] = qsnow_pencils_ghosted[pi, nz+kmin-k-1]
                    #Then compute velocities
                    for k in xrange(nz+2*kmin):
                        rain_prop.mf = qrain_pencils_ghosted[pi, k]
                        vel_cols_r[k] = get_rain_vel_c(Ref.alpha0_half[k], qrain_pencils_ghosted[pi, k], &rain_param, &rain_prop)
                        snow_prop.mf = qsnow_pencils_ghosted[pi, k]
                        vel_cols_s[k] = get_snow_vel_c(Ref.alpha0_half[k], qsnow_pencils_ghosted[pi, k], &snow_param, &snow_prop)
                        dt_ = fmin(dt_, 0.5 * Gr.dims.dx[2] / fmax( fmax(vel_cols_r[k], vel_cols_s[k]), 1.0e-10) )

                    #Now do advection (first order upwind scheme)

                    #First for RAIN
                    for k in xrange(nz+2*kmin):
                        a[k] = qrain_pencils_ghosted[pi, k]
                        vel_cols_r[k] = -vel_cols_r[k]

                    # for k in xrange(2, nz+2*kmin-2):
                    #     a_bar_i[k] = a[k+1]

                    for k in xrange(kmin, kmax):
                        vel_i_m = 0.5*(vel_cols_r[k-1] + vel_cols_r[k])
                        vel_i_p = 0.5*(vel_cols_r[k] + vel_cols_r[k+1])
                        xx = Gr.dims.dxi[2] * (vel_i_m*a[k]*Ref.rho0[k-1] - vel_i_p*a[k+1]*Ref.rho0[k]) * Ref.alpha0_half[k]

                        qrain_tendency_pencils[pi, k-kmin] += xx*dt_/TS.dt
                        qrain_flux_pencils[pi, k-kmin] += Ref.alpha0_half[k]*0.5\
                                                        *(vel_i_m*a[k]*Ref.rho0[k-1] +
                                                          vel_i_p*a[k+1]*Ref.rho0[k])*dt_/TS.dt
                        qrain_pencils_ghosted[pi, k] = fmax(qrain_pencils_ghosted[pi, k] + xx*dt_, 0.0)
                        qrain_pencils[pi, k-kmin] = qrain_pencils_ghosted[pi, k]

                        qrain_vel_pencils[pi, k-kmin] = -vel_cols_r[k]


                    #Then for SNOW
                    for k in xrange(nz+2*kmin):
                        a[k] = qsnow_pencils_ghosted[pi, k]
                        vel_cols_s[k] = -vel_cols_s[k]

                    # for k in xrange(2, nz+2*kmin-2):
                    #     a_bar_i[k] = a[k+1]

                    for k in xrange(kmin, kmax):
                        vel_i_m = 0.5*(vel_cols_s[k-1] + vel_cols_s[k])
                        vel_i_p = 0.5*(vel_cols_s[k] + vel_cols_s[k+1])
                        xx = Gr.dims.dxi[2] * (vel_i_m*a[k]*Ref.rho0[k-1] - vel_i_p*a[k+1]*Ref.rho0[k]) * Ref.alpha0_half[k]

                        qsnow_tendency_pencils[pi, k-kmin] += xx*dt_/TS.dt
                        qsnow_flux_pencils[pi, k-kmin] += Ref.alpha0_half[k]*0.5\
                                                        *(vel_i_m*a[k]*Ref.rho0[k-1] +
                                                          vel_i_p*a[k+1]*Ref.rho0[k])*dt_/TS.dt
                        qsnow_pencils_ghosted[pi, k] = fmax(qsnow_pencils_ghosted[pi, k] + xx*dt_, 0.0)
                        qsnow_pencils[pi, k-kmin] = qsnow_pencils_ghosted[pi, k]

                        qsnow_vel_pencils[pi, k-kmin] = -vel_cols_s[k]

                    #Increment the local time variable
                    time_added += dt_
                    iter_count += 1

                    # if iter_count > 20 and (TS.dt - time_added) > 0.0:
                    #     with gil:
                    #         print " ******  "
                    #         print "Substeps: ", iter_count, (TS.dt - time_added), snow_prop.mf, qsnow_pencils


        self.z_pencil.reverse_double(&Gr.dims, Pa, qrain_pencils, &qrain_tmp[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, qsnow_pencils, &qsnow_tmp[0])

        cdef:
            double rain_dt, snow_dt

        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k

                        #Sedimentation only affect precipitation tendencies
                        rain_dt = (qrain_tmp[ijk] - PV.values[qrain_shift + ijk])/TS.dt
                        PV.tendencies[qrain_shift + ijk] += rain_dt

                        snow_dt = (qsnow_tmp[ijk] - PV.values[qsnow_shift + ijk])/TS.dt
                        PV.tendencies[qsnow_shift + ijk] += snow_dt

                        #For DEBUG: 07/27/2015
                        # if fabs(snow_dt) > 1.0e-3:
                        #     with gil:
                        #         print(i, j, k, TS.dt, qsnow_tmp[ijk], snow_dt, PV.tendencies[qsnow_shift+ijk])
                        # #

        #Now prepare for output

        self.z_pencil.reverse_double(&Gr.dims, Pa, qrain_flux_pencils, &self.qrain_flux[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, qsnow_flux_pencils, &self.qsnow_flux[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, qrain_tendency_pencils, &self.qrain_tendency[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, qsnow_tendency_pencils, &self.qsnow_tendency[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, qrain_vel_pencils, &self.qrain_vel[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, qsnow_vel_pencils, &self.qsnow_vel[0])

        #Get number density for output
        cdef:
            double [:] rain_number = self.rain_number_density
            double [:] snow_number = self.snow_number_density
            double [:] ice_number = self.ice_number_density

        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        snow_prop.n0 = get_n0_snow_c(Ref.alpha0_half[k], PV.values[qsnow_shift+ijk], &snow_param)
                        #snow_prop.lam = get_lambda_c(Ref.alpha0_half[k], &snow_prop, &snow_param)
                        snow_number[ijk] = snow_prop.n0#/snow_prop.lam

                        rain_prop.n0 = get_n0_rain_c(Ref.alpha0_half[k], PV.values[qrain_shift+ijk], &rain_param)
                        #rain_prop.lam = get_lambda_c(Ref.alpha0_half[k], &rain_prop, &rain_param)
                        rain_number[ijk] = rain_prop.n0#/rain_prop.lam

                        ice_prop.n0 = get_n0_ice_c(Ref.alpha0_half[k], DV.values[qi_shift+ijk], self.n0_ice, &ice_param)
                        #ice_prop.lam = get_lambda_c(Ref.alpha0_half[k], &ice_prop, &ice_param)
                        ice_number[ijk] = ice_prop.n0#/ice_prop.lam




        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            double [:] tmp = np.zeros((Gr.dims.npg), dtype=np.double, order='c')


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

        tmp = Pa.HorizontalMean(Gr, &self.ice_number_density[0])
        NS.write_profile('n_ice_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.snow_number_density[0])
        NS.write_profile('n_snow_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.qrain_flux[0])
        NS.write_profile('rain_sedimentation_flux', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.qrain_tendency[0])
        NS.write_profile('rain_sedimentation_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.qrain_vel[0])
        NS.write_profile('rain_sedimentation_velocity', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.qsnow_flux[0])
        NS.write_profile('snow_sedimentation_flux', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.qsnow_tendency[0])
        NS.write_profile('snow_sedimentation_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.qsnow_vel[0])
        NS.write_profile('snow_sedimentation_velocity', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        return