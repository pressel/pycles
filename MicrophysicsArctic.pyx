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

from libc.math cimport fmax

include 'micro_parameters.pxi'

cdef extern from "microphysics_functions.h":
    void micro_substep_c(Lookup.LookupStruct *LT, double alpha, double vapor_star, double T, double cnn, double n0_ice,
                         hm_parameters *rain_param, hm_parameters *snow_param, hm_parameters *liquid_param, hm_parameters *ice_param,
                         hm_properties *rain_prop, hm_properties *snow_prop, hm_properties *liquid_prop, hm_properties *ice_prop,
                         double* aut_rain, double* aut_snow, ret_acc *src_acc, double* evp_rain,
                         double* evp_snow, double* melt_snow) nogil



cdef class MicrophysicsArctic:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):

        LH.Lambda_fp = lambda_Arctic
        LH.L_fp = latent_heat_Arctic

        self.thermodynamics_type = 'SA'
        print(self.thermodynamics_type)

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

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        #Define all source terms that need to be stored
        #Ghosted or not???
        self.autoconversion = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.evaporation = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.accretion = np.zeros(Gr.dims.npg*2, dtype=np.double, order='c')
        self.melting = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.liquid_fraction = np.zeros(Gr.dims.npg, dtype=np.double, order='c')

        #Add precipitation variables
        PV.add_variable('qrain', 'kg kg^-1', "sym", "scalar", Pa)
        PV.add_variable('qsnow', 'kg kg^-1', "sym", "scalar", Pa)

        #Initialize Statistical Output
        NS.add_profile('n_rain_mean', Gr, Pa)
        NS.add_profile('n_rain_mean2', Gr, Pa)
        NS.add_profile('n_snow_mean', Gr, Pa)
        NS.add_profile('n_snow_mean2', Gr, Pa)
        NS.add_profile('n_ice_mean', Gr, Pa)
        NS.add_profile('n_ice_mean2', Gr, Pa)

        NS.add_profile('rain_auto_mass', Gr, Pa)
        NS.add_profile('snow_auto_mass', Gr, Pa)
        NS.add_profile('rain_accr_mass', Gr, Pa)
        NS.add_profile('snow_accr_mass', Gr, Pa)
        NS.add_profile('rain_evap_mass', Gr, Pa)
        NS.add_profile('snow_depo_mass', Gr, Pa)
        NS.add_profile('snow_melt_mass', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        #Get parameters
        cdef:
            #struct pointers???
            # hm_properties rain_prop
            # hm_properties snow_prop
            # hm_properties ice_prop
            # hm_properties liquid_prop
            #
            # ret_acc src_acc

            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
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
            double [:] lf = self.liquid_fraction
            double aut_rain, aut_snow, evp_rain, evp_snow, melt_snow
            double vapor_star

            # double iter_count, time_added, dt_, rate
            double machine_eps = np.finfo(np.float64).eps


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
                        self.rain_prop.mf = PV.values[qrain_shift + ijk]
                        self.snow_prop.mf = PV.values[qsnow_shift + ijk]
                        self.liquid_prop.mf = DV.values[ql_shift + ijk]
                        self.ice_prop.mf = DV.values[qi_shift + ijk]
                        vapor_star = DV.values[qv_shift + ijk]

                        # #Do substepping (iterations < 20)
                        #
                        # iter_count = 0
                        # time_added = 0.0
                        #
                        # while time_added < TS.dt and iter_count < 20:
                        #     if (liquid_prop.mf+ice_prop.mf) < small and (rain_prop.mf+snow_prop.mf) < small:
                        #         break

                        micro_substep_c(&self.CC.LT.LookupStructC, DV.values[alpha_shift + ijk], DV.values[qv_shift + ijk], DV.values[t_shift + ijk],
                                        self.ccn, self.n0_ice, &rain_param, &snow_param, &liquid_param, &ice_param,
                                        &self.rain_prop, &self.snow_prop, &self.liquid_prop, &self.ice_prop, &aut_rain, &aut_snow,
                                        &self.src_acc, &evp_rain, &evp_snow, &melt_snow)

                        # dt_ = TS.dt - time_added
                        # rate1 = 1.05 * (aut_rain + src_acc.dyr +  evp_rain - melt_snow)/(-(rain_prop.mf+machine_eps)/dt_)
                        # rate2 = 1.05 * (aut_snow + src_acc.dys +  evp_snow + melt_snow)/(-(snow_prop.mf+machine_eps)/dt_)
                        # rate3 = 1.05 * (-aut_rain + src_acc.dyl)/(-(liquid_prop.mf+machine_eps)/dt_)
                        # rate4 = 1.05 * (-aut_snow + src_acc.dyi)/(-(ice_prop.mf+machine_eps)/dt_)
                        #
                        # rate = fmax(fmax(fmax(rate1,rate2),rate3),rate4)
                        # if rate > 1.0:
                        #     # Limit the timestep, but don't let it get too small
                        #     dt_ = fmax(dt_/rate,1e-10)


                        # Integrate forward in time
                        self.rain_prop.mf = fmax(self.rain_prop.mf + (aut_rain + self.src_acc.dyr + evp_rain - melt_snow)* TS.dt,0.0)
                        self.snow_prop.mf = fmax(self.snow_prop.mf + (aut_snow + self.src_acc.dys + evp_snow + melt_snow)* TS.dt,0.0)
                        self.liquid_prop.mf = fmax(self.liquid_prop.mf + (-aut_rain + self.src_acc.dyl) * TS.dt,0.0)
                        self.ice_prop.mf = fmax(self.ice_prop.mf + (-aut_snow + self.src_acc.dyi) * TS.dt,0.0)
                        vapor_star = fmax(vapor_star + (-evp_rain - evp_snow) * TS.dt,0.0)


                        # Update the contributions of each source term
                        aut[ijk] = aut[ijk] + aut_rain
                        acc[ijk] = acc[ijk] + self.src_acc.dyr
                        evp[ijk] = evp[ijk] + evp_rain
                        aut[ijk+allshift] = aut[ijk+allshift] + aut_snow
                        acc[ijk+allshift] = acc[ijk+allshift] + self.src_acc.dys
                        evp[ijk+allshift] = evp[ijk+allshift] + evp_snow
                        melt[ijk] = melt[ijk] + melt_snow
                        # # Increment the local time variables
                        # time_added = time_added + dt_
                        # iter_count = iter_count + 1

                        # if iter_count > 19:
                        #     with gil:
                        #         print " ******  "
                        #         print "Substeps: ", iter_count, dt_, rain_prop.mf, snow_prop.mf

                        PV.tendencies[qrain_shift + ijk] = PV.tendencies[qrain_shift + ijk] + (self.rain_prop.mf - PV.values[qrain_shift + ijk])/TS.dt
                        PV.tendencies[qsnow_shift + ijk] = PV.tendencies[qsnow_shift + ijk] + (self.snow_prop.mf - PV.values[qsnow_shift + ijk])/TS.dt

                        #Get rain properties

                        #Get snow properties

                        #Get liquid properties

                        #Get ice properties

                        #Do autoconversion

                        #Do accretion

                        #Do evaporation/sublimation

                        #Do melting

                        #get the next sub-timestep by determining the max source tendency term

                        #Integrate forward in time

                        #Update the contributions of each source term (sub-timestep weighted)

                        #Increment local time variables (if iter > 19, print out)

                        #Update precip PV tendencies

                        #Loop inside all source functions (?)

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
        NS.write_profile('snow_evap_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.accretion[0])
        NS.write_profile('rain_accr_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.accretion[Gr.dims.npg])
        NS.write_profile('snow_accr_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.melting[0])
        NS.write_profile('snow_melt_mass', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)



        return