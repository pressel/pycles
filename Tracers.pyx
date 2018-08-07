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

from NetCDFIO cimport NetCDFIO_Stats
import cython

from libc.math cimport fmax, fmin, sqrt, copysign

cimport numpy as np
import numpy as np
include "parameters.pxi"

import cython

cdef extern from "thermodynamic_functions.h":
    inline double pv_c(double p0, double qt, double qv) nogil


def TracersFactory(namelist):
    try:
        use_tracers = namelist['tracers']['use_tracers']
    except:
        use_tracers = False
    if use_tracers:
        try:
            tracer_scheme = namelist['tracers']['scheme']
            if tracer_scheme == 'UpdraftTracers':
                return UpdraftTracers(namelist)
            elif tracer_scheme == 'PurityTracers':
                return PurityTracers(namelist)
            else:
                print('Tracer scheme is not recognized, using TracersNone')
                return  TracersFactory()
        except:
            tracer_scheme = 'UpdraftTracers'
            return UpdraftTracers(namelist)
    else:
        return TracersNone()


cdef class TracersNone:
    def __init__(self):
        return
    cpdef initialize(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update_cleanup(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        return
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class UpdraftTracers:

    def __init__(self, namelist):
        try:
            self.lcl_tracers = namelist['tracers']['use_lcl_tracers']
        except:
            self.lcl_tracers = False

        self.index_lcl = 0

        return

    cpdef initialize(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.updraft_indicator = np.zeros((Gr.dims.npg),dtype=np.double, order='c')

        # Assemble a dictionary with the tracer information
        # Can be expanded for different init heights or timescales
        self.tracer_dict = {}
        self.tracer_dict['surface'] = {}
        self.tracer_dict['surface']['c_srf_15'] = {}
        self.tracer_dict['surface']['c_srf_15']['timescale'] = 15.0 * 60.0
        if self.lcl_tracers:
            self.tracer_dict['lcl'] = {}
            self.tracer_dict['lcl']['c_lcl_15'] = {}
            self.tracer_dict['lcl']['c_lcl_15']['timescale'] = 15.0 * 60.0

        for var in self.tracer_dict['surface'].keys():
           PV.add_variable(var, '-', var, 'tracer diagnostics' , "sym", "scalar", Pa)

        if self.lcl_tracers:
            for var in self.tracer_dict['lcl'].keys():
                PV.add_variable(var, '-', var, 'tracer diagnostics', "sym", "scalar", Pa)
            NS.add_ts('grid_lcl', Gr, Pa )


        NS.add_profile('updraft_fraction', Gr, Pa, units=r'-',nice_name=r'a_u',
                       desc=r'updraft area fraction' )
        NS.add_profile('updraft_w', Gr, Pa, units=r'm s^{-1}', nice_name=r'w_u',
                       desc=r'updraft vertical velocity at cell centers')
        NS.add_profile('updraft_w2', Gr, Pa, units=r'm^{2} s^{-2}', nice_name=r'w^2_u',
                       desc=r'updraft vertical velocity square at cell centers')
        NS.add_profile('updraft_u', Gr, Pa, units=r'm s^{-1}', nice_name=r'u_u',
                       desc=r'updraft u velocity')
        NS.add_profile('updraft_u2', Gr, Pa, units=r'm^{2} s^{-2}', nice_name=r'u^2_u',
                       desc=r'updraft u velocity square at cell centers')
        NS.add_profile('updraft_v', Gr, Pa, units=r'm s^{-1}', nice_name=r'v_u',
                       desc=r'updraft v velocity')
        NS.add_profile('updraft_v2', Gr, Pa, units=r'm^{2} s^{-2}', nice_name=r'v^2_u',
                       desc=r'updraft v velocity square at cell centers')
        NS.add_profile('updraft_qt', Gr, Pa, units=r'kg kg^{-1}',nice_name=r'q_{t,u}',
                       desc=r'updraft total water specific humidity')
        NS.add_profile('updraft_qt2', Gr, Pa, units=r'kg^{2} kg^{-2}',nice_name=r'q_{t,u}^2',
                       desc=r'updraft total water specific humidity square')
        NS.add_profile('updraft_thetali', Gr, Pa, units=r'K', nice_name=r'\theta_{li,u}',
                       desc=r'updraft liquid ice potential temperature')
        NS.add_profile('updraft_thetali2', Gr, Pa, units=r'K^2', nice_name=r'\theta_{li,u}^2',
                       desc=r'updraft liquid ice potential temperature square')
        NS.add_profile('updraft_b', Gr, Pa, units=r'm s^{-2}', nice_name=r'b_{u}',
                       desc=r'updraft buoyancy (not anomaly)')
        NS.add_profile('updraft_b2', Gr, Pa, units=r'm s^{-2}', nice_name=r'b_{u}^2',
                       desc=r'updraft buoyancy (not anomaly) square')
        NS.add_profile('updraft_w_qt', Gr, Pa, units=r' m s^{-1} kg kg^{-1}', nice_name=r'(w q_t)_u',
                       desc=r'updraft product of w and q_t')
        NS.add_profile('updraft_w_thetali', Gr, Pa, units=r' m s^{-1} K', nice_name=r'(w \theta_{li})_u',
                       desc=r'updraft product of w and \theta_{li}')
        NS.add_profile('updraft_qt_thetali', Gr, Pa, units=r'kg kg^{-1} K', nice_name=r'(q_t \theta_{li})_u',
                       desc=r'updraft product of q_t and \theta_{li}')



        NS.add_profile('updraft_bvf', Gr, Pa, units=r's^{-1}', nice_name=r'N_{b,u}',
                       desc=r'updraft buoyancy frequency')
        NS.add_profile('updraft_thetarho', Gr, Pa, units=r'K', nice_name=r'\theta_{\rho,u}',
                       desc=r'updraft density potential temperature')

        NS.add_profile('updraft_temperature', Gr, Pa, units=r'K', nice_name=r'T_{u}',
                       desc=r'updraft temperature')
        NS.add_profile('updraft_temperature2', Gr, Pa, units=r'K^2', nice_name=r'T_{u}^2',
                       desc=r'updraft temperature square')
        NS.add_profile('updraft_qv', Gr, Pa, units=r'kg kg^{-1}',nice_name=r'q_{v,u}',
                       desc=r'updraft vapor specific humidity')
        NS.add_profile('updraft_qv2', Gr, Pa, units=r'kg^{2} kg^{-2}',nice_name=r'q_{v,u}^2',
                       desc=r'updraft vapor specific humidity square')
        NS.add_profile('updraft_cloudfraction', Gr, Pa, units=r'--', nice_name= r'f_{c,u}',
                       desc=r'updraft cloud fraction')


        NS.add_profile('env_fraction', Gr, Pa, units=r'-',nice_name=r'a_e',
                       desc=r'environment area fraction' )
        NS.add_profile('env_w', Gr, Pa, units=r'm s^{-1}', nice_name=r'w_e',
                       desc=r'environment vertical velocity at cell centers')
        NS.add_profile('env_w2', Gr, Pa, units=r'm^{2} s^{-2}', nice_name=r'w^2_e',
                       desc=r'environment vertical velocity square at cell centers')
        NS.add_profile('env_u', Gr, Pa, units=r'm s^{-1}', nice_name=r'u_e',
                       desc=r'environment u velocity')
        NS.add_profile('env_u2', Gr, Pa, units=r'm^{2} s^{-2}', nice_name=r'u^2_e',
                       desc=r'environment u velocity square at cell centers')
        NS.add_profile('env_v', Gr, Pa, units=r'm s^{-1}', nice_name=r'v_e',
                       desc=r'environment v velocity')
        NS.add_profile('env_v2', Gr, Pa, units=r'm^{2} s^{-2}', nice_name=r'v^2_e',
                       desc=r'environment v velocity square at cell centers')
        NS.add_profile('env_qt', Gr, Pa, units=r'kg kg^{-1}',nice_name=r'q_{t,e}',
                       desc=r'environment total water specific humidity')
        NS.add_profile('env_qt2', Gr, Pa, units=r'kg^{2} kg^{-2}',nice_name=r'q_{t,e}^2',
                       desc=r'environment total water specific humidity square')
        NS.add_profile('env_thetali', Gr, Pa, units=r'K', nice_name=r'\theta_{li,e}',
                       desc=r'environment liquid ice potential temperature')
        NS.add_profile('env_thetali2', Gr, Pa, units=r'K^2', nice_name=r'\theta_{li,e}^2',
                       desc=r'environment liquid ice potential temperature square')
        NS.add_profile('env_b', Gr, Pa, units=r'm s^{-2}', nice_name=r'b_{e}',
                       desc=r'environment buoyancy (not anomaly)')
        NS.add_profile('env_b2', Gr, Pa, units=r'm s^{-2}', nice_name=r'b_{e}^2',
                       desc=r'environment buoyancy (not anomaly) square')
        NS.add_profile('env_w_qt', Gr, Pa, units=r' m s^{-1} kg kg^{-1}', nice_name=r'(w q_t)_e',
                       desc=r'environmnet product of w and q_t')
        NS.add_profile('env_w_thetali', Gr, Pa, units=r' m s^{-1} K', nice_name=r'(w \theta_{li})_e',
                       desc=r'environment product of w and \theta_{li}')
        NS.add_profile('env_qt_thetali', Gr, Pa, units=r'kg kg^{-1} K', nice_name=r'(q_t \theta_{li})_e',
                       desc=r'environment product of q_t and \theta_{li}')

        NS.add_profile('env_bvf', Gr, Pa, units=r's^{-1}', nice_name=r'N_{b,e}',
                       desc=r'environment buoyancy frequency')


        NS.add_profile('env_temperature', Gr, Pa, units=r'K', nice_name=r'T_{e}',
                       desc=r'environment temperature')
        NS.add_profile('env_temperature2', Gr, Pa, units=r'K^2', nice_name=r'T_{e}^2',
                       desc=r'environment temperature square')
        NS.add_profile('env_qv', Gr, Pa, units=r'kg kg^{-1}',nice_name=r'q_{v,e}',
                       desc=r'environment vapor specific humidity')
        NS.add_profile('env_qv2', Gr, Pa, units=r'kg^{2} kg^{-2}',nice_name=r'q_{v,e}^2',
                       desc=r'environment vapor specific humidity square')

        NS.add_profile('env_cloudfraction', Gr, Pa, units=r'--', nice_name= r'f_{c,e}',
                       desc=r'environment cloud fraction')
        NS.add_profile('env_thetarho', Gr, Pa, units=r'K', nice_name=r'\theta_{\rho,e}',
                       desc=r'environment density potential temperature')


        if 'ql' in DV.name_index:
            NS.add_profile('updraft_ql', Gr, Pa, units=r'kg kg^{-1}',nice_name=r'q_{l,u}',
                       desc=r'updraft liquid water specific humidity')
            NS.add_profile('updraft_ql2', Gr, Pa, units=r'kg^2 kg^{-2}',nice_name=r'q_{t,u}^2',
                       desc=r'updraft total water specific humidity square')
            NS.add_profile('env_ql', Gr, Pa, units=r'kg kg^{-1}',nice_name=r'q_{l,e}',
                       desc=r'environment liquid water specific humidity')
            NS.add_profile('env_ql2', Gr, Pa, units=r'kg^2 kg^{-2}',nice_name=r'q_{l,u}^2',
                       desc=r'environment total water specific humidity square')
        if 'qr' in PV.name_index:
            NS.add_profile('updraft_qr', Gr, Pa, units=r'kg kg^{-1}',nice_name=r'q_{r,u}',
                       desc=r'updraft rain water specific humidity')
            NS.add_profile('updraft_qr2', Gr, Pa, units=r'kg^2 kg^{-2}',nice_name=r'q_{r,u}^2',
                       desc=r'updraft rain water specific humidity square')
            NS.add_profile('env_qr', Gr, Pa, units=r'kg kg^{-1}',nice_name=r'q_{r,e}',
                       desc=r'environment rain water specific humidity')
            NS.add_profile('env_qr2', Gr, Pa, units=r'kg^2 kg^{-2}',nice_name=r'q_{r,e}^2',
                       desc=r'environment rain water specific humidity square')
            NS.add_profile('updraft_ql_qr', Gr, Pa, units=r'kg^2 kg^{-2}', nice_name=r'(q_l q_r)_u',
                       desc=r'updraft product of q_l and q_r')
            NS.add_profile('env_ql_qr', Gr, Pa, units=r'kg^2 kg^{-2}', nice_name=r'(q_l q_r)_e',
                       desc=r'environment product of q_l and q_r')


        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk

            Py_ssize_t var_shift
            double tau

        # Set the source term
        for level in self.tracer_dict.keys():
            for var in self.tracer_dict[level].keys():
                var_shift = PV.get_varshift(Gr, var)
                tau = self.tracer_dict[level][var]['timescale']
                with nogil:
                    for i in xrange(Gr.dims.npg):
                        PV.tendencies[var_shift + i] -= fmax(PV.values[var_shift + i],0.0)/tau

        #Below we assume domain uses only x-y decomposition, for the general case we should use pencils
        #but we don't address that complication for now

        # Set the value of the surface based tracers
        for var in self.tracer_dict['surface'].keys():
            var_shift = PV.get_varshift(Gr, var)
            with nogil:
                for i in xrange(Gr.dims.nlg[0]):
                    for j in xrange(Gr.dims.nlg[1]):
                        ijk = i * istride + j * jstride + Gr.dims.gw
                        PV.values[var_shift + ijk] = 1.0



        # Find surface thermodynamic properties to compute LCL (lifting condensation level)
        # Ref: M. G. Lawrence, "The relationship between relative humidity and the dew point
        # temperature in moist air: A simple conversion and applications", Bull. Am. Meteorol. Soc., 86, 225-233, 2005
        cdef:
            double [:] T_dew = np.zeros(Gr.dims.nl[0]*Gr.dims.nl[1],dtype=np.double,order='c')
            double [:] z_lcl = np.zeros(Gr.dims.nl[0]*Gr.dims.nl[1],dtype=np.double,order='c')
            Py_ssize_t t_shift, qv_shift, qt_shift
            Py_ssize_t  count = 0
            double B1 = 243.04  # C
            double A1 = 17.625 # no units
            double C1 = 610.94 # Pa
            double CtoK = 273.15 # additive celsius to kelvin conversion factor
            double vapor_pressure
            double nxny_i =1.0/( Gr.dims.n[0] * Gr.dims.n[1])
            double lcl_, lcl



        if self.lcl_tracers:
            t_shift = DV.get_varshift(Gr,'temperature')
            qt_shift = PV.get_varshift(Gr, 'qt')
            qv_shift = DV.get_varshift(Gr,'qv')
            with nogil:
                for i in xrange(Gr.dims.gw, Gr.dims.nlg[0]-Gr.dims.gw):
                    for j in xrange(Gr.dims.gw, Gr.dims.nlg[1]-Gr.dims.gw):
                        ijk = i * istride + j * jstride + Gr.dims.gw
                        vapor_pressure = pv_c(Ref.p0_half[Gr.dims.gw], PV.values[qt_shift + ijk], DV.values[qv_shift + ijk])
                        T_dew[count] = B1 * log(vapor_pressure/C1)/(A1-log(vapor_pressure/C1))  + CtoK
                        z_lcl[count] = 125.0 * (DV.values[t_shift+ijk] - T_dew[count])
                        count += 1
            lcl_ = np.sum(z_lcl) * nxny_i
            lcl = Pa.domain_scalar_sum(lcl_)

            for k in xrange(Gr.dims.nlg[2]-Gr.dims.gw, Gr.dims.gw-1, -1):
                if Gr.zl_half[k] < lcl:
                    self.index_lcl = k + 1
                    break

            for var in self.tracer_dict['lcl'].keys():
                var_shift = PV.get_varshift(Gr, var)
                with nogil:
                    for i in xrange( Gr.dims.nlg[0]):
                        for j in xrange( Gr.dims.nlg[1]):
                            ijk = i * istride + j * jstride + self.index_lcl
                            PV.values[var_shift + ijk] = 1.0


        return


    cpdef update_cleanup(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        cdef:
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t var_shift


        #Below we assume domain uses only x-y decomposition, for the general case we should use pencils
        #but we don't address that complication for now

        # Set the value of the surface based tracers

        for var in self.tracer_dict['surface'].keys():
            var_shift = PV.get_varshift(Gr, var)
            with nogil:
                for i in xrange(Gr.dims.nlg[0]):
                    for j in xrange(Gr.dims.nlg[1]):
                        ijk = i * istride + j * jstride + Gr.dims.gw
                        PV.tendencies[var_shift + ijk] = 0.0
                        for k in xrange( Gr.dims.nlg[2]):
                            ijk = i * istride + j * jstride + k
                            PV.values[var_shift + ijk] = fmax(PV.values[var_shift + ijk],0.0)
        if self.lcl_tracers:
            for var in self.tracer_dict['lcl'].keys():
                var_shift = PV.get_varshift(Gr, var)
                with nogil:
                    for i in xrange( Gr.dims.nlg[0]):
                        for j in xrange( Gr.dims.nlg[1]):
                            ijk = i * istride + j * jstride + self.index_lcl
                            PV.tendencies[var_shift + ijk] = 0.0
                            for k in xrange( Gr.dims.nlg[2]):
                                ijk = i * istride + j * jstride + k
                                PV.values[var_shift + ijk] = fmax(PV.values[var_shift + ijk],0.0)


        return


    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr,'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr,'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr,'w')
            Py_ssize_t q_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t c_shift = PV.get_varshift(Gr,'c_srf_15')
            Py_ssize_t b_shift = DV.get_varshift(Gr, 'buoyancy')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t bvf_shift = DV.get_varshift(Gr, 'buoyancy_frequency')
            Py_ssize_t thr_shift = DV.get_varshift(Gr, 'theta_rho')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t ql_shift, th_shift, qr_shift
            double [:] cloudfraction = np.zeros((Gr.dims.npg),dtype=np.double, order='c')
            double [:] tracer_normed = np.zeros((Gr.dims.npg),dtype=np.double, order='c')
            double [:] env_indicator = np.zeros((Gr.dims.npg),dtype=np.double, order='c')
            double [:] u_half = np.zeros((Gr.dims.npg),dtype=np.double, order='c')
            double [:] v_half = np.zeros((Gr.dims.npg),dtype=np.double, order='c')
            double [:] w_half = np.zeros((Gr.dims.npg),dtype=np.double, order='c')
            double [:] mean = Pa.HorizontalMean(Gr, &PV.values[c_shift])
            double [:] mean_square = Pa.HorizontalMeanofSquares(Gr, &PV.values[c_shift], &PV.values[c_shift])
            Py_ssize_t i,j,k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

        if self.lcl_tracers:
            NS.write_ts('grid_lcl',Gr.zl_half[self.index_lcl], Pa)
        if 'ql' in DV.name_index:
            ql_shift = DV.get_varshift(Gr,'ql')
            self.get_cloud_heights(Gr, DV, Pa)
            print('cloud base, height ', self.cloud_base, self.cloud_top)
            updraft_indicator_sc_w_ql(&Gr.dims, &PV.values[c_shift], &tracer_normed[0], &mean[0], &mean_square[0],
                                      &PV.values[w_shift],&DV.values[ql_shift], &Gr.z_half[0], self.cloud_base, self.cloud_top)
            # updraft_indicator_sc_w(&Gr.dims, &PV.values[c_shift], &tracer_normed[0], &mean[0], &mean_square[0], &PV.values[w_shift])

        else:
            updraft_indicator_sc_w(&Gr.dims, &PV.values[c_shift], &tracer_normed[0], &mean[0], &mean_square[0], &PV.values[w_shift])

        with nogil:
            for i in range(Gr.dims.nlg[0]):
                ishift = i * istride
                for j in range(Gr.dims.nlg[1]):
                    jshift = j * jstride
                    for k in range(Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k
                        if tracer_normed[ijk] >= 1.0:
                            self.updraft_indicator[ijk] = 1.0
                        else:
                            self.updraft_indicator[ijk] = 0.0
                        env_indicator[ijk] = 1.0 - self.updraft_indicator[ijk]
            for i in range(imin,imax):
                ishift = i * istride
                for j in range(jmin,jmax):
                    jshift = j * jstride
                    for k in range(kmin,kmax):
                        ijk = ishift + jshift + k
                        u_half[ijk] = 0.5 * (PV.values[u_shift+ijk-istride] + PV.values[u_shift+ijk])
                        v_half[ijk] = 0.5 * (PV.values[v_shift+ijk-jstride] + PV.values[v_shift+ijk])
                        w_half[ijk] = 0.5 * (PV.values[w_shift+ijk-1] + PV.values[w_shift+ijk])

        tmp = Pa.HorizontalMean(Gr, &self.updraft_indicator[0])
        NS.write_profile('updraft_fraction', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &w_half[0], &self.updraft_indicator[0])
        NS.write_profile('updraft_w', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &w_half[0], &w_half[0], &self.updraft_indicator[0])
        NS.write_profile('updraft_w2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &u_half[0], &self.updraft_indicator[0])
        NS.write_profile('updraft_u', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &u_half[0], &u_half[0], &self.updraft_indicator[0])
        NS.write_profile('updraft_u2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &v_half[0], &self.updraft_indicator[0])
        NS.write_profile('updraft_v', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr,&v_half[0], &v_half[0], &self.updraft_indicator[0])
        NS.write_profile('updraft_v2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[q_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_qt', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[q_shift], &PV.values[q_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_qt2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        if 'thetali' in DV.name_index:
            th_shift = DV.get_varshift(Gr, 'thetali')
        else:
            th_shift = DV.get_varshift(Gr, 'theta')

        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[th_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_thetali', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[th_shift], &DV.values[th_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_thetali2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[b_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_b', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[b_shift], &DV.values[b_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_b2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &w_half[0], &PV.values[q_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_w_qt', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &w_half[0], &DV.values[th_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_w_thetali', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[th_shift], &PV.values[q_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_qt_thetali', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[t_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_temperature', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[t_shift], &DV.values[t_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_temperature2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[qv_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_qv', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[qv_shift], &DV.values[qv_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_qv2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[thr_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_thetarho', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[bvf_shift], &self.updraft_indicator[0])
        NS.write_profile('updraft_bvf', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)




        tmp = Pa.HorizontalMean(Gr, &env_indicator[0])
        NS.write_profile('env_fraction', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &w_half[0], &env_indicator[0])
        NS.write_profile('env_w', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &w_half[0], &w_half[0], &env_indicator[0])
        NS.write_profile('env_w2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[u_shift], &env_indicator[0])
        NS.write_profile('env_u', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[u_shift], &PV.values[u_shift], &env_indicator[0])
        NS.write_profile('env_u2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[v_shift], &env_indicator[0])
        NS.write_profile('env_v', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[v_shift], &PV.values[v_shift], &env_indicator[0])
        NS.write_profile('env_v2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[q_shift], &env_indicator[0])
        NS.write_profile('env_qt', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[q_shift], &PV.values[q_shift], &env_indicator[0])
        NS.write_profile('env_qt2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[th_shift], &env_indicator[0])
        NS.write_profile('env_thetali', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[th_shift], &DV.values[th_shift], &env_indicator[0])
        NS.write_profile('env_thetali2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[b_shift], &env_indicator[0])
        NS.write_profile('env_b', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[b_shift], &DV.values[b_shift], &env_indicator[0])
        NS.write_profile('env_b2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanofSquaresConditional(Gr,&w_half[0], &PV.values[q_shift], &env_indicator[0])
        NS.write_profile('env_w_qt', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &w_half[0], &DV.values[th_shift], &env_indicator[0])
        NS.write_profile('env_w_thetali', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[th_shift], &PV.values[q_shift], &env_indicator[0])
        NS.write_profile('env_qt_thetali', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[t_shift], &env_indicator[0])
        NS.write_profile('env_temperature', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[t_shift], &DV.values[t_shift], &env_indicator[0])
        NS.write_profile('env_temperature2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[qv_shift], &env_indicator[0])
        NS.write_profile('env_qv', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[qv_shift], &DV.values[qv_shift], &env_indicator[0])
        NS.write_profile('env_qv2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[thr_shift], &env_indicator[0])
        NS.write_profile('env_thetarho', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[bvf_shift], &env_indicator[0])
        NS.write_profile('env_bvf', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        if 'ql' in DV.name_index:
            ql_shift = DV.get_varshift(Gr, 'ql')
            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[ql_shift], &self.updraft_indicator[0])
            NS.write_profile('updraft_ql', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[ql_shift], &DV.values[ql_shift], &self.updraft_indicator[0])
            NS.write_profile('updraft_ql2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

            tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[ql_shift], &env_indicator[0])
            NS.write_profile('env_ql', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[ql_shift], &DV.values[ql_shift], &env_indicator[0])
            NS.write_profile('env_ql2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

            with nogil:
                for i in range(Gr.dims.nlg[0]):
                    ishift = i * istride
                    for j in range(Gr.dims.nlg[1]):
                        jshift = j * jstride
                        for k in range(Gr.dims.nlg[2]):
                            ijk = ishift + jshift + k
                            if DV.values[ql_shift+ ijk] >= ql_threshold:
                                cloudfraction[ijk] = 1.0

            tmp = Pa.HorizontalMeanConditional(Gr, &cloudfraction[0], &self.updraft_indicator[0])
            NS.write_profile('updraft_cloudfraction', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanConditional(Gr, &cloudfraction[0], &env_indicator[0])
            NS.write_profile('env_cloudfraction', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)



        if 'qr' in PV.name_index:
            qr_shift = PV.get_varshift(Gr, 'qr')
            tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qr_shift], &self.updraft_indicator[0])
            NS.write_profile('updraft_qr', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[qr_shift], &PV.values[qr_shift], &self.updraft_indicator[0])
            NS.write_profile('updraft_qr2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

            tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qr_shift], &env_indicator[0])
            NS.write_profile('env_qr', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[qr_shift], &PV.values[qr_shift], &env_indicator[0])
            NS.write_profile('env_qr2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[ql_shift], &PV.values[qr_shift], &self.updraft_indicator[0])
            NS.write_profile('updraft_ql_qr', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

            tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[ql_shift], &PV.values[qr_shift], &env_indicator[0])
            NS.write_profile('env_ql_qr', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        return

    cpdef get_cloud_heights(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift, ijk
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            double cb, ct


        # Compute cloud top and cloud base height
        cb = 99999.9
        ct = -99999.9
        with nogil:
            for i in range(imin,imax):
                ishift = i * istride
                for j in range(jmin,jmax):
                    jshift = j * jstride
                    for k in range(kmin,kmax):
                        ijk = ishift + jshift + k
                        if DV.values[ql_shift+ijk] > ql_threshold:
                            cb = fmin(cb, Gr.z_half[k])
                            ct = fmax(ct, Gr.z_half[k])

        self.cloud_base = Pa.domain_scalar_min(cb)
        self.cloud_top =  Pa.domain_scalar_max(ct)

        return



cdef class PurityTracers:
    def __init__(self, namelist):
        cdef UpdraftTracers TracersUpdraft
        self.TracersUpdraft = UpdraftTracers(namelist)
        return

    cpdef initialize(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.TracersUpdraft.initialize(Gr, PV, DV, NS, Pa)
        # Here we need to add purity + origin info tracers for each of the updraft diagnostic tracers
        # To get this working, we assume only 15 min timescale diagnostic tracers

        PV.add_variable('purity_srf', '-', 'purity_srf', 'tracer diagnostics', "sym", "scalar", Pa)
        PV.add_variable('time_srf', '-', 'time_srf', 'tracer diagnostics', "sym", "scalar", Pa)
        PV.add_variable('qt_srf', '-', 'qt_srf', 'tracer diagnostics', "sym", "scalar", Pa)
        PV.add_variable('thetali_srf', '-', 'thetali_srf', 'tracer diagnostics', "sym", "scalar", Pa)

        if self.TracersUpdraft.lcl_tracers:

            PV.add_variable('purity_lcl', '-', 'purity_lcl', 'tracer diagnostics', "sym", "scalar", Pa)
            PV.add_variable('time_lcl', '-', 'time_lcl', 'tracer diagnostics', "sym", "scalar", Pa)
            PV.add_variable('qt_lcl', '-', 'qt_lcl', 'tracer diagnostics', "sym", "scalar", Pa)
            PV.add_variable('thetali_lcl', '-', 'thetali_lcl', 'tracer diagnostics', "sym", "scalar", Pa)

        NS.add_profile('updraft_purity_srf', Gr, Pa)
        NS.add_profile('updraft_time_srf', Gr, Pa)
        NS.add_profile('updraft_qt_srf', Gr, Pa)
        NS.add_profile('updraft_thetali_srf', Gr, Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        self.TracersUpdraft.update(Gr,Ref, PV, DV, TS, Pa)

        # First do the surface tracers
        cdef:
            Py_ssize_t w_shift = PV.get_varshift(Gr,'w')
            Py_ssize_t q_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t th_shift #= DV.get_varshift(Gr,'thetali')
            Py_ssize_t c_shift = PV.get_varshift(Gr,'c_srf_15')
            Py_ssize_t p_shift = PV.get_varshift(Gr,'purity_srf')
            Py_ssize_t pt_shift = PV.get_varshift(Gr,'time_srf')
            Py_ssize_t pq_shift = PV.get_varshift(Gr,'qt_srf')
            Py_ssize_t pth_shift = PV.get_varshift(Gr,'thetali_srf')
            Py_ssize_t index_lcl
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift, jshift, ijk, i,j,k
            Py_ssize_t gw = Gr.dims.gw, ql_shift


            double [:] tracer_normed = np.zeros((Gr.dims.npg),dtype=np.double, order='c')
            double [:] mean = Pa.HorizontalMean(Gr, &PV.values[c_shift])
            double [:] mean_square = Pa.HorizontalMeanofSquares(Gr, &PV.values[c_shift], &PV.values[c_shift])
            double max_x, max_y, max_z, total_max
            Py_ssize_t conditions_flag = 1

        if 'ql' not in DV.name_index and conditions_flag == 3:
            conditions_flag = 2

        if conditions_flag == 1:
            updraft_indicator_sc(&Gr.dims, &PV.values[c_shift], &tracer_normed[0],
                                   &mean[0], &mean_square[0])
        elif conditions_flag == 2:
            updraft_indicator_sc_w(&Gr.dims, &PV.values[c_shift], &tracer_normed[0],
                                   &mean[0], &mean_square[0], &PV.values[w_shift])
        elif conditions_flag == 3:
            ql_shift = DV.get_varshift(Gr, 'ql')
            self.TracersUpdraft.get_cloud_heights(Gr,DV, Pa)
            updraft_indicator_sc_w_ql(&Gr.dims, &PV.values[c_shift], &tracer_normed[0],&mean[0], &mean_square[0],
                                      &PV.values[w_shift], &DV.values[ql_shift], &Gr.z_half[0],
                                      self.TracersUpdraft.cloud_base, self.TracersUpdraft.cloud_top )


        if 'thetali' in DV.name_index:
            th_shift = DV.get_varshift(Gr,'thetali')
        else:
            th_shift = DV.get_varshift(Gr,'theta')

        with nogil:
            for i in xrange(gw, Gr.dims.nlg[0]-gw):
                for j in xrange(gw, Gr.dims.nlg[1]-gw):
                    ijk = i * istride + j * jstride + Gr.dims.gw
                    PV.values[p_shift + ijk] = 1.0
                    PV.values[pt_shift + ijk] = TS.t
                    PV.values[pq_shift + ijk] = PV.values[q_shift + ijk]
                    PV.values[pth_shift + ijk] = DV.values[th_shift + ijk]
                    for k in xrange(gw + 1, Gr.dims.nlg[2]-1):
                        ijk = i * istride + j * jstride + k
                        max_x = fmax(tracer_normed[ijk-istride],tracer_normed[ijk+istride])
                        max_y = fmax(tracer_normed[ijk-jstride],tracer_normed[ijk+jstride])
                        max_z = fmax(tracer_normed[ijk-1],tracer_normed[ijk+1])
                        total_max = fmax(fmax(fmax(max_x,max_y),max_z), tracer_normed[ijk])
                        if total_max < 0.9:
                            PV.values[p_shift + ijk] = 0.0
                            PV.values[pt_shift + ijk] = 0.0
                            PV.values[pq_shift + ijk] = 0.0
                            PV.values[pth_shift + ijk] = 0.0
                        else:
                            PV.values[p_shift + ijk] = fmin(fmax(0.0, PV.values[p_shift + ijk]), 1.0)
                            PV.values[pt_shift + ijk] = fmin(fmax(0.0,PV.values[pt_shift + ijk] ), TS.t)



        if self.TracersUpdraft.lcl_tracers:
            index_lcl = self.TracersUpdraft.index_lcl

            c_shift = PV.get_varshift(Gr,'c_lcl_15')
            p_shift = PV.get_varshift(Gr,'purity_lcl')
            pt_shift = PV.get_varshift(Gr,'time_lcl')
            pq_shift = PV.get_varshift(Gr,'qt_lcl')
            pth_shift = PV.get_varshift(Gr,'thetali_lcl')
            mean = Pa.HorizontalMean(Gr, &PV.values[c_shift])
            mean_square = Pa.HorizontalMeanofSquares(Gr, &PV.values[c_shift], &PV.values[c_shift])

            if conditions_flag == 1:
                updraft_indicator_sc(&Gr.dims, &PV.values[c_shift], &tracer_normed[0],
                                       &mean[0], &mean_square[0])
            elif conditions_flag == 2:
                updraft_indicator_sc_w(&Gr.dims, &PV.values[c_shift], &tracer_normed[0],
                                       &mean[0], &mean_square[0], &PV.values[w_shift])
            elif conditions_flag == 3:
                ql_shift = DV.get_varshift(Gr, 'ql')
                self.TracersUpdraft.get_cloud_heights(Gr,DV, Pa)
                updraft_indicator_sc_w_ql(&Gr.dims, &PV.values[c_shift], &tracer_normed[0],&mean[0], &mean_square[0],
                                          &PV.values[w_shift], &DV.values[ql_shift], &Gr.z_half[0],
                                          self.TracersUpdraft.cloud_base, self.TracersUpdraft.cloud_top )

            with nogil:
                for i in xrange(gw, Gr.dims.nlg[0]-gw):
                    for j in xrange(gw, Gr.dims.nlg[1]-gw):
                        for k in xrange(gw,index_lcl+1):
                            ijk = i * istride + j * jstride + k
                            PV.values[p_shift + ijk] = 1.0
                            PV.values[pt_shift + ijk] = TS.t
                            PV.values[pq_shift + ijk] = PV.values[q_shift + ijk]
                            PV.values[pth_shift + ijk] = DV.values[th_shift + ijk]
                        for k in xrange(index_lcl+1, Gr.dims.nlg[2]-gw):
                            ijk = i * istride + j * jstride + k
                            max_x = fmax(tracer_normed[ijk-istride],tracer_normed[ijk+istride])
                            max_y = fmax(tracer_normed[ijk-jstride],tracer_normed[ijk+jstride])
                            max_z = fmax(tracer_normed[ijk-1],tracer_normed[ijk+1])
                            total_max = fmax(fmax(fmax(max_x,max_y),max_z), tracer_normed[ijk])
                            if total_max < 0.9:
                                PV.values[p_shift + ijk] = 0.0
                                PV.values[pt_shift + ijk] = 0.0
                                PV.values[pq_shift + ijk] = 0.0
                                PV.values[pth_shift + ijk] = 0.0
                            else:
                                PV.values[p_shift + ijk] = fmin(fmax(0.0, PV.values[p_shift + ijk]), 1.0)
                                PV.values[pt_shift + ijk] = fmin(fmax(0.0,PV.values[pt_shift + ijk] ), TS.t)

        # Update the boundary points
        PV.Update_all_bcs(Gr, Pa )

        return

    cpdef update_cleanup(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        self.TracersUpdraft.update_cleanup(Gr, Ref, PV, DV, Pa, TS)
        cdef:
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t p_shift = PV.get_varshift(Gr,'purity_srf')
            Py_ssize_t pt_shift = PV.get_varshift(Gr,'time_srf')
            Py_ssize_t pq_shift = PV.get_varshift(Gr,'qt_srf')
            Py_ssize_t pth_shift = PV.get_varshift(Gr,'thetali_srf')



        #Below we assume domain uses only x-y decomposition, for the general case we should use pencils
        #but we don't address that complication for now

        # Set the value of the surface based tracers

        with nogil:
            for i in xrange(Gr.dims.nlg[0]):
                for j in xrange(Gr.dims.nlg[1]):
                    ijk = i * istride + j * jstride + Gr.dims.gw
                    PV.tendencies[p_shift + ijk] = 0.0
                    PV.tendencies[pt_shift + ijk] = 0.0
                    PV.tendencies[pq_shift + ijk] = 0.0
                    PV.tendencies[pth_shift + ijk] = 0.0
                    for k in xrange( Gr.dims.nlg[2]):
                        ijk = i * istride + j * jstride + k
                        PV.values[p_shift + ijk] = fmin(fmax(PV.values[p_shift + ijk],0.0),1.0)
                        PV.values[pt_shift + ijk] = fmin(fmax(PV.values[pt_shift + ijk],0.0), TS.t)
                        PV.values[pq_shift + ijk] = fmax(PV.values[pq_shift + ijk],0.0)
                        PV.values[pth_shift + ijk] = fmax(PV.values[pth_shift + ijk],0.0)


        if self.TracersUpdraft.lcl_tracers:
            p_shift = PV.get_varshift(Gr,'purity_lcl')
            pt_shift = PV.get_varshift(Gr,'time_lcl')
            pq_shift = PV.get_varshift(Gr,'qt_lcl')
            pth_shift = PV.get_varshift(Gr,'thetali_lcl')
            with nogil:
                for i in xrange( Gr.dims.nlg[0]):
                    for j in xrange( Gr.dims.nlg[1]):
                        for k in xrange(self.TracersUpdraft.index_lcl + 1):
                            ijk = i * istride + j * jstride + k
                            PV.tendencies[p_shift + ijk] = 0.0
                            PV.tendencies[pt_shift + ijk] = 0.0
                            PV.tendencies[pq_shift + ijk] = 0.0
                            PV.tendencies[pth_shift + ijk] = 0.0
                        for k in xrange(self.TracersUpdraft.index_lcl + 1, Gr.dims.nlg[2]):
                            ijk = i * istride + j * jstride + k
                            PV.values[p_shift + ijk] = fmin(fmax(PV.values[p_shift + ijk],0.0),1.0)
                            PV.values[pt_shift + ijk] = fmin(fmax(PV.values[pt_shift + ijk],0.0), TS.t)
                            PV.values[pq_shift + ijk] = fmax(PV.values[pq_shift + ijk],0.0)
                            PV.values[pth_shift + ijk] = fmax(PV.values[pth_shift + ijk],0.0)


        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.TracersUpdraft.stats_io(Gr, PV, DV, TS, NS, Pa)

        cdef:
            double [:] extracted_purity_var = np.zeros((Gr.dims.npg),dtype=np.double, order='c')
            Py_ssize_t p_shift = PV.get_varshift(Gr,'purity_srf')
            Py_ssize_t pt_shift = PV.get_varshift(Gr,'time_srf')
            Py_ssize_t pq_shift = PV.get_varshift(Gr,'qt_srf')
            Py_ssize_t pth_shift = PV.get_varshift(Gr,'thetali_srf')

        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[p_shift], &self.TracersUpdraft.updraft_indicator[0])
        NS.write_profile('updraft_purity_srf', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        purity_extract_time(&Gr.dims,  &PV.values[p_shift], &PV.values[pt_shift], &extracted_purity_var[0], TS.t)
        tmp = Pa.HorizontalMeanConditional(Gr, &extracted_purity_var[0], &self.TracersUpdraft.updraft_indicator[0])
        NS.write_profile('updraft_time_srf', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        purity_extract_value(&Gr.dims,  &PV.values[p_shift], &PV.values[pq_shift], &extracted_purity_var[0])
        tmp = Pa.HorizontalMeanConditional(Gr, &extracted_purity_var[0], &self.TracersUpdraft.updraft_indicator[0])
        NS.write_profile('updraft_qt_srf', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        purity_extract_value(&Gr.dims,  &PV.values[p_shift], &PV.values[pth_shift], &extracted_purity_var[0])
        tmp = Pa.HorizontalMeanConditional(Gr, &extracted_purity_var[0], &self.TracersUpdraft.updraft_indicator[0])
        NS.write_profile('updraft_thetali_srf', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        return

cdef updraft_indicator_sc(Grid.DimStruct *dims, double *tracer_raw, double *tracer_normed,
                          double *mean, double *meansquare_sigma):
    cdef:
        Py_ssize_t imax = dims.nlg[0]
        Py_ssize_t jmax = dims.nlg[1]
        Py_ssize_t kmax = dims.nlg[2]
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double sigma_min
        double sigma_sum = 0.0


    with nogil:
        for k in xrange(kmax):
            meansquare_sigma[k] = meansquare_sigma[k] - mean[k] * mean[k]
            meansquare_sigma[k] = sqrt(fmax(meansquare_sigma[k],0.0))
            sigma_sum += meansquare_sigma[k]
            sigma_min = sigma_sum/(k+1.0) * 0.05
            if meansquare_sigma[k] < sigma_min:
               for i in xrange(imax):
                    ishift = i*istride
                    for j in xrange(jmax):
                        jshift = j*jstride
                        ijk = ishift + jshift + k
                        tracer_normed[ijk] = 0.0
            else:
               for i in xrange(imax):
                    ishift = i*istride
                    for j in xrange(jmax):
                        jshift = j*jstride
                        ijk = ishift + jshift + k
                        tracer_normed[ijk] =  (tracer_raw[ijk] - mean[k])/ meansquare_sigma[k]

    return




cdef updraft_indicator_sc_w(Grid.DimStruct *dims, double *tracer_raw, double *tracer_normed, double *mean, double *meansquare_sigma, double *w):
    cdef:
        Py_ssize_t imax = dims.nlg[0]
        Py_ssize_t jmax = dims.nlg[1]
        Py_ssize_t kmax = dims.nlg[2]
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double sigma_min
        double sigma_sum = 0.0
        double w_half


    with nogil:
        for k in xrange(1,kmax):
            meansquare_sigma[k] = meansquare_sigma[k] - mean[k] * mean[k]
            meansquare_sigma[k] = sqrt(fmax(meansquare_sigma[k],0.0))
            sigma_sum += meansquare_sigma[k]
            sigma_min = sigma_sum/(k+1.0) * 0.05
            if meansquare_sigma[k] < sigma_min:
               for i in xrange(imax):
                    ishift = i*istride
                    for j in xrange(jmax):
                        jshift = j*jstride
                        ijk = ishift + jshift + k
                        tracer_normed[ijk] = 0.0
            else:
               for i in xrange(imax):
                    ishift = i*istride
                    for j in xrange(jmax):
                        jshift = j*jstride
                        ijk = ishift + jshift + k
                        w_half = 0.5*(w[ijk-1] + w[ijk])
                        tracer_normed[ijk] = copysign( (tracer_raw[ijk] - mean[k])/ meansquare_sigma[k]    , w_half)

    return

cdef updraft_indicator_sc_w_ql(Grid.DimStruct *dims,  double *tracer_raw, double *tracer_normed, double *mean,
                               double *meansquare_sigma, double *w, double *ql, double *z_half, double z_cb, double z_ct):
    cdef:
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double sigma_min
        double sigma_sum = 0.0
        double z_ql = z_cb + 0.25 * (z_ct - z_cb)
        double w_half


    with nogil:
        for k in xrange(dims.nlg[2]):
            meansquare_sigma[k] = meansquare_sigma[k] - mean[k] * mean[k]
            meansquare_sigma[k] = sqrt(fmax(meansquare_sigma[k],0.0))
            sigma_sum += meansquare_sigma[k]
            sigma_min = sigma_sum/(k+1.0) * 0.05
            if meansquare_sigma[k] < sigma_min:
               for i in xrange(dims.nlg[0]):
                    ishift = i*istride
                    for j in xrange(dims.nlg[1]):
                        jshift = j*jstride
                        ijk = ishift + jshift + k
                        tracer_normed[ijk] = 0.0
            else:
               for i in xrange(dims.nlg[0]):
                    ishift = i*istride
                    for j in xrange(dims.nlg[1]):
                        jshift = j*jstride
                        ijk = ishift + jshift + k
                        w_half = 0.5*(w[ijk-1] + w[ijk])
                        tracer_normed[ijk] = copysign( (tracer_raw[ijk] - mean[k])/ meansquare_sigma[k], w_half)

    if z_ct > z_cb:
        with nogil:
            for k in xrange(dims.nlg[2]):
                if z_half[k] >= z_ql and z_half[k] <= z_ct:
                    for i in xrange(dims.nlg[0]):
                        for j in xrange(dims.nlg[1]):
                            ijk = i * istride + j * jstride + k
                            if ql[ijk] < ql_threshold:
                                tracer_normed[ijk] = 0.0
    return


cdef purity_extract_time(Grid.DimStruct *dims,  double *purity_tracer, double *time_tracer_raw, double *time_tracer,
                         double current_time):
    cdef:
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k


    with nogil:
        for i in xrange(dims.gw, dims.nlg[0]-dims.gw):
            ishift = i * istride
            for j in xrange(dims.gw, dims.nlg[1]-dims.gw):
                jshift = j * jstride
                for k in xrange(dims.gw,dims.nlg[2]-dims.gw):
                    ijk = ishift + jshift + k
                    time_tracer[ijk] = (fmin(fmax(time_tracer_raw[ijk],0.0),current_time))/fmin(fmax(purity_tracer[ijk],1e-3),1.0)
                    time_tracer[ijk] = current_time - time_tracer[ijk]
    return



cdef purity_extract_value(Grid.DimStruct *dims,  double *purity_tracer, double *value_tracer_raw, double *value_tracer):
    cdef:
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k


    with nogil:
        for i in xrange(dims.gw, dims.nlg[0]-dims.gw):
            ishift = i * istride
            for j in xrange(dims.gw, dims.nlg[1]-dims.gw):
                jshift = j * jstride
                for k in xrange(dims.gw,dims.nlg[2]-dims.gw):
                    ijk = ishift + jshift + k
                    value_tracer[ijk] = fmax(value_tracer_raw[ijk],0.0)/fmin(fmax(purity_tracer[ijk],1e-3),1.0)
    return
