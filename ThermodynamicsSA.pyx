#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

cimport numpy as np
import numpy as np
cimport Lookup
cimport ParallelMPI
cimport Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport PrognosticVariables
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
from thermodynamic_functions cimport thetas_c, theta_c, thetali_c
import cython
from NetCDFIO cimport NetCDFIO_Stats, NetCDFIO_Fields
from libc.math cimport fmax, fmin

cdef extern from "thermodynamics_sa.h":
    inline double alpha_c(double p0, double T, double qt, double qv) nogil
    void eos_c(Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double), double p0, double s, double qt, double *T, double *qv, double *ql, double *qi) nogil
    # __
    void eos_c_refstate(Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double), double p0, double s, double qt, double *T, double *qv, double *ql, double *qi) nogil
    void eos_update(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double), double *p0, double *s, double *qt, double *T,
                    double * qv, double * ql, double * qi, double * alpha)
    # __
    # void eos_update(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double), double *p0, double *s, double *qt, double *T,
    #                 double * qv, double * ql, double * qi, double * alpha)
    void buoyancy_update_sa(Grid.DimStruct *dims, double *alpha0, double *alpha, double *buoyancy, double *wt)
    void bvf_sa(Grid.DimStruct * dims, Lookup.LookupStruct * LT, double(*lam_fp)(double), double(*L_fp)(double, double), double *p0, double *T, double *qt, double *qv, double *theta_rho, double *bvf)
    void thetali_update(Grid.DimStruct *dims, double (*lam_fp)(double), double (*L_fp)(double, double), double *p0, double *T, double *qt, double *ql, double *qi, double *thetali)
    void clip_qt(Grid.DimStruct *dims, double  *qt, double clip_value)

cdef extern from "thermodynamic_functions.h":
    # Dry air partial pressure
    inline double pd_c(double p0, double qt, double qv) nogil
    # Water vapor partial pressure
    inline double pv_c(double p0, double qt, double qv) nogil


cdef extern from "entropies.h":
    # Specific entropy of dry air
    inline double sd_c(double pd, double T) nogil
    # Specific entropy of water vapor
    inline double sv_c(double pv, double T) nogil
    # Specific entropy of condensed water
    inline double sc_c(double L, double T) nogil


cdef class ThermodynamicsSA:
    def __init__(self, dict namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
        '''
        Init method saturation adjsutment thermodynamics.
        :param namelist: dictionary
        :param LH: LatentHeat class instance
        :param Par: ParallelMPI class instance
        :return:
        '''

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Par)

        #Check to see if qt clipping is to be done. By default qt_clipping is on.
        try:
            self.do_qt_clipping = namelist['thermodynamics']['do_qt_clipping']
        except:
            self.do_qt_clipping = True

        return


    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        '''
        Initialize ThermodynamicsSA class. Adds variables to PrognocitVariables and DiagnosticVariables classes. Add
        output fields to NetCDFIO_Stats.
        :param Gr: Grid class instance
        :param PV: PrognosticVariables class instance
        :param DV: DiagnsoticVariables class instance
        :param NS: NetCDFIO_Stats class instance
        :param Pa: ParallelMPI class instance
        :return:
        '''

        PV.add_variable('s', 'm/s', "sym", "scalar", Pa)
        PV.add_variable('qt', 'kg/kg', "sym", "scalar", Pa)

        # Initialize class member arrays
        DV.add_variables('buoyancy', '--', 'sym', Pa)
        DV.add_variables('alpha', '--', 'sym', Pa)
        DV.add_variables('temperature', 'K', 'sym', Pa)
        DV.add_variables('buoyancy_frequency', '1/s', 'sym', Pa)
        DV.add_variables('qv', 'kg/kg', 'sym', Pa)
        DV.add_variables('ql', 'kg/kg', 'sym', Pa)
        DV.add_variables('qi', 'kg/kg', 'sym', Pa)
        DV.add_variables('theta_rho', 'K', 'sym', Pa)
        DV.add_variables('thetali', 'K', 'sym', Pa)


        # Add statistical output
        NS.add_profile('thetas_mean', Gr, Pa)
        NS.add_profile('thetas_mean2', Gr, Pa)
        NS.add_profile('thetas_mean3', Gr, Pa)
        NS.add_profile('thetas_max', Gr, Pa)
        NS.add_profile('thetas_min', Gr, Pa)
        NS.add_ts('thetas_max', Gr, Pa)
        NS.add_ts('thetas_min', Gr, Pa)

        NS.add_profile('theta_mean', Gr, Pa)
        NS.add_profile('theta_mean2', Gr, Pa)
        NS.add_profile('theta_mean3', Gr, Pa)
        NS.add_profile('theta_max', Gr, Pa)
        NS.add_profile('theta_min', Gr, Pa)
        NS.add_ts('theta_max', Gr, Pa)
        NS.add_ts('theta_min', Gr, Pa)


        NS.add_profile('rh_mean', Gr, Pa)
        # NS.add_profile('rh_mean2', Gr, Pa)
        # NS.add_profile('rh_mean3', Gr, Pa)
        NS.add_profile('rh_max', Gr, Pa)
        NS.add_profile('rh_min', Gr, Pa)
        # NS.add_ts('rh_max', Gr, Pa)
        # NS.add_ts('rh_min', Gr, Pa)


        NS.add_profile('cloud_fraction', Gr, Pa)
        NS.add_ts('cloud_fraction', Gr, Pa)
        NS.add_ts('cloud_top', Gr, Pa)
        NS.add_ts('cloud_base', Gr, Pa)
        NS.add_ts('lwp', Gr, Pa)


        return

    cpdef entropy(self, double p0, double T, double qt, double ql, double qi):
        '''
        Provide a python wrapper for the c function that computes the specific entropy
        consistent with Pressel et al. 2015 equation (40)
        :param p0: reference state pressure [Pa]
        :param T: thermodynamic temperature [K]
        :param qt: total water specific humidity [kg/kg]
        :param ql: liquid water specific humidity [kg/kg]
        :param qi: ice water specific humidity [kg/kg]
        :return: moist specific entropy
        '''
        cdef:
            double qv = qt - ql - qi
            double qd = 1.0 - qt
            double pd = pd_c(p0, qt, qv)
            double pv = pv_c(p0, qt, qv)
            double Lambda = self.Lambda_fp(T)
            double L = self.L_fp(T, Lambda)

        return sd_c(pd, T) * (1.0 - qt) + sv_c(pv, T) * qt + sc_c(L, T) * (ql + qi)

    cpdef alpha(self, double p0, double T, double qt, double qv):
        '''
        Provide a python wrapper for the C function that computes the specific volume
        consistent with Pressel et al. 2015 equation (44).
        :param p0: reference state pressure [Pa]
        :param T:  thermodynamic temperature [K]
        :param qt: total water specific humidity [kg/kg]
        :param qv: water vapor specific humidity [kg/kg]
        :return: specific volume [m^3/kg]
        '''
        return alpha_c(p0, T, qt, qv)

    cpdef eos(self, double p0, double s, double qt):
        cdef:
            double T, qv, qc, ql, qi, lam
        # eos_c(&self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, p0, s, qt, &T, &qv, &ql, &qi)
        eos_c_refstate(&self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, p0, s, qt, &T, &qv, &ql, &qi)
        return T, ql, qi

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
        print('Th_SA.update')
        # Get relevant variables shifts
        cdef:
            Py_ssize_t buoyancy_shift = DV.get_varshift(Gr, 'buoyancy')
            Py_ssize_t alpha_shift = DV.get_varshift(Gr, 'alpha')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t bvf_shift = DV.get_varshift(Gr, 'buoyancy_frequency')
            Py_ssize_t thr_shift = DV.get_varshift(Gr, 'theta_rho')
            Py_ssize_t thl_shift = DV.get_varshift(Gr, 'thetali')


        '''Apply qt clipping if requested. Defaults to on. Call this before other thermodynamic routines. Note that this
        changes the values in the qt array directly. Perhaps we should eventually move this to the timestepping function
        so that the output statistics correctly reflect clipping.
        '''
        if self.do_qt_clipping:
            clip_qt(&Gr.dims, &PV.values[qt_shift], 1e-11)
        # __
        # self.debug_tend('000',PV,DV,Gr)#,Pa)
        #  __

        eos_update(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.p0_half[0],
                    &PV.values[s_shift], &PV.values[qt_shift], &DV.values[t_shift], &DV.values[qv_shift], &DV.values[ql_shift],
                    &DV.values[qi_shift], &DV.values[alpha_shift])

        buoyancy_update_sa(&Gr.dims, &RS.alpha0_half[0], &DV.values[alpha_shift], &DV.values[buoyancy_shift], &PV.tendencies[w_shift])

        # __
        # self.debug_tend('222',PV,DV,Gr)#,Pa)        # nans in
        # __

        bvf_sa( &Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[thr_shift], &DV.values[bvf_shift])

        # # __
        # message = '333'
        # self.debug_tend(message,PV,DV,Gr)#,Pa)
        # # __

        thetali_update(&Gr.dims,self.Lambda_fp, self.L_fp, &RS.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], &DV.values[ql_shift],&DV.values[qi_shift],&DV.values[thl_shift])

        return

    cpdef get_pv_star(self, t):
        return self.CC.LT.fast_lookup(t)

    cpdef get_lh(self, t):
        cdef double lam = self.Lambda_fp(t)
        return self.L_fp(t, lam)

    cpdef write_fields(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                       PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Fields NF, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t count
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double[:] data = np.empty((Gr.dims.npl,), dtype=np.double, order='c')


        # Add entropy potential temperature to 3d fields
        with nogil:
            count = 0
            for i in range(imin, imax):
                ishift = i * istride
                for j in range(jmin, jmax):
                    jshift = j * jstride
                    for k in range(kmin, kmax):
                        ijk = ishift + jshift + k
                        data[count] = thetas_c(PV.values[s_shift + ijk], PV.values[qt_shift + ijk])
                        count += 1
        NF.add_field('thetas')
        NF.write_field('thetas', data)
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = 0
            Py_ssize_t jmin = 0
            Py_ssize_t kmin = 0
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t kmax = Gr.dims.nlg[2]
            Py_ssize_t count
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double[:] data = np.empty((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] tmp



        # Ouput profiles of thetas
        with nogil:
            count = 0
            for i in range(imin, imax):
                ishift = i * istride
                for j in range(jmin, jmax):
                    jshift = j * jstride
                    for k in range(kmin, kmax):
                        ijk = ishift + jshift + k
                        data[count] = thetas_c(PV.values[s_shift + ijk], PV.values[qt_shift + ijk])

                        count += 1



        # Compute and write mean

        tmp = Pa.HorizontalMean(Gr, &data[0])
        NS.write_profile('thetas_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Compute and write mean of squres
        tmp = Pa.HorizontalMeanofSquares(Gr, &data[0], &data[0])
        NS.write_profile('thetas_mean2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Compute and write mean of cubes
        tmp = Pa.HorizontalMeanofCubes(Gr, &data[0], &data[0], &data[0])
        NS.write_profile('thetas_mean3', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Compute and write maxes
        tmp = Pa.HorizontalMaximum(Gr, &data[0])
        NS.write_profile('thetas_max', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_ts('thetas_max', np.amax(tmp[Gr.dims.gw:-Gr.dims.gw]), Pa)

        # Compute and write mins
        tmp = Pa.HorizontalMinimum(Gr, &data[0])
        NS.write_profile('thetas_min', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_ts('thetas_min', np.amin(tmp[Gr.dims.gw:-Gr.dims.gw]), Pa)


        #Output profiles of theta (dry potential temperature)
        cdef:
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')

        with nogil:
            count = 0
            for i in range(imin, imax):
                ishift = i * istride
                for j in range(jmin, jmax):
                    jshift = j * jstride
                    for k in range(kmin, kmax):
                        ijk = ishift + jshift + k
                        data[count] = theta_c(RS.p0_half[k], DV.values[t_shift + ijk])
                        count += 1

        # Compute and write mean
        tmp = Pa.HorizontalMean(Gr, &data[0])
        NS.write_profile('theta_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Compute and write mean of squres
        tmp = Pa.HorizontalMeanofSquares(Gr, &data[0], &data[0])
        NS.write_profile('theta_mean2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Compute and write mean of cubes
        tmp = Pa.HorizontalMeanofCubes(Gr, &data[0], &data[0], &data[0])
        NS.write_profile('theta_mean3', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Compute and write maxes
        tmp = Pa.HorizontalMaximum(Gr, &data[0])
        NS.write_profile('theta_max', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_ts('theta_max', np.amax(tmp[Gr.dims.gw:-Gr.dims.gw]), Pa)

        # Compute and write mins
        tmp = Pa.HorizontalMinimum(Gr, &data[0])
        NS.write_profile('theta_min', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_ts('theta_min', np.amin(tmp[Gr.dims.gw:-Gr.dims.gw]), Pa)


        cdef:
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            double pv_star, pv


        # Ouput profiles of relative humidity
        with nogil:
            count = 0
            for i in range(imin, imax):
                ishift = i * istride
                for j in range(jmin, jmax):
                    jshift = j * jstride
                    for k in range(kmin, kmax):
                        ijk = ishift + jshift + k
                        pv_star = self.CC.LT.fast_lookup(DV.values[t_shift + ijk])
                        pv = pv_c(RS.p0_half[k], PV.values[qt_shift+ijk], DV.values[qv_shift+ijk])
                        data[count] = pv/pv_star

                        count += 1



        # Compute and write mean

        tmp = Pa.HorizontalMean(Gr, &data[0])
        NS.write_profile('rh_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Compute and write mean of squres
        # tmp = Pa.HorizontalMeanofSquares(Gr, &data[0], &data[0])
        # NS.write_profile('rh_mean2', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        #
        # # Compute and write mean of cubes
        # tmp = Pa.HorizontalMeanofCubes(Gr, &data[0], &data[0], &data[0])
        # NS.write_profile('rh_mean3', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Compute and write maxes
        tmp = Pa.HorizontalMaximum(Gr, &data[0])
        NS.write_profile('rh_max', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        # NS.write_ts('rh_max', np.amax(tmp[Gr.dims.gw:-Gr.dims.gw]), Pa)

        # Compute and write mins
        tmp = Pa.HorizontalMinimum(Gr, &data[0])
        NS.write_profile('rh_min', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        # NS.write_ts('rh_min', np.amin(tmp[Gr.dims.gw:-Gr.dims.gw]), Pa)

        #Output profiles of thetali  (liquid-ice potential temperature)
        # Compute additional stats
        self.liquid_stats(Gr, RS, PV, DV, NS, Pa)

        return

    cpdef liquid_stats(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                       DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t kmin = 0
            Py_ssize_t kmax = Gr.dims.n[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t pi, k
            ParallelMPI.Pencil z_pencil = ParallelMPI.Pencil()
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            double[:, :] ql_pencils
            # Cloud indicator
            double[:] ci
            double cb
            double ct
            # Weighted sum of local cloud indicator
            double ci_weighted_sum = 0.0
            double mean_divisor = np.double(Gr.dims.n[0] * Gr.dims.n[1])

            double dz = Gr.dims.dx[2]
            double[:] lwp
            double lwp_weighted_sum = 0.0

            double[:] cf_profile = np.zeros((Gr.dims.n[2]), dtype=np.double, order='c')

        # Initialize the z-pencil
        z_pencil.initialize(Gr, Pa, 2)
        ql_pencils =  z_pencil.forward_double( &Gr.dims, Pa, &DV.values[ql_shift])

        # Compute cloud fraction profile
        with nogil:
            for pi in xrange(z_pencil.n_local_pencils):
                for k in xrange(kmin, kmax):
                    if ql_pencils[pi, k] > 0.0:
                        cf_profile[k] += 1.0 / mean_divisor

        cf_profile = Pa.domain_vector_sum(cf_profile, Gr.dims.n[2])
        NS.write_profile('cloud_fraction', cf_profile, Pa)

        # Compute all or nothing cloud fraction
        ci = np.empty((z_pencil.n_local_pencils), dtype=np.double, order='c')
        with nogil:
            for pi in xrange(z_pencil.n_local_pencils):
                for k in xrange(kmin, kmax):
                    if ql_pencils[pi, k] > 0.0:
                        ci[pi] = 1.0
                        break
                    else:
                        ci[pi] = 0.0
            for pi in xrange(z_pencil.n_local_pencils):
                ci_weighted_sum += ci[pi]
            ci_weighted_sum /= mean_divisor

        ci_weighted_sum = Pa.domain_scalar_sum(ci_weighted_sum)
        NS.write_ts('cloud_fraction', ci_weighted_sum, Pa)

        # Compute cloud top and cloud base height
        cb = 99999.9
        ct = -99999.9
        with nogil:
            for pi in xrange(z_pencil.n_local_pencils):
                for k in xrange(kmin, kmax):
                    if ql_pencils[pi, k] > 0.0:
                        cb = fmin(cb, Gr.z_half[gw + k])
                        ct = fmax(ct, Gr.z_half[gw + k])

        cb = Pa.domain_scalar_min(cb)
        ct = Pa.domain_scalar_max(ct)
        NS.write_ts('cloud_base', cb, Pa)
        NS.write_ts('cloud_top', ct, Pa)

        # Compute liquid water path
        lwp = np.empty((z_pencil.n_local_pencils), dtype=np.double, order='c')
        with nogil:
            for pi in xrange(z_pencil.n_local_pencils):
                lwp[pi] = 0.0
                for k in xrange(kmin, kmax):
                    lwp[pi] += RS.rho0_half[k] * ql_pencils[pi, k] * dz

            for pi in xrange(z_pencil.n_local_pencils):
                lwp_weighted_sum += lwp[pi]
            lwp_weighted_sum /= mean_divisor

        lwp_weighted_sum = Pa.domain_scalar_sum(lwp_weighted_sum)
        NS.write_ts('lwp', lwp_weighted_sum, Pa)

        return



# _______________
    cpdef debug_tend(self, message, PrognosticVariables.PrognosticVariables PV_, DiagnosticVariables.DiagnosticVariables DV,
                     Grid.Grid Gr_):#, ParallelMPI.ParallelMPI Pa):
        # print('debug_tend, rank: ', Pa.rank)
        # message = 'hoi'
        cdef:
            Py_ssize_t u_varshift = PV_.get_varshift(Gr_,'u')
            Py_ssize_t v_varshift = PV_.get_varshift(Gr_,'v')
            Py_ssize_t w_varshift = PV_.get_varshift(Gr_,'w')
            Py_ssize_t s_varshift = PV_.get_varshift(Gr_,'s')
            Py_ssize_t b_shift = DV.get_varshift(Gr_, 'buoyancy')
            Py_ssize_t alpha_shift = DV.get_varshift(Gr_, 'alpha')
            Py_ssize_t t_shift = DV.get_varshift(Gr_, 'temperature')

            Py_ssize_t istride = Gr_.dims.nlg[1] * Gr_.dims.nlg[2]
            Py_ssize_t jstride = Gr_.dims.nlg[2]
            Py_ssize_t imax = Gr_.dims.nlg[0]
            Py_ssize_t jmax = Gr_.dims.nlg[1]
            Py_ssize_t kmax = Gr_.dims.nlg[2]
            Py_ssize_t gw = Gr_.dims.gw
            Py_ssize_t ijk_max = imax*istride + jmax*jstride + kmax

            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t imin = 0#Gr_.dims.gw
            Py_ssize_t jmin = 0#Gr_.dims.gw
            Py_ssize_t kmin = 0#Gr_.dims.gw
            # int [:] sk_arr = np.zeros(1, dtype=np.int)
            # int [:] qtk_arr = np.zeros(1, dtype=int)
        sk_arr = np.zeros(1,dtype=np.int)
        qtk_arr = np.zeros(1,dtype=np.int)

        u_max = np.nanmax(PV_.tendencies[u_varshift:v_varshift])
        uk_max = np.nanargmax(PV_.tendencies[u_varshift:v_varshift])
        u_min = np.nanmin(PV_.tendencies[u_varshift:v_varshift])
        uk_min = np.nanargmin(PV_.tendencies[u_varshift:v_varshift])
        v_max = np.nanmax(PV_.tendencies[v_varshift:w_varshift])
        vk_max = np.nanargmax(PV_.tendencies[v_varshift:w_varshift])
        v_min = np.nanmin(PV_.tendencies[v_varshift:w_varshift])
        vk_min = np.nanargmin(PV_.tendencies[v_varshift:w_varshift])
        w_max = np.nanmax(PV_.tendencies[w_varshift:s_varshift])
        wk_max = np.nanargmax(PV_.tendencies[w_varshift:s_varshift])
        w_min = np.nanmin(PV_.tendencies[w_varshift:s_varshift])
        wk_min = np.nanargmin(PV_.tendencies[w_varshift:s_varshift])

        # if np.isnan(PV_.tendencies).any():
        if 1==1:
            u_nan = np.isnan(PV_.tendencies[u_varshift:v_varshift]).any()
            uk_nan = np.argmax(PV_.tendencies[u_varshift:v_varshift])
            v_nan = np.isnan(PV_.tendencies[v_varshift:w_varshift]).any()
            vk_nan = np.argmax(PV_.tendencies[v_varshift:w_varshift])
            w_nan = np.isnan(PV_.tendencies[w_varshift:s_varshift]).any()
            wk_nan = np.argmax(PV_.tendencies[w_varshift:s_varshift])

        w_max_val= np.nanmax(PV_.values[w_varshift:s_varshift])
        wk_max_val = np.nanargmax(PV_.values[w_varshift:s_varshift])
        w_min_val = np.nanmin(PV_.values[w_varshift:s_varshift])
        wk_min_val = np.nanargmin(PV_.tendencies[w_varshift:s_varshift])
        w_nan_val = np.isnan(PV_.values[w_varshift:s_varshift]).any()
        # if (PV_.values[w_varshifts + 13840] != PV_.values[w_varshifts + 13840])
        wk_nan_val = np.argmax(PV_.values[w_varshift:s_varshift])

        # if Pa.rank == 0:
        if 1==1:
            print(message, 'debugging (max, min, nan): ')
            print('shifts', u_varshift, v_varshift, w_varshift, s_varshift, 'Gr.npg', (imax+2*gw)*(jmax+2*gw)*(kmax+2*gw))
            print('u tend: ', u_max, uk_max, u_min, uk_min, u_nan, uk_nan)
            print('v tend: ', v_max, vk_max, v_min, vk_min, v_nan, vk_nan)
            print('w tend: ', w_max, wk_max, w_min, wk_min, w_nan, wk_nan)
            print('w val: ', w_max_val, wk_max_val, w_min_val, wk_min_val, w_nan_val, wk_nan_val)


        # if np.isnan(DV.values).any():
        if 1==1:
            b_nan_val = np.isnan(DV.values[b_shift:b_shift+ ijk_max]).any()
            bk_nan_val = np.argmax(DV.values[b_shift:b_shift+ ijk_max])
            alpha_nan_val = np.isnan(DV.values[alpha_shift:alpha_shift+ ijk_max]).any()
            alphak_nan_val = np.argmax(DV.values[alpha_shift:alpha_shift+ ijk_max])
            t_nan_val = np.isnan(DV.values[t_shift:t_shift+ ijk_max]).any()
            tk_nan_val = np.argmax(DV.values[t_shift:t_shift+ ijk_max])

        b_max_val= np.nanmax(DV.values[b_shift:b_shift+ ijk_max])
        bk_max_val = np.nanargmax(DV.values[b_shift:b_shift+ ijk_max])
        b_min_val = np.nanmin(DV.values[b_shift:b_shift+ ijk_max])
        bk_min_val = np.nanargmin(DV.values[b_shift:b_shift+ ijk_max])

        alpha_max_val= np.nanmax(DV.values[alpha_shift:alpha_shift+ ijk_max])
        alphak_max_val = np.nanargmax(DV.values[alpha_shift:alpha_shift+ ijk_max])
        alpha_min_val = np.nanmin(DV.values[alpha_shift:alpha_shift+ ijk_max])
        alphak_min_val = np.nanargmin(DV.values[alpha_shift:alpha_shift+ ijk_max])

        t_max_val= np.nanmax(DV.values[t_shift:t_shift+ ijk_max])
        tk_max_val = np.nanargmax(DV.values[t_shift:t_shift+ ijk_max])
        t_min_val = np.nanmin(DV.values[t_shift:t_shift+ ijk_max])
        tk_min_val = np.nanargmin(DV.values[t_shift:t_shift+ ijk_max])
        # if Pa.rank == 0:
        if 1==1:
            print('b val: ', b_max_val, bk_max_val, b_min_val, bk_min_val, b_nan_val, bk_nan_val)
            print('alpha val: ', v_max, vk_max, v_min, vk_min, v_nan, vk_nan)
            print('t val: ', t_max_val, tk_max_val, t_min_val, tk_min_val, t_nan_val, tk_nan_val)



        if 'qt' in PV_.name_index:
            qt_varshift = PV_.get_varshift(Gr_,'qt')
            ql_varshift = DV.get_varshift(Gr_, 'ql')

            s_max = np.nanmax(PV_.tendencies[s_varshift:qt_varshift])
            sk_max = np.nanargmax(PV_.tendencies[s_varshift:qt_varshift])
            s_min = np.nanmin(PV_.tendencies[s_varshift:qt_varshift])
            sk_min = np.nanargmin(PV_.tendencies[s_varshift:qt_varshift])
            qt_max = np.nanmax(PV_.tendencies[qt_varshift:(qt_varshift + ijk_max)])
            qtk_max = np.nanargmax(PV_.tendencies[qt_varshift:(qt_varshift + ijk_max)])
            qt_min = np.nanmin(PV_.tendencies[qt_varshift:(qt_varshift + ijk_max)])
            qtk_min = np.nanargmin(PV_.tendencies[qt_varshift:(qt_varshift + ijk_max)])

            s_nan = np.isnan(PV_.tendencies[s_varshift:qt_varshift]).any()
            sk_nan = np.argmax(PV_.tendencies[s_varshift:qt_varshift])
            qt_nan = np.isnan(PV_.tendencies[qt_varshift:(qt_varshift + ijk_max)]).any()
            qtk_nan = np.argmax(PV_.tendencies[qt_varshift:(qt_varshift + ijk_max)])

            s_max_val= np.nanmax(PV_.values[s_varshift:qt_varshift])
            sk_max_val = np.nanargmax(PV_.values[s_varshift:qt_varshift])
            s_min_val = np.nanmin(PV_.values[s_varshift:qt_varshift])
            sk_min_val = np.nanargmin(PV_.tendencies[s_varshift:qt_varshift])
            s_nan_val = np.isnan(PV_.values[s_varshift:qt_varshift]).any()
            sk_nan_val = np.argmax(PV_.values[s_varshift:qt_varshift])
            qt_max_val = np.nanmax(PV_.values[qt_varshift:(qt_varshift + ijk_max)])
            qtk_max_val = np.nanargmax(PV_.values[qt_varshift:(qt_varshift + ijk_max)])
            qt_min_val = np.nanmin(PV_.values[qt_varshift:(qt_varshift + ijk_max)])
            if qt_min_val < 0:
                # Pa.root_print('qt val negative')
                print('qt val negative')
            qtk_min_val = np.nanargmin(PV_.values[qt_varshift:(qt_varshift + ijk_max)])
            qt_nan_val = np.isnan(PV_.values[qt_varshift:(qt_varshift + ijk_max)]).any()
            qtk_nan_val = np.argmax(PV_.values[qt_varshift:(qt_varshift + ijk_max)])

            ql_max_val = np.nanmax(DV.values[ql_varshift:(ql_varshift + ijk_max)])
            ql_min_val = np.nanmin(DV.values[ql_varshift:(ql_varshift + ijk_max)])
            qlk_max_val = np.nanargmax(DV.values[ql_varshift:(ql_varshift + ijk_max)])
            qlk_min_val = np.nanargmin(DV.values[ql_varshift:(ql_varshift + ijk_max)])
            ql_nan_val = np.isnan(DV.values[ql_varshift:(ql_varshift + ijk_max)]).any()
            qlk_nan_val = np.argmax(DV.values[ql_varshift:(ql_varshift + ijk_max)])

            # if Pa.rank == 0:
            if 1==1:
                print('s tend: ', s_max, sk_max, s_min, sk_min, s_nan, sk_nan)
                print('s val: ', s_max_val, sk_max_val, s_min_val, sk_min_val, s_nan_val, sk_nan_val)
                print('qt tend: ', qt_max, qtk_max, qt_min, qtk_min, qt_nan, qtk_nan)
                print('qt val: ', qt_max_val, qtk_max_val, qt_min_val, qtk_min_val, qt_nan_val, qtk_nan_val)
                print('ql val: ', ql_max_val, qlk_max_val, ql_min_val, qlk_min_val, ql_nan_val, qlk_nan_val)
            #self.Pa.root_print('ql: ' + str(ql_max) + ', ' + str(ql_min))

        #for name in PV.name_index.keys():
            # with nogil:
            if 1 == 1:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            if np.isnan(PV_.values[s_varshift+ijk]):
                                sk_arr = np.append(sk_arr,ijk)
                            if np.isnan(PV_.values[qt_varshift+ijk]):
                                qtk_arr = np.append(qtk_arr,ijk)

            if np.size(sk_arr) > 1:
                # self.output_nan_array(sk_arr,'s',message, Pa)
                # if Pa.rank == 0:
                if 1==1:
                    print('sk_arr size: ', sk_arr.shape)
                    print('sk_arr:', sk_arr)
                    # self.output_nan_array()
            if np.size(qtk_arr) > 1:
                # self.output_nan_array(qtk_arr,'qt',message, Pa)
                # if Pa.rank == 0:
                if 1==1:
                    print('qtk_arr size: ', qtk_arr.shape)
                    print('qtk_arr: ', qtk_arr)

        else:
            s_max = np.nanmax(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            sk_max = np.nanargmax(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            s_min = np.nanmin(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            sk_min = np.nanargmin(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            s_nan = np.isnan(PV_.tendencies[s_varshift:(s_varshift + ijk_max)]).any()
            sk_nan = np.argmax(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])

            s_max_val= np.nanmax(PV_.values[s_varshift:(s_varshift + ijk_max)])
            sk_max_val = np.nanargmax(PV_.values[s_varshift:(s_varshift + ijk_max)])
            s_min_val = np.nanmin(PV_.values[s_varshift:(s_varshift + ijk_max)])
            sk_min_val = np.nanargmin(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            s_nan_val = np.isnan(PV_.values[s_varshift:(s_varshift + ijk_max)]).any()
            sk_nan_val = np.argmax(PV_.values[s_varshift:(s_varshift + ijk_max)])

            # if Pa.rank == 0:
            if 1==1:
                print('s tend: ', s_max, sk_max, s_min, sk_min, s_nan, sk_nan)
                print('s val: ', s_max_val, sk_max_val, s_min_val, sk_min_val, s_nan_val, sk_nan_val)


            if 1 == 1:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            if np.isnan(PV_.values[s_varshift+ijk]):
                                sk_arr = np.append(sk_arr,ijk)


            if np.size(sk_arr) > 1:
                # self.output_nan_array(sk_arr,'s',message, Pa)
                # if Pa.rank == 0:
                if 1==1:
                    print('sk_arr size: ', sk_arr.shape)
                    print('sk_arr:', sk_arr)


        return