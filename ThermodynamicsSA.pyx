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
    void eos_update(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double), double *p0, double *s, double *qt, double *T,
                    double * qv, double * ql, double * qi, double * alpha)
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
        eos_c(&self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, p0, s, qt, &T, &qv, &ql, &qi)
        return T, ql, qi

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):

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


        eos_update(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.p0_half[0],
                    &PV.values[s_shift], &PV.values[qt_shift], &DV.values[t_shift], &DV.values[qv_shift], &DV.values[ql_shift],
                    &DV.values[qi_shift], &DV.values[alpha_shift])

        buoyancy_update_sa(&Gr.dims, &RS.alpha0_half[0], &DV.values[alpha_shift], &DV.values[buoyancy_shift], &PV.tendencies[w_shift])

        bvf_sa( &Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[thr_shift], &DV.values[bvf_shift])

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
                        cb = fmin(cb, Gr.zp_half[gw + k])
                        ct = fmax(ct, Gr.zp_half[gw + k])

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
