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
cimport Filter
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
from thermodynamic_functions cimport thetas_c, theta_c, thetali_c
import cython
from NetCDFIO cimport NetCDFIO_Stats, NetCDFIO_Fields
from libc.math cimport fmax, fmin, sqrt, log, exp, fabs

include "parameters.pxi"

cdef extern from "thermodynamics_sa.h":
    inline double alpha_c(double p0, double T, double qt, double qv) nogil
    void eos_c(Lookup.LookupStruct * LT, double(*lam_fp)(double), double(*L_fp)(double, double), double p0, double s, double qt, double * T, double * qv, double * ql, double * qi) nogil
    void buoyancy_update_sa(Grid.DimStruct * dims, double * alpha0, double * alpha, double * buoyancy, double * wt)
    void bvf_sa(Grid.DimStruct * dims, Lookup.LookupStruct * LT, double(*lam_fp)(double), double(*L_fp)(double, double), double * p0, double * T, double * qt, double * qv, double * theta_rho, double * bvf)
    void thetali_update(Grid.DimStruct *dims, double (*lam_fp)(double), double (*L_fp)(double, double), double*  p0, double*  T, double*  qt, double*  ql, double*  qi, double*  thetali)
    inline double temperature_no_ql(double pd, double pv, double s, double qt) nogil

cdef extern from "thermodynamic_functions.h":
    # Dry air partial pressure
    inline double pd_c(double p0, double qt, double qv) nogil
    # Water vapor partial pressure
    inline double pv_c(double p0, double qt, double qv) nogil
    inline double qv_star_c(double p0, double qt, double pv)nogil
    inline double alpha_c(double p0, double T, double  qt, double qv) nogil


cdef extern from "entropies.h":
    # Specific entropy of dry air
    inline double sd_c(double pd, double T) nogil
    # Specific entropy of water vapor
    inline double sv_c(double pv, double T) nogil
    # Specific entropy of condensed water
    inline double sc_c(double L, double T) nogil


cdef class ThermodynamicsSA_SGS:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):
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
        try:
            self.quadrature_order = namelist['sgs']['condensation']['quadrature_order']
        except:
            self.quadrature_order = 5

        try:
            self.c_variance = namelist['sgs']['condensation']['c_variance']
        except:
            self.c_variance = 0.2857

        try:
            self.use_scale_sim = namelist['sgs']['condensation']['scale_similarity_model']
        except:
            self.use_scale_sim = True




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

        self.s_variance  = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.qt_variance = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.qt_variance_clip = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.covariance  = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.correlation  = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        self.cloud_fraction  = np.zeros(Gr.dims.npg, dtype=np.double, order='c')



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
        NS.add_profile('s_sgs_variance', Gr, Pa)
        NS.add_profile('qt_sgs_variance', Gr, Pa)
        NS.add_profile('qt_sgs_variance_clip', Gr, Pa)
        NS.add_profile('sgs_covariance', Gr, Pa)
        NS.add_profile('sgs_correlation', Gr, Pa)

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


        NS.add_profile('cloud_fraction', Gr, Pa)
        NS.add_ts('cloud_fraction', Gr, Pa)
        NS.add_ts('cloud_top', Gr, Pa)
        NS.add_ts('cloud_base', Gr, Pa)
        NS.add_ts('lwp', Gr, Pa)

        # Initialize the filter operator class
        self.VarianceFilter = Filter.Filter(Gr, Pa)


        return

    cpdef entropy(self, double p0, double T, double qt, double ql, double qi):
        '''
        Provide a python rapper for the c function that computes the specific entropy
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

    cpdef compute_variances(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa, PrognosticVariables.PrognosticVariables PV):
        cdef:
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            double [:] qt_t = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] s_t = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] qt_T = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] s_T = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] leonard_Tt = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double [:] leonard_tg = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
            double little_test_factor = 2.0
            double big_test_factor = 4.0
            double c_sim = 1.0/((big_test_factor/little_test_factor)**0.667 - 1.0)
            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw




        print('c_sim = ', c_sim)


        if self.use_scale_sim:
            qt_t = self.VarianceFilter.spectral_2d(Gr,Pa, &PV.values[qt_shift], little_test_factor)
            s_t = self.VarianceFilter.spectral_2d(Gr,Pa, &PV.values[s_shift], little_test_factor)
            qt_T = self.VarianceFilter.spectral_2d(Gr,Pa, &PV.values[qt_shift], big_test_factor)
            s_T = self.VarianceFilter.spectral_2d(Gr,Pa, &PV.values[s_shift], big_test_factor)

            # First do qt variance, form the first part of the leonard terms that must be filtered
            with nogil:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            leonard_tg[ijk] =  PV.values[qt_shift + ijk] * PV.values[qt_shift + ijk]
                            leonard_Tt[ijk] =  qt_t[ijk] * qt_t[ijk]

            leonard_tg = self.VarianceFilter.spectral_2d(Gr,Pa, &leonard_tg[0], little_test_factor)
            leonard_Tt = self.VarianceFilter.spectral_2d(Gr,Pa, &leonard_Tt[0], big_test_factor)


            # Now get the variance
            with nogil:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            self.qt_variance[ijk] =  c_sim * (leonard_Tt[ijk]- qt_T[ijk] *qt_T[ijk]) - leonard_tg[ijk] + qt_t[ijk]*qt_t[ijk]
                            self.qt_variance[ijk] = fmax(self.qt_variance[ijk],0.0)


            # Now s variance, form the first part of the leonard terms that must be filtered
            with nogil:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            leonard_tg[ijk] =  PV.values[s_shift + ijk] * PV.values[s_shift + ijk]
                            leonard_Tt[ijk] =  s_t[ijk] * s_t[ijk]

            leonard_tg = self.VarianceFilter.spectral_2d(Gr,Pa, &leonard_tg[0], little_test_factor)
            leonard_Tt = self.VarianceFilter.spectral_2d(Gr,Pa, &leonard_Tt[0], big_test_factor)


            # Now get the variance
            with nogil:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            self.s_variance[ijk] =  c_sim * (leonard_Tt[ijk]- s_T[ijk] * s_T[ijk]) - leonard_tg[ijk] + s_t[ijk]* s_t[ijk]
                            self.s_variance[ijk] = fmax(self.s_variance[ijk],0.0)



            # Now co-variance, form the first part of the leonard terms that must be filtered
            with nogil:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            leonard_tg[ijk] =  PV.values[s_shift + ijk] * PV.values[qt_shift + ijk]
                            leonard_Tt[ijk] =  s_t[ijk] * qt_t[ijk]

            leonard_tg = self.VarianceFilter.spectral_2d(Gr,Pa, &leonard_tg[0], little_test_factor)
            leonard_Tt = self.VarianceFilter.spectral_2d(Gr,Pa, &leonard_Tt[0], big_test_factor)


            # Now get the variance
            with nogil:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            self.covariance[ijk] =  c_sim * (leonard_Tt[ijk]- qt_T[ijk] * s_T[ijk]) - leonard_tg[ijk] + qt_t[ijk]* s_t[ijk]
                            self.covariance[ijk] = fmax(fmin(self.s_variance[ijk]*self.qt_variance[ijk],self.covariance[ijk]),-self.s_variance[ijk]*self.qt_variance[ijk])




        else:
            compute_sgs_variance_gradient(&Gr.dims,  &PV.values[s_shift],  &self.s_variance[0], self.c_variance)
            compute_sgs_variance_gradient(&Gr.dims,  &PV.values[qt_shift], &self.qt_variance[0], self.c_variance)
            compute_sgs_covariance_gradient(&Gr.dims, &PV.values[s_shift],  &PV.values[qt_shift], &self.covariance[0], self.c_variance)


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

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

        self.compute_variances(Gr, Pa, PV)

        eos_update_SA_sgs(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.p0_half[0],  &PV.values[s_shift],
                            &self.s_variance[0], &PV.values[qt_shift], &self.qt_variance[0], &self.covariance[0], &DV.values[t_shift],
                            &DV.values[alpha_shift], &DV.values[qv_shift], &DV.values[ql_shift], &DV.values[qi_shift],
                            &self.cloud_fraction[0],  self.quadrature_order, &self.correlation[0], &self.qt_variance_clip[0])


        #update the Boundary conditions and ghost cells
        ndv = DV.name_index['temperature']
        DV.communicate_variable(Gr, Pa, ndv)

        ndv = DV.name_index['ql']
        DV.communicate_variable(Gr, Pa, ndv)

        ndv = DV.name_index['qv']
        DV.communicate_variable(Gr, Pa, ndv)

        ndv = DV.name_index['qi']
        DV.communicate_variable(Gr, Pa, ndv)

        ndv = DV.name_index['alpha']
        DV.communicate_variable(Gr, Pa, ndv)

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


        tmp = Pa.HorizontalMean(Gr, &self.s_variance[0])
        NS.write_profile('s_sgs_variance', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.qt_variance[0])
        NS.write_profile('qt_sgs_variance', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.covariance[0])
        NS.write_profile('sgs_covariance', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.qt_variance_clip[0])
        NS.write_profile('qt_sgs_variance_clip', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMean(Gr, &self.correlation[0])
        NS.write_profile('sgs_correlation', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


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

        tmp = Pa.HorizontalMean(Gr, &self.cloud_fraction[0])
        NS.write_profile('cloud_fraction', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


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


        # Initialize the z-pencil
        z_pencil.initialize(Gr, Pa, 2)
        ql_pencils =  z_pencil.forward_double( &Gr.dims, Pa, &DV.values[ql_shift])

        # Compute all or nothing cloud fraction
        ci = np.empty((z_pencil.n_local_pencils), dtype=np.double, order='c')
        with nogil:
            for pi in xrange(z_pencil.n_local_pencils):
                for k in xrange(kmin, kmax):
                    if ql_pencils[pi, k] > ql_threshold:
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
                    if ql_pencils[pi, k] > ql_threshold:
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


cdef eos_update_SA_sgs(Grid.DimStruct *dims, Lookup.LookupStruct *LT, double(*lam_fp)(double), double(*L_fp)(double, double),
                       double *p0, double *s, double *s_var, double *qt, double *qt_var, double *covar,
                       double *T, double *alpha, double *qv, double *ql, double *qi, double *cf, Py_ssize_t order,
                       double *correlation, double *qt_variance_clip):

    a, w = np.polynomial.hermite.hermgauss(order)
    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k, m_q, m1
        double [:] abscissas = a
        double [:] weights = w
        double  outer_int_ql, outer_int_qi, outer_int_T, outer_int_alpha, outer_int_cf
        double  inner_int_ql, inner_int_qi, inner_int_T, inner_int_alpha, inner_int_cf
        double s_hat, qt_hat, sd_s, sd_q, corr, mu_s_star, sigma_s_star
        double sqpi_inv = 1.0/sqrt(pi)
        double temp_m, alpha_m, qv_m, ql_m, qi_m
        double sqrt2 = sqrt(2.0)
        double sd_q_lim


    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k

                    sd_q = sqrt(qt_var[ijk])
                    sd_s = sqrt(s_var[ijk])
                    corr = fmax(fmin(covar[ijk]/fmax(sd_s*sd_q, 1e-13),1.0),-1.0)
                    correlation[ijk] = corr
                    # limit sd_q to prevent negative qt_hat
                    sd_q_lim = (1e-10 - qt[ijk])/(sqrt2 * abscissas[0])
                    sd_q = fmin(sd_q, sd_q_lim)
                    qt_variance_clip[ijk] = sd_q * sd_q
                    sigma_s_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_s
                    outer_int_alpha = 0.0
                    outer_int_T = 0.0
                    outer_int_ql = 0.0
                    outer_int_qi = 0.0
                    outer_int_cf = 0.0
                    for m_q in xrange(order):
                        qt_hat    = qt[ijk] + sqrt2 * sd_q * abscissas[m_q]
                        mu_s_star = s[ijk]  + sqrt2 * corr * sd_s * abscissas[m_q]
                        inner_int_T     = 0.0
                        inner_int_qi    = 0.0
                        inner_int_ql    = 0.0
                        inner_int_alpha = 0.0
                        inner_int_cf    = 0.0
                        for m_s in xrange(order):
                            s_hat = sqrt2 * sigma_s_star * abscissas[m_s] + mu_s_star

                            eos_c(LT, lam_fp, L_fp, p0[k], s_hat, qt_hat, &temp_m,  &qv_m, &ql_m, &qi_m)
                            alpha_m = alpha_c(p0[k], temp_m, qt_hat, qv_m)
                            inner_int_ql    += ql_m    * weights[m_s] * sqpi_inv
                            inner_int_qi    += qi_m    * weights[m_s] * sqpi_inv
                            inner_int_T     += temp_m  * weights[m_s] * sqpi_inv
                            inner_int_alpha += alpha_m * weights[m_s] * sqpi_inv
                            if ql_m  + qi_m > ql_threshold:
                                inner_int_cf += weights[m_s] * sqpi_inv
                        outer_int_ql    += inner_int_ql    * weights[m_q] * sqpi_inv
                        outer_int_qi    += inner_int_qi    * weights[m_q] * sqpi_inv
                        outer_int_T     += inner_int_T     * weights[m_q] * sqpi_inv
                        outer_int_alpha += inner_int_alpha * weights[m_q] * sqpi_inv
                        outer_int_cf    += inner_int_cf    * weights[m_q] * sqpi_inv
                    ql[ijk]    = outer_int_ql
                    qi[ijk]    = outer_int_qi
                    alpha[ijk] = outer_int_alpha
                    T[ijk]     = outer_int_T
                    cf[ijk]    = outer_int_cf
                    qv[ijk]    = qt[ijk] - ql[ijk] - qi[ijk]
    return



cdef compute_sgs_variance_gradient(Grid.DimStruct *dims,  double *s, double *s_var, double coeff):


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
        double delta2 = (dims.dx[0] * dims.dx[1] * dims.dx[2])**(2.0/3.0)
        double dsdx, dsdy, dsdz
        double dxi = 1.0/(2.0*dims.dx[0])
        double dyi = 1.0/(2.0*dims.dx[1])
        double dzi = 1.0/(2.0*dims.dx[2])

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    dsdx = (s[ijk + istride] - s[ijk - istride]) * dxi
                    dsdy = (s[ijk + jstride] - s[ijk - jstride]) * dyi
                    dsdz = (s[ijk + 1] - s[ijk - 1]) * dzi
                    s_var[ijk] = coeff * delta2 * (dsdx * dsdx + dsdy * dsdy + dsdz * dsdz)


    return



cdef compute_sgs_covariance_gradient(Grid.DimStruct *dims,  double *a, double *b, double *covar, double coeff):


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
        double delta2 = (dims.dx[0] * dims.dx[1] * dims.dx[2])**(2.0/3.0)
        double dadx, dady, dadz
        double dbdx, dbdy, dbdz
        double dxi = 1.0/(2.0*dims.dx[0])
        double dyi = 1.0/(2.0*dims.dx[1])
        double dzi = 1.0/(2.0*dims.dx[2])

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    dadx = (a[ijk + istride] - a[ijk - istride]) * dxi
                    dady = (a[ijk + jstride] - a[ijk - jstride]) * dyi
                    dadz = (a[ijk + 1] - a[ijk - 1]) * dzi
                    dbdx = (b[ijk + istride] - b[ijk - istride]) * dxi
                    dbdy = (b[ijk + jstride] - b[ijk - jstride]) * dyi
                    dbdz = (b[ijk + 1] - b[ijk - 1]) * dzi
                    covar[ijk] = coeff * delta2 * (dadx * dbdx + dady * dbdy + dadz * dbdz)

    return



