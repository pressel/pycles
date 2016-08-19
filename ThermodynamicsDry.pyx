#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport numpy as np
import numpy as np

cimport ParallelMPI
cimport Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport PrognosticVariables
cimport Thermodynamics
from NetCDFIO cimport NetCDFIO_Fields, NetCDFIO_Stats
from thermodynamic_functions cimport thetas_c
import cython

from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cdef extern from "entropies.h":
    inline double sd_c(double p0, double T) nogil


cdef extern from "thermodynamics_dry.h":
    inline double eos_c(double p0, double s) nogil
    inline double alpha_c(double p0, double T, double qt, double qv) nogil
    void eos_update(Grid.DimStruct *dims, double *pd, double *s, double *T,
                    double *alpha)
    void buoyancy_update(Grid.DimStruct *dims, double *alpha0, double *alpha,double *buoyancy,
                         double *wt)
    void bvf_dry(Grid.DimStruct* dims,  double* p0, double* T, double* theta, double* bvf)


cdef class ThermodynamicsDry:
    def __init__(self,namelist,LatentHeat LH, ParallelMPI.ParallelMPI Pa):
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist,LH,Pa)

        return

    cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        PV.add_variable('s','m/s',"sym","scalar",Pa)

        #Initialize class member arrays
        DV.add_variables('buoyancy','--','sym',Pa)
        DV.add_variables('alpha','--','sym',Pa)
        DV.add_variables('temperature','K','sym',Pa)
        DV.add_variables('buoyancy_frequency','1/s','sym',Pa)
        DV.add_variables('theta','K','sym',Pa)


        #Add statistical output
        NS.add_profile('thetas_mean',Gr,Pa)
        NS.add_profile('thetas_mean2',Gr,Pa)
        NS.add_profile('thetas_mean3',Gr,Pa)
        NS.add_profile('thetas_max',Gr,Pa)
        NS.add_profile('thetas_min',Gr,Pa)
        NS.add_ts('thetas_max',Gr,Pa)
        NS.add_ts('thetas_min',Gr,Pa)

        return

    cpdef entropy(self,double p0, double T,double qt, double ql, double qi):
        qt = 0.0
        ql = 0.0
        qi = 0.0
        return sd_c(p0,T)


    cpdef eos(self,double p0, double s, double qt):
        ql = 0.0
        qi = 0.0
        return eos_c(p0,s), ql, qi


    cpdef alpha(self, double p0, double T, double qt, double qv):
        qv = 0.0
        qt = 0.0
        return alpha_c(p0,T,qv,qt)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):

        cdef Py_ssize_t buoyancy_shift = DV.get_varshift(Gr,'buoyancy')
        cdef Py_ssize_t alpha_shift = DV.get_varshift(Gr,'alpha')
        cdef Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
        cdef Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
        cdef Py_ssize_t w_shift  = PV.get_varshift(Gr,'w')
        cdef Py_ssize_t theta_shift = DV.get_varshift(Gr,'theta')
        cdef Py_ssize_t bvf_shift = DV.get_varshift(Gr,'buoyancy_frequency')


        eos_update(&Gr.dims,&RS.p0_half[0],&PV.values[s_shift],&DV.values[t_shift],&DV.values[alpha_shift])
        buoyancy_update(&Gr.dims,&RS.alpha0_half[0],&DV.values[alpha_shift],&DV.values[buoyancy_shift],&PV.tendencies[w_shift])
        bvf_dry(&Gr.dims,&RS.p0_half[0],&DV.values[t_shift],&DV.values[theta_shift],&DV.values[bvf_shift])
        # __
        self.debug_tend('Thermodynamics Dry: ',PV,DV,Gr)
        #  __
        return

    cpdef get_pv_star(self,t):
        return self.CC.LT.fast_lookup(t)


    cpdef get_lh(self,t):
        cdef double lam = self.Lambda_fp(t)
        return self.L_fp(lam,t)

    cpdef write_fields(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Fields NF, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i,j,k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t count
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            double [:] data = np.empty((Gr.dims.npl,),dtype=np.double,order='c')

        #Add entropy potential temperature to 3d fields
        with nogil:
            count = 0
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        data[count] = thetas_c(PV.values[s_shift+ijk],0.0)
                        count += 1
        NF.add_field('thetas')
        NF.write_field('thetas',data)
        print(np.amax(data),np.amin(data))
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i,j,k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = 0
            Py_ssize_t jmin = 0
            Py_ssize_t kmin = 0
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t kmax = Gr.dims.nlg[2]
            Py_ssize_t count
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            double [:] data = np.empty((Gr.dims.npg,),dtype=np.double,order='c')
            double [:] tmp

        #Add entropy potential temperature to 3d fields
        with nogil:
            count = 0
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        data[count] = thetas_c(PV.values[s_shift+ijk],0.0)
                        count += 1

        #Compute and write mean
        tmp = Pa.HorizontalMean(Gr,&data[0])
        NS.write_profile('thetas_mean',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        #Compute and write mean of squres
        tmp = Pa.HorizontalMeanofSquares(Gr,&data[0],&data[0])
        NS.write_profile('thetas_mean2',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        #Compute and write mean of cubes
        tmp = Pa.HorizontalMeanofCubes(Gr,&data[0],&data[0],&data[0])
        NS.write_profile('thetas_mean3',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        #Compute and write maxes
        tmp = Pa.HorizontalMaximum(Gr,&data[0])
        NS.write_profile('thetas_max',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_ts('thetas_max',np.amax(tmp[Gr.dims.gw:-Gr.dims.gw]),Pa)

        #Compute and write mins
        tmp = Pa.HorizontalMinimum(Gr,&data[0])
        NS.write_profile('thetas_min',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_ts('thetas_min',np.amin(tmp[Gr.dims.gw:-Gr.dims.gw]),Pa)

        return






# _______________
    cpdef debug_tend(self, message, PrognosticVariables.PrognosticVariables PV_, DiagnosticVariables.DiagnosticVariables DV,
                     Grid.Grid Gr_):

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
            Py_ssize_t imax = Gr_.dims.nlg[0]-1
            Py_ssize_t jmax = Gr_.dims.nlg[1]-1
            Py_ssize_t kmax = Gr_.dims.nlg[2]-1
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



        # if np.isnan(PV_.tendencies[0:s_varshift]).any() or np.isnan(PV_.values[0:s_varshift]).any():
        # if np.isnan(PV_.tendencies[s_varshift:s_varshift+ijk_max]).any() or np.isnan(PV_.values[s_varshift:s_varshift+ijk_max]).any():
        if np.isnan(PV_.tendencies).any() or np.isnan(PV_.values).any():
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

            w_max_val= np.nanmax(PV_.values[w_varshift:s_varshift])
            wk_max_val = np.nanargmax(PV_.values[w_varshift:s_varshift])
            w_min_val = np.nanmin(PV_.values[w_varshift:s_varshift])
            wk_min_val = np.nanargmin(PV_.tendencies[w_varshift:s_varshift])

            u_nan = np.isnan(PV_.tendencies[u_varshift:v_varshift]).any()
            uk_nan = np.argmax(PV_.tendencies[u_varshift:v_varshift])
            v_nan = np.isnan(PV_.tendencies[v_varshift:w_varshift]).any()
            vk_nan = np.argmax(PV_.tendencies[v_varshift:w_varshift])
            w_nan = np.isnan(PV_.tendencies[w_varshift:s_varshift]).any()
            wk_nan = np.argmax(PV_.tendencies[w_varshift:s_varshift])
            w_nan_val = np.isnan(PV_.values[w_varshift:s_varshift]).any()
            wk_nan_val = np.argmax(PV_.values[w_varshift:s_varshift])

            print(message, 'debugging (max, min, nan): ')
            print('shifts', u_varshift, v_varshift, w_varshift, s_varshift, 'Gr.npg', (imax+2*gw)*(jmax+2*gw)*(kmax+2*gw))
            print('u tend: ', u_max, uk_max, u_min, uk_min, u_nan, uk_nan)
            print('v tend: ', v_max, vk_max, v_min, vk_min, v_nan, vk_nan)
            print('w tend: ', w_max, wk_max, w_min, wk_min, w_nan, wk_nan)
            print('w val: ', w_max_val, wk_max_val, w_min_val, wk_min_val, w_nan_val, wk_nan_val)


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

            print('s tend: ', s_max, sk_max, s_min, sk_min, s_nan, sk_nan)
            print('s val: ', s_max_val, sk_max_val, s_min_val, sk_min_val, s_nan_val, sk_nan_val)


        # if np.isnan(DV.values).any():
            # buoyancy_update(&Gr.dims,&RS.alpha0_half[0],&DV.values[alpha_shift],&DV.values[buoyancy_shift],&PV.tendencies[w_shift])

            b_nan_val = np.isnan(DV.values[b_shift:b_shift+ ijk_max]).any()
            alpha_nan_val = np.isnan(DV.values[alpha_shift:alpha_shift+ ijk_max]).any()
            t_nan_val = np.isnan(DV.values[t_shift:t_shift+ ijk_max]).any()

            bk_nan_val = np.argmax(DV.values[b_shift:b_shift+ ijk_max])
            alphak_nan_val = np.argmax(DV.values[alpha_shift:alpha_shift+ ijk_max])
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

            print('b val: ', b_max_val, bk_max_val, b_min_val, bk_min_val, b_nan_val, bk_nan_val)
            print('alpha val: ', v_max, vk_max, v_min, vk_min, v_nan, vk_nan)
            print('t val: ', t_max_val, tk_max_val, t_min_val, tk_min_val, t_nan_val, tk_nan_val)

        return