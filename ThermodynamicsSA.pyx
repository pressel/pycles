#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport numpy as np
import numpy as np
cimport Lookup
cimport ParallelMPI
cimport Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport PrognosticVariables
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
from thermodynamic_functions cimport thetas_c
import cython
from NetCDFIO cimport NetCDFIO_Stats, NetCDFIO_Fields

cdef extern from "thermodynamics_sa.h":
    inline double alpha_c(double p0, double T, double qt, double qv) nogil
    void eos_c(Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double,double), double p0, double s, double qt, double* T, double* qv, double* qc ) nogil
    void eos_update(Grid.DimStruct* dims, Lookup.LookupStruct* LT, double (*lam_fp)(double), double (*L_fp)(double, double), double* p0, double* s, double* qt, double* T,
    double* qv, double* ql, double* qi, double* alpha )
    void buoyancy_update_sa(Grid.DimStruct* dims, double* alpha0, double* alpha, double* buoyancy, double* wt)
    void bvf_sa(Grid.DimStruct* dims, Lookup.LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), double* p0, double* T, double* qt, double* qv, double* theta_rho, double* bvf)

cdef extern from "thermodynamic_functions.h":
    inline double pd_c(double p0, double qt, double qv) nogil
    inline double pv_c(double p0, double qt, double qv) nogil

cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil
    inline double sc_c(double L, double T) nogil
cdef class ThermodynamicsSA:
    def __init__(self,namelist,LatentHeat LH, ParallelMPI.ParallelMPI Par):
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist,LH,Par)
        return


    cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        PV.add_variable('s','m/s',"sym","scalar",Pa)
        PV.add_variable('qt','kg/kg',"sym","scalar",Pa)

        #Initialize class member arrays
        DV.add_variables('buoyancy','--','sym',Pa)
        DV.add_variables('alpha','--','sym',Pa)
        DV.add_variables('temperature','K','sym',Pa)
        DV.add_variables('buoyancy_frequency','1/s','sym',Pa)
        DV.add_variables('qv','kg/kg','sym',Pa)
        DV.add_variables('ql','kg/kg','sym',Pa)
        DV.add_variables('qi','kg/kg','sym',Pa)
        DV.add_variables('theta_rho','K','sym',Pa)

        #Add statistical output
        NS.add_profile('thetas_mean',Gr,Pa)
        NS.add_profile('thetas_mean2',Gr,Pa)
        NS.add_profile('thetas_mean3',Gr,Pa)
        NS.add_profile('thetas_max',Gr,Pa)
        NS.add_profile('thetas_min',Gr,Pa)
        NS.add_ts('thetas_max',Gr,Pa)
        NS.add_ts('thetas_min',Gr,Pa)

        return

    cpdef entropy(self, double p0, double T, double qt, double ql, double qi):
        cdef:
            double qv = qt - ql - qi
            double qd = 1.0 - qt
            double pd = pd_c(p0, qt, qv)
            double pv = pv_c(p0, qt, qv)
            double Lambda = self.Lambda_fp(T)
            double L = self.L_fp(T, Lambda)

        return sd_c(pd,T) * (1.0 - qt) + sv_c(pv,T) * qt+ sc_c(L,T)*(ql + qi)

    cpdef alpha(self, double p0, double T, double qt, double qv):
        return alpha_c(p0, T, qt, qv)

    cpdef eos(self, double p0, double s, double qt):
        cdef:
            double T, qv, qc, ql, qi,lam
        eos_c(&self.CC.LT.LookupStructC,self.Lambda_fp,self.L_fp,p0,s,qt,&T,&qv,&qc)
        lam = self.Lambda_fp(T)
        ql = qc * lam
        qi = qc * (1.0 - lam)
        return T, ql, qi

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):

        #Get relevant variables shifts
        cdef:
            Py_ssize_t buoyancy_shift = DV.get_varshift(Gr,'buoyancy')
            Py_ssize_t alpha_shift = DV.get_varshift(Gr,'alpha')
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr,'qi')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t w_shift  = PV.get_varshift(Gr,'w')
            Py_ssize_t bvf_shift = DV.get_varshift(Gr, 'buoyancy_frequency')
            Py_ssize_t thr_shift = DV.get_varshift(Gr,'theta_rho')

        eos_update(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.p0_half[0],
                   &PV.values[s_shift], &PV.values[qt_shift], &DV.values[t_shift], &DV.values[qv_shift], &DV.values[ql_shift],
                   &DV.values[qi_shift],&DV.values[alpha_shift])

        buoyancy_update_sa(&Gr.dims,&RS.alpha0_half[0],&DV.values[alpha_shift],&DV.values[buoyancy_shift],&PV.tendencies[w_shift])

        bvf_sa(&Gr.dims, &self.CC.LT.LookupStructC, self.Lambda_fp, self.L_fp, &RS.p0_half[0], &DV.values[t_shift], &PV.values[qt_shift],&DV.values[qv_shift],&DV.values[thr_shift], &DV.values[bvf_shift])

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
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            double [:] data = np.empty((Gr.dims.npl,),dtype=np.double,order='c')

        #Add entropy potential temperature to 3d fields
        with nogil:
            count = 0
            for i in range(imin,imax):
                ishift = i * istride
                for j in range(jmin,jmax):
                    jshift = j * jstride
                    for k in range(kmin,kmax):
                        ijk = ishift + jshift + k
                        data[count] = thetas_c(PV.values[s_shift+ijk],PV.values[qt_shift+ijk])
                        count += 1
        NF.add_field('thetas')
        NF.write_field('thetas',data)
        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

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
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            double [:] data = np.empty((Gr.dims.npg,),dtype=np.double,order='c')
            double [:] tmp

        #Ouput profiles of thetas
        with nogil:
            count = 0
            for i in range(imin,imax):
                ishift = i * istride
                for j in range(jmin,jmax):
                    jshift = j * jstride
                    for k in range(kmin,kmax):
                        ijk = ishift + jshift + k
                        data[count] = thetas_c(PV.values[s_shift+ijk],PV.values[qt_shift +ijk])
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