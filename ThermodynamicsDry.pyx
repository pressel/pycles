cimport numpy as np
import numpy as np

cimport ParallelMPI
cimport Grid
cimport ReferenceState
cimport DiagnosticVariables


from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cdef extern from "entropies.h":
    inline double sd_c(double p0, double T) nogil


cdef extern from "thermodynamics_dry.h":
    inline double eos_c(double p0, double s) nogil
    inline double alpha_c(double p0, double T, double qt, double qv) nogil
    void eos_update(Grid.DimStruct *dims, double *pd, double *s, double *T,
                    double *alpha)
    void buoyancy_update(Grid.DimStruct * dims, double *alpha0, double *alpha,double *buoyancy,
                         double *wt)

cdef class ThermodynamicsDry:
    def __init__(self,namelist,LatentHeat LH, ParallelMPI.ParallelMPI Par):
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist,LH,Par)

        return

    cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Par):

        PV.add_variable('s','m/s',"sym","scalar",Par)

        #Initialize class member arrays
        DV.add_variables('buoyancy','--','sym',Par)
        DV.add_variables('alpha','--','sym',Par)
        DV.add_variables('temperature','K','sym',Par)
        DV.add_variables('buoyancy_frequency','1/s','sym',Par)

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

        cdef int buoyancy_shift = DV.get_varshift(Gr,'buoyancy')
        cdef int alpha_shift = DV.get_varshift(Gr,'alpha')
        cdef int t_shift = DV.get_varshift(Gr,'temperature')
        cdef int s_shift = PV.get_varshift(Gr,'s')
        cdef int w_shift  = PV.get_varshift(Gr,'w')


        eos_update(&Gr.dims,&RS.p0_half[0],&PV.values[s_shift],&DV.values[t_shift],&DV.values[alpha_shift])
        buoyancy_update(&Gr.dims,&RS.alpha0[0],&DV.values[alpha_shift],&DV.values[buoyancy_shift],&PV.tendencies[w_shift])


        return

    cpdef get_pv_star(self,t):
        return self.CC.LT.fast_lookup(t)


    cpdef get_lh(self,t):
        cdef double lam = self.Lambda_fp(t)
        return self.L_fp(lam,t)