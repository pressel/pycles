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


