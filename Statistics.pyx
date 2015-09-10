#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport DiagnosticVariables
cimport PrognosticVariables
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats
import cython
from thermodynamic_functions import exner, theta_rho
from libc.math cimport sqrt, log, fabs,atan, exp, fmax
cimport numpy as np
import numpy as np
include "parameters.pxi"

class CumulusStatistics:
    def __init__(self,Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        conditions = ['cloud','core']
        scalars = ['theta_rho','qt','ql']
        for cond in conditions:
            NS.add_profile('fraction_'+cond,Gr,Pa)
            NS.add_profile('w_'+cond,Gr,Pa)
            NS.add_profile('w2_'+cond,Gr,Pa)
            for scalar in scalars:
                NS.add_profile(scalar+'_'+cond,Gr,Pa)
                NS.add_profile(scalar+'2_'+cond,Gr,Pa)

    def stats_io(self, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            double [:] cloudmask = np.zeros(Gr.dims.npg,dtype=np.double, order='c')
            double [:] coremask = np.zeros(Gr.dims.npg,dtype=np.double, order='c')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t b_shift = DV.get_varshift(Gr,'buoyancy')
            Py_ssize_t i,j,k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            double [:] mean_buoyancy

        mean_buoyancy = Pa.HorizontalMean(Gr, &DV.values[b_shift])

        with nogil:
            count = 0
            for i in range(imin,imax):
                ishift = i * istride
                for j in range(jmin,jmax):
                    jshift = j * jstride
                    for k in range(kmin,kmax):
                        ijk = ishift + jshift + k
                        if DV.values[ql_shift+ijk] > 0.0:
                            cloudmask[ijk] = 1.0
                            if DV.values[b_shift+ijk] > mean_buoyancy[k]:
                                coremask[ijk] = 1.0

        cdef double [:] tmp
        cdef Py_ssize_t w_shift = PV.get_varshift(Gr,'w')
        cdef Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
        cdef Py_ssize_t thr_shift = DV.get_varshift(Gr,'theta_rho')

        #Compute the statistics
        #-fractions        # cdef Py_ssize_t ths_shift = DV.get_varshift(Gr,'thetas')
        tmp = Pa.HorizontalMean(Gr,&cloudmask[0])
        NS.write_profile('fraction_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMean(Gr,&coremask[0])
        NS.write_profile('fraction_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        #-w
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[w_shift],&cloudmask[0])
        NS.write_profile('w_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[w_shift],&PV.values[w_shift],&cloudmask[0])
        NS.write_profile('w2_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[w_shift],&coremask[0])
        NS.write_profile('w_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[w_shift],&PV.values[w_shift],&coremask[0])
        NS.write_profile('w2_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        #-qt
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qt_shift],&cloudmask[0])
        NS.write_profile('qt_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[qt_shift],&PV.values[qt_shift],&cloudmask[0])
        NS.write_profile('qt2_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &PV.values[qt_shift],&coremask[0])
        NS.write_profile('qt_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &PV.values[qt_shift],&PV.values[qt_shift],&coremask[0])
        NS.write_profile('qt2_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        #-ql
        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[ql_shift],&cloudmask[0])
        NS.write_profile('ql_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[ql_shift],&DV.values[ql_shift],&cloudmask[0])
        NS.write_profile('ql2_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[ql_shift],&coremask[0])
        NS.write_profile('ql_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[ql_shift],&DV.values[ql_shift],&coremask[0])
        NS.write_profile('ql2_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        #--theta_rho
        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[thr_shift],&cloudmask[0])
        NS.write_profile('theta_rho_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[thr_shift],&DV.values[thr_shift],&cloudmask[0])
        NS.write_profile('theta_rho2_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[thr_shift],&coremask[0])
        NS.write_profile('theta_rho_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[thr_shift],&DV.values[thr_shift],&coremask[0])
        NS.write_profile('theta_rho2_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        #--theta_s
        # tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[ths_shift],&cloudmask[0])
        # NS.write_profile('thetas_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        # tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[ths_shift],&DV.values[ths_shift],&cloudmask[0])
        # NS.write_profile('thetas2_cloud',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        # tmp = Pa.HorizontalMeanConditional(Gr, &DV.values[ths_shift],&coremask[0])
        # NS.write_profile('thetas_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
        # tmp = Pa.HorizontalMeanofSquaresConditional(Gr, &DV.values[ths_shift],&DV.values[ths_shift],&coremask[0])
        # NS.write_profile('thetas2_core',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        return









