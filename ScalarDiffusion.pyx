cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI

import numpy as np
cimport numpy as np

from FluxDivergence cimport scalar_flux_divergence

import cython

cdef extern from "advection_interpolation.h":
    inline double interp_2(double phi, double phip1) nogil

cdef class ScalarDiffusion:
    def __init__(self, DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        DV.add_variables('diffusivity','--','sym',Pa)
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')
        return

    cpdef update(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV):


        cdef:
            Py_ssize_t diff_shift = DV.get_nv('diffusivity')
            Py_ssize_t d, i ,scalar_shift, scalar_count = 0, flux_shift

        for i in xrange(PV.nv):
            if PV.var_type[i] == 1:
                scalar_shift = i * Gr.dims.npg
                for d in xrange(Gr.dims.dims):
                    flux_shift = scalar_count * Gr.dims.npg + d * Gr.dims.npg

                    compute_diffusive_flux(&Gr.dims,&RS.alpha0[0],&RS.alpha0_half[0],
                                           &DV.values[diff_shift],&PV.values[scalar_shift],
                                           &self.flux[flux_shift],Gr.dims.dx[d],d,2)

                    scalar_flux_divergence(&Gr.dims,&RS.alpha0[0],&RS.alpha0_half[0],
                                           &self.flux[flux_shift],&PV.tendencies[scalar_shift],Gr.dims.dx[d],d)

                scalar_count += 1

        return

cdef compute_diffusive_flux(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                            double *diffusivity, double *scalar, double* flux, double dx, Py_ssize_t d, Py_ssize_t scheme):

    if scheme == 2:
        second_order(dims, alpha0, alpha0_half,
                            diffusivity, scalar, flux, dx, d)

    return

cdef second_order(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                            double *diffusivity, double *scalar, double* flux, double dx, Py_ssize_t d):

    cdef:
        Py_ssize_t imin = 0
        Py_ssize_t jmin = 0
        Py_ssize_t kmin = 0
        Py_ssize_t imax = dims.nlg[0] -1
        Py_ssize_t jmax = dims.nlg[1] -1
        Py_ssize_t kmax = dims.nlg[2] -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2];
        Py_ssize_t jstride = dims.nlg[2];
        Py_ssize_t ishift, jshift, ijk, i,j,k
        Py_ssize_t sp1
        double dxi = 1.0/dx

    if d==0:
        sp1 = istride
    elif d==1:
        sp1 = jstride
    else:
        sp1 = 1

    if d == 2:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    flux[ijk] = -interp_2(diffusivity[ijk],diffusivity[ijk+sp1]) * (scalar[ijk+sp1]-scalar[ijk])/alpha0[k]*dxi

    else:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    flux[ijk] = -interp_2(diffusivity[ijk],diffusivity[ijk+sp1])*(scalar[ijk+sp1]-scalar[ijk])/alpha0_half[k]*dxi

    return