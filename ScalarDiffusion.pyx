cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats
import numpy as np
cimport numpy as np

from FluxDivergence cimport scalar_flux_divergence

import cython

cdef extern from "advection_interpolation.h":
    inline double interp_2(double phi, double phip1) nogil

cdef extern from "scalar_diffusion.h":
    void second_order_diffusion(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *diffusivity,
                                double *scalar, double *flux, double dx, size_t d)

cdef class ScalarDiffusion:
    def __init__(self, DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        DV.add_variables('diffusivity','--','sym',Pa)
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')
        return

    cpdef update(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV):

        cdef:
            Py_ssize_t diff_shift = DV.get_nv('diffusivity')
            Py_ssize_t n_qt
            Py_ssize_t d, i ,scalar_shift, scalar_count = 0, flux_shift

        if 'qt' in PV.name_index:
            n_qt = PV.name_index['qt']

        for i in xrange(PV.nv):
            if PV.var_type[i] == 1:
                scalar_shift = i * Gr.dims.npg
                for d in xrange(Gr.dims.dims):
                    flux_shift = scalar_count * Gr.dims.npg + d * Gr.dims.npg

                    compute_diffusive_flux(&Gr.dims,&RS.rho0[0],&RS.rho0_half[0],
                                           &DV.values[diff_shift],&PV.values[scalar_shift],
                                           &self.flux[flux_shift],Gr.dims.dx[d],d,2)

                    scalar_flux_divergence(&Gr.dims,&RS.alpha0[0],&RS.alpha0_half[0],
                                           &self.flux[flux_shift],&PV.tendencies[scalar_shift],Gr.dims.dx[d],d)

                    if i == n_qt:
                        pass

                scalar_count += 1

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return

cdef compute_diffusive_flux(Grid.DimStruct *dims, double *rho0, double *rho0_half,
                            double *diffusivity, double *scalar, double* flux, double dx, Py_ssize_t d, Py_ssize_t scheme):

    if scheme == 2:
        second_order_diffusion(dims, rho0, rho0_half,
                            diffusivity, scalar, flux, dx, d)

    return