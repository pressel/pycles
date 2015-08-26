cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats
import numpy as np
cimport numpy as np
from Thermodynamics cimport LatentHeat
from FluxDivergence cimport scalar_flux_divergence

import cython

#cdef extern from "advection_interpolation.h":
#    inline double interp_2(double phi, double phip1) nogil

cdef extern from "scalar_diffusion.h":
    void compute_diffusive_flux(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *diffusivity,
                                double *scalar, double *flux, double dx, size_t d, Py_ssize_t scheme)
    void compute_qt_diffusion_s_source(Grid.DimStruct *dims, double *p0_half, double *alpha0, double *alpha0_half,
                                       double *flux, double *qt, double *qv, double *T, double *tendency, double (*lam_fp)(double),
                                       double (*L_fp)(double, double), double dx, Py_ssize_t d )

cdef class ScalarDiffusion:
    def __init__(self, LatentHeat LH, DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        DV.add_variables('diffusivity','--','sym',Pa)
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')
        return

    cpdef update(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV):

        cdef:
            Py_ssize_t diff_shift = DV.get_nv('diffusivity')
            Py_ssize_t s_shift
            Py_ssize_t qt_shift
            Py_ssize_t t_shift
            Py_ssize_t qv_shift
            Py_ssize_t n_qt
            Py_ssize_t d, i ,scalar_shift, scalar_count = 0, flux_shift

        if 'qt' in PV.name_index:
            n_qt = PV.name_index['qt']
            s_shift = PV.get_varshift(Gr,'s')
            qt_shift = PV.get_varshift(Gr,'qt')
            t_shift = DV.get_varshift(Gr,'temperature')
            qv_shift = DV.get_varshift(Gr,'qv')

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
                        compute_qt_diffusion_s_source(&Gr.dims, &RS.p0_half[0], &RS.alpha0[0],&RS.alpha0_half[0],
                                                      &self.flux[flux_shift],&PV.values[qt_shift], &DV.values[qv_shift],
                                                      &DV.values[t_shift],&PV.tendencies[s_shift],self.Lambda_fp,
                                                      self.L_fp,Gr.dims.dx[d],d)
                scalar_count += 1

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return