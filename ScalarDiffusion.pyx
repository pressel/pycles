#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True


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

cdef extern from "scalar_diffusion.h":
    void compute_diffusive_flux(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *diffusivity,
                                double *scalar, double *flux, double dx, size_t d, Py_ssize_t scheme, double factor)
    void compute_qt_diffusion_s_source(Grid.DimStruct *dims, double *p0_half, double *alpha0, double *alpha0_half,
                                       double *flux, double *qt, double *qv, double *T, double *tendency, double (*lam_fp)(double),
                                       double (*L_fp)(double, double), double dx, Py_ssize_t d )

cdef class ScalarDiffusion:
    def __init__(self, namelist, LatentHeat LH, DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        DV.add_variables('diffusivity',r'm^2s^{-1}', 'D_t', 'eddy diffusivity', 'sym', Pa)
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp

        try:
            self.qt_entropy_source = namelist['diffusion']['qt_entropy_source']
        except:
            self.qt_entropy_source = False
            Pa.root_print('By default not including entropy source resulting from diffusion of qt!')

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        '''
        Initialization method for the scalar diffusion class. Initializes the flux array to zero and adds output profiles
        to the Statistics output profile.
        :param Gr: Grid class
        :param PV: PrognosticVariables class
        :param DV: DiagnosticVariables class
        :param NS: NetCDFIO_Stats class
        :param Pa: ParallelMPI class
        :return:
        '''
        self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')

        #Initialize output fields
        for i in xrange(PV.nv):
            if PV.var_type[i] == 1:
                NS.add_profile(PV.index_name[i] + '_sgs_flux_z',Gr,Pa)

        if self.qt_entropy_source:
            NS.add_profile('sgs_qt_s_source_mean',Gr,Pa)
            NS.add_profile('sgs_qt_s_source_min',Gr,Pa)
            NS.add_profile('sgs_qt_s_source_max',Gr,Pa)

        return

    cpdef update(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV):
        '''
        Update method for scalar diffusion class, based on a second order finite difference scheme. The method should
        only be called following a call to update method for the SGS class.
        :param Gr: Grid class
        :param RS: ReferenceState class
        :param PV: PrognosticVariables class
        :param DV: DiagnosticVariables class
        :return:
        '''

        cdef:
            Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            Py_ssize_t s_shift
            Py_ssize_t qt_shift
            Py_ssize_t t_shift
            Py_ssize_t qv_shift
            Py_ssize_t n_qt = -9999
            Py_ssize_t n_e = -9999
            Py_ssize_t d, i ,scalar_shift, scalar_count = 0, flux_shift
            Py_ssize_t diff_shift_n = DV.get_varshift(Gr,'diffusivity')
            double flux_factor = 1.0

        if 'qt' in PV.name_index:
            n_qt = PV.name_index['qt']
            qt_shift = PV.get_varshift(Gr,'qt')
            t_shift = DV.get_varshift(Gr,'temperature')
            qv_shift = DV.get_varshift(Gr,'qv')
        if 'e' in PV.name_index:
            n_e = PV.name_index['e']


        for i in xrange(PV.nv):
            #Only compute fluxes for prognostic variables here
            if PV.var_type[i] == 1:
                scalar_shift = i * Gr.dims.npg
                if i == n_e:
                    diff_shift_n = DV.get_varshift(Gr,'viscosity')
                    flux_factor = 2.0
                else:
                    diff_shift_n = DV.get_varshift(Gr,'viscosity')
                    flux_factor = 1.0
                for d in xrange(Gr.dims.dims):

                    flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d* Gr.dims.npg


                    compute_diffusive_flux(&Gr.dims,&RS.rho0[0],&RS.rho0_half[0],
                                           &DV.values[diff_shift],&PV.values[scalar_shift],
                                           &self.flux[flux_shift],Gr.dims.dx[d],d,2, flux_factor)

                    scalar_flux_divergence(&Gr.dims,&RS.alpha0[0],&RS.alpha0_half[0],
                                           &self.flux[flux_shift],&PV.tendencies[scalar_shift],Gr.dims.dx[d],d)

                    if i == n_qt and self.qt_entropy_source:
                        s_shift = PV.get_varshift(Gr,'s')
                        compute_qt_diffusion_s_source(&Gr.dims, &RS.p0_half[0], &RS.alpha0[0],&RS.alpha0_half[0],
                                                      &self.flux[flux_shift],&PV.values[qt_shift], &DV.values[qv_shift],
                                                      &DV.values[t_shift],&PV.tendencies[s_shift],self.Lambda_fp,
                                                      self.L_fp,Gr.dims.dx[d],d)
                scalar_count += 1

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        '''
        Statistical output for ScalarDiffusion class.
        :param Gr: Grid class
        :param RS: ReferenceState class
        :param PV: PrognosticVariables class
        :param DV: DiagnosticVariables class
        :param NS: NetCDFIO_Stats class
        :param Pa: ParallelMPI class
        :return:
        '''

        cdef:
            Py_ssize_t d
            Py_ssize_t i
            Py_ssize_t k
            double[:] data = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double[:] tmp
            double [:] tmp_interp = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

            Py_ssize_t s_shift
            Py_ssize_t qt_shift
            Py_ssize_t t_shift
            Py_ssize_t qv_shift
            Py_ssize_t scalar_count = 0

        if 'qt' in PV.name_index:
            qt_shift = PV.get_varshift(Gr,'qt')
            t_shift = DV.get_varshift(Gr,'temperature')
            qv_shift = DV.get_varshift(Gr,'qv')

        #Output vertical component of SGS scalar fluxes
        d = 2
        for i in xrange(PV.nv):
            if PV.var_type[i] == 1:
                flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d* Gr.dims.npg
                tmp = Pa.HorizontalMean(Gr, &self.flux[flux_shift])
                for k in xrange(Gr.dims.gw,Gr.dims.nlg[2]-Gr.dims.gw):
                    tmp_interp[k] = 0.5*(tmp[k-1]+tmp[k])
                NS.write_profile(PV.index_name[i] + '_sgs_flux_z', tmp_interp[Gr.dims.gw:-Gr.dims.gw], Pa)
                scalar_count += 1

        if self.qt_entropy_source:
            s_shift = PV.get_varshift(Gr,'s')
            #Ouput entropy source term from qt diffusion
            scalar_count = 0
            if 'qt' in PV.name_index:
                for i in xrange(PV.nv):
                    if PV.var_type[i] == 1:
                        if PV.index_name[i] == 'qt':
                            break
                        scalar_count += 1

                for d in xrange(Gr.dims.dims):
                    flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d* Gr.dims.npg

                    compute_qt_diffusion_s_source(&Gr.dims, &RS.p0_half[0], &RS.alpha0[0],&RS.alpha0_half[0],
                                                  &self.flux[flux_shift],&PV.values[qt_shift], &DV.values[qv_shift],
                                                  &DV.values[t_shift],&data[0],self.Lambda_fp,
                                                      self.L_fp,Gr.dims.dx[d],d)

                tmp = Pa.HorizontalMean(Gr, &data[0])
                NS.write_profile('sgs_qt_s_source_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
                tmp = Pa.HorizontalMaximum(Gr, &data[0])
                NS.write_profile('sgs_qt_s_source_max', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
                tmp = Pa.HorizontalMinimum(Gr, &data[0])
                NS.write_profile('sgs_qt_s_source_min', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
            else:
                tmp = Pa.HorizontalMean(Gr, &data[0])
                NS.write_profile('sgs_qt_s_source_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
                tmp = Pa.HorizontalMaximum(Gr, &data[0])
                NS.write_profile('sgs_qt_s_source_max', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
                tmp = Pa.HorizontalMinimum(Gr, &data[0])
                NS.write_profile('sgs_qt_s_source_min', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        return
