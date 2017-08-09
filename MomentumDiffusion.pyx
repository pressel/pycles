#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Kinematics
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats

import numpy as np
cimport numpy as np

from FluxDivergence cimport momentum_flux_divergence

cdef extern from 'momentum_diffusion.h':
    cdef void compute_diffusive_flux_m(Grid.DimStruct *dims, double *strain_rate,
                                       double *viscosity, double *flux, double *rho0,
                                       double *rho0_half, Py_ssize_t i1, Py_ssize_t i2)
    cdef void compute_entropy_source(Grid.DimStruct *dims, double *viscosity,
                                     double *strain_rate_mag, double *temperature, double *entropy_tendency)

cdef class MomentumDiffusion:

    def __init__(self, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        DV.add_variables('viscosity', r'm^2s^{-1}', r'\nu_t', 'eddy viscosity', 'sym', Pa)
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.flux = np.zeros(
            (Gr.dims.dims *
             Gr.dims.npg *
             Gr.dims.dims,
             ),
            dtype=np.double,
            order='c')


        #Initialize output fields
        for i in xrange(Gr.dims.dims):
            NS.add_profile(PV.velocity_names_directional[i] + '_sgs_flux_z',Gr,Pa)

        NS.add_profile('sgs_visc_s_source_mean',Gr,Pa)
        NS.add_profile('sgs_visc_s_source_min',Gr,Pa)
        NS.add_profile('sgs_visc_s_source_max',Gr,Pa)

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, Kinematics.Kinematics Ke):

        cdef:
            Py_ssize_t i1
            Py_ssize_t i2
            Py_ssize_t shift_v1
            Py_ssize_t shift_vgrad1
            Py_ssize_t shift_vgrad2
            Py_ssize_t shift_flux
            Py_ssize_t count = 0
            Py_ssize_t visc_shift = DV.get_varshift(Gr, 'viscosity')
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift

        for i1 in xrange(Gr.dims.dims):
            shift_v1 = PV.velocity_directions[i1] * Gr.dims.npg
            for i2 in xrange(Gr.dims.dims):
                shift_flux = count * Gr.dims.npg

                # First we compute the flux
                compute_diffusive_flux_m(&Gr.dims, &Ke.strain_rate[shift_flux], &DV.values[visc_shift], &self.flux[shift_flux], &Rs.rho0[0], &Rs.rho0_half[0], i1, i2)
                momentum_flux_divergence(&Gr.dims, &Rs.alpha0[0], &Rs.alpha0_half[0], &self.flux[shift_flux], &PV.tendencies[shift_v1], i1, i2)

                count += 1

        if 's' in PV.name_index:
            s_shift = PV.get_varshift(Gr, 's')
            compute_entropy_source(&Gr.dims, &DV.values[visc_shift], &Ke.strain_rate_mag[0], &DV.values[temp_shift], &PV.tendencies[s_shift])
        return

    cpdef stats_io(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        '''
        Statistical output for MomentumDiffusion Class.
        :param Gr: Grid class
        :param PV: PrognosticVariables class
        :param DV: DiagnosticVariables class
        :param Ke: Kinematics class
        :param NS: NetCDFIO_Stats class
        :param Pa: ParallelMPI class
        :return:
        '''
        cdef:
            Py_ssize_t i,k, d = 2
            Py_ssize_t shift_flux
            double[:] tmp
            double [:] tmp_interp = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        # Output vertical fluxes
        for i in xrange(Gr.dims.dims):
            shift_flux = (i*Gr.dims.dims + d) * Gr.dims.npg
            tmp = Pa.HorizontalMean(Gr,&self.flux[shift_flux])
            if i<2:
                for k in xrange(Gr.dims.gw,Gr.dims.nlg[2]-Gr.dims.gw):
                    tmp_interp[k] = 0.5*(tmp[k-1]+tmp[k])
            else:
                tmp_interp[:] = tmp[:]
            NS.write_profile(PV.velocity_names_directional[i] + '_sgs_flux_z', tmp_interp[Gr.dims.gw:-Gr.dims.gw], Pa)

        # Output entropy source from resolved TKE dissipation
        cdef:
            double[:] data = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            Py_ssize_t visc_shift = DV.get_varshift(Gr, 'viscosity')
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')

        compute_entropy_source(&Gr.dims, &DV.values[visc_shift], &Ke.strain_rate_mag[0], &DV.values[temp_shift], &data[0])
        tmp = Pa.HorizontalMean(Gr, &data[0])
        NS.write_profile('sgs_visc_s_source_mean', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMaximum(Gr, &data[0])
        NS.write_profile('sgs_visc_s_source_max', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp = Pa.HorizontalMinimum(Gr, &data[0])
        NS.write_profile('sgs_visc_s_source_min', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        return




