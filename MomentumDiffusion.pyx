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

import numpy as np
cimport numpy as np

import cython

from FluxDivergence cimport momentum_flux_divergence

cdef extern from 'momentum_diffusion.h':
    cdef void compute_diffusive_flux(Grid.DimStruct * dims, double * strain_rate, double * viscosity, double * flux, Py_ssize_t i1, Py_ssize_t i2)
    cdef void compute_entropy_source(Grid.DimStruct * dims, double * viscosity, double * strain_rate_mag, double * temperature, double * entropy_tendency)

cdef class MomentumDiffusion:

    def __init__(self, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        DV.add_variables('viscosity', '--', 'sym', Pa)
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        self.flux = np.zeros(
            (Gr.dims.dims *
             Gr.dims.npg *
             Gr.dims.dims,
             ),
            dtype=np.double,
            order='c')
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
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')

        for i1 in xrange(Gr.dims.dims):
            shift_v1 = PV.velocity_directions[i1] * Gr.dims.npg
            for i2 in xrange(Gr.dims.dims):
                shift_flux = count * Gr.dims.npg

                # First we compute the flux
                compute_diffusive_flux( & Gr.dims, & Ke.strain_rate[shift_flux], & DV.values[visc_shift], & self.flux[shift_flux], i1, i2)
                momentum_flux_divergence( & Gr.dims, & Rs.alpha0[0], & Rs.alpha0_half[0], & self.flux[shift_flux], & PV.tendencies[shift_v1], i1, i2)

                count += 1

        compute_entropy_source(& Gr.dims, & DV.values[visc_shift], & Ke.strain_rate_mag[0], & DV.values[temp_shift], & PV.tendencies[s_shift])
        return
