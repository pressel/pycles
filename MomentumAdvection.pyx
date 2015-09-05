#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from FluxDivergence cimport momentum_flux_divergence

import numpy as np
cimport numpy as np

import cython

cdef extern from "momentum_advection.h":
    void compute_advective_fluxes_m(Grid.DimStruct * dims, double * rho0, double * rho0_half, double * vel_advected, double * vel_advecting,
                                    double * flux, Py_ssize_t d_advected, Py_ssize_t d_advecting, Py_ssize_t scheme) nogil
cdef class MomentumAdvection:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        try:
            self.order = namelist['momentum_transport']['order']
        except:
            Pa.root_prPy_ssize_t(
                'momentum_transport order not given in namelist')
            Pa.root_prPy_ssize_t('Killing simulation now!')
            Pa.kill()
            Pa.kill()

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):
        self.flux = np.zeros(
            (PV.nv_velocities *
             Gr.dims.npg *
             Gr.dims.dims,
             ),
            dtype=np.double,
            order='c')
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i_advecting  # Direction of advecting velocity
            Py_ssize_t i_advected  # Direction of momentum component
            # Shift to beginning of momentum (velocity) component in the
            # PV.values array
            Py_ssize_t shift_advected
            # Shift to beginning of advecting velocity componentin the
            # PV.values array
            Py_ssize_t shift_advecting
            # Shift to the i_advecting, i_advected flux component in the
            # self.flux array
            Py_ssize_t shift_flux

        for i_advected in xrange(Gr.dims.dims):
            # Compute the shift to the starting location of the advected
            # velocity in the PV values array
            shift_advected = PV.velocity_directions[i_advected] * Gr.dims.npg
            for i_advecting in xrange(Gr.dims.dims):

                # Compute the shift to the starting location of the advecting
                # velocity in the PV values array
                shift_advecting = PV.velocity_directions[
                    i_advecting] * Gr.dims.npg

                # Compute the shift to the starting location of the advecting
                # flux.
                shift_flux = i_advected * Gr.dims.dims * \
                    Gr.dims.npg + i_advecting * Gr.dims.npg

                # Compute the fluxes
                compute_advective_fluxes_m(& Gr.dims, & Rs.rho0[0], & Rs.rho0_half[0],
                                            & PV.values[shift_advected], & PV.values[shift_advecting], & self.flux[shift_flux],
                                            i_advected, i_advecting, self.order)
                # Compute flux divergence
                momentum_flux_divergence(& Gr.dims, & Rs.alpha0[0], & Rs.alpha0_half[0], & self.flux[shift_flux],
                                          & PV.tendencies[shift_advected], i_advected, i_advecting)
        return

    @cython.boundscheck(False)  # Turn off numpy array index bounds checking
    @cython.wraparound(False)  # Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef double[:, :, :] get_flux(self, Py_ssize_t i_advected, Py_ssize_t i_advecting, Grid.Grid Gr):
        cdef:
            Py_ssize_t shift_flux = i_advected * Gr.dims.dims * Gr.dims.npg + i_advecting * Gr.dims.npg
            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]

            Py_ssize_t imin = 0
            Py_ssize_t jmin = 0
            Py_ssize_t kmin = 0

            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t kmax = Gr.dims.nlg[2]

            cdef double[:, :, :] return_flux = np.empty((Gr.dims.nlg[0], Gr.dims.nlg[1], Gr.dims.nlg[2]), dtype=np.double, order='c')
            cdef double[:] flux = self.flux

        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        return_flux[
                            i,
                            j,
                            k] = flux[
                            shift_flux +
                            ishift +
                            jshift +
                            k]
        return return_flux
