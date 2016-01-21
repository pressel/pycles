#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from NetCDFIO cimport NetCDFIO_Stats


import numpy as np
cimport numpy as np

cdef extern from "momentum_advection.h":
    void compute_advective_tendencies_m(Grid.DimStruct *dims, double *rho0, double *rho0_half,
                                    double *alpha0, double *alpha0_half, double *vel_advected,
                                    double *vel_advecting, double *tendency, Py_ssize_t d_advected,
                                    Py_ssize_t d_advecting, Py_ssize_t scheme) nogil
cdef class MomentumAdvection:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        try:
            self.order = namelist['momentum_transport']['order']
        except:
            Pa.root_print(
                'momentum_transport order not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()

        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        #for i in xrange(Gr.dims.dims):
        #    NS.add_profile(PV.velocity_names_directional[i] + '_flux_z',Gr,Pa)

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


        for i_advected in xrange(Gr.dims.dims):
            # Compute the shift to the starting location of the advected
            # velocity in the PV values array
            shift_advected = PV.velocity_directions[i_advected] * Gr.dims.npg
            for i_advecting in xrange(Gr.dims.dims):

                # Compute the shift to the starting location of the advecting
                # velocity in the PV values array
                shift_advecting = PV.velocity_directions[
                    i_advecting] * Gr.dims.npg

                # Compute the fluxes
                compute_advective_tendencies_m(&Gr.dims, &Rs.rho0[0], &Rs.rho0_half[0], &Rs.alpha0[0], &Rs.alpha0_half[0],
                                            &PV.values[shift_advected], &PV.values[shift_advecting],
                                           &PV.tendencies[shift_advected], i_advected, i_advecting, self.order)
        return


    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        # cdef:
        #     Py_ssize_t i_advected, i_advecting = 2, shift_flux, k
        #     double[:] tmp
        #     double [:] tmp_interp = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        #
        #
        # for i_advected in xrange(Gr.dims.dims):
        #     shift_flux = i_advected * Gr.dims.dims* Gr.dims.npg + i_advecting * Gr.dims.npg
        #     tmp = Pa.HorizontalMean(Gr, &self.flux[shift_flux])
        #     if i_advected < 2:
        #         for k in xrange(Gr.dims.gw,Gr.dims.nlg[2]-Gr.dims.gw):
        #             tmp_interp[k] = 0.5*(tmp[k-1]+tmp[k])
        #     else:
        #         tmp_interp[:] = tmp[:]
        #     NS.write_profile(PV.velocity_names_directional[i_advected] + '_flux_z', tmp_interp[Gr.dims.gw:-Gr.dims.gw], Pa)

        return

    cpdef double [:, :, :] get_flux(self, Py_ssize_t i_advected, Py_ssize_t i_advecting, Grid.Grid Gr):
        '''
        Returns momentum flux tensor component.
        :param i_advected: direction of advection velocity
        :param i_advecting:  direction of advecting velocity
        :param Gr: Grid class
        :return: memory view type double rank-3
        '''
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
