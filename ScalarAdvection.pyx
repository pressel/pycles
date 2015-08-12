cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState

from FluxDivergence cimport scalar_flux_divergence_adv

import numpy as np
cimport numpy as np

import cython

cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *velocity, double *scalar, double* flux, int d, int scheme) nogil
cdef class ScalarAdvection:
    def __init__(self, namelist, ParallelMPI.ParallelMPI Pa):
        try:
            self.order = namelist['scalar_transport']['order']
        except:
            Pa.root_print('scalar_transport order not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()
            Pa.kill()

        return

    cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):
        self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')
        return

    cpdef update_cython(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs,PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):

        cdef:
            long d, i, vel_shift,scalar_shift, scalar_count = 0, flux_shift

        for i in xrange(PV.nv): #Loop over the prognostic variables
            if PV.var_type[i] == 1: #Only compute advection if variable i is a scalar
                scalar_shift = i * Gr.dims.npg
                for d in xrange(Gr.dims.dims): #Loop over the cardinal direction
                    #The flux has a different shift since it is only for the scalars
                    flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d * Gr.dims.npg

                    #Make sure that we get the velocity components in the correct order
                    vel_shift = PV.velocity_directions[d]*Gr.dims.npg

                    compute_advective_fluxes_a(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&PV.values[vel_shift],
                                             &PV.values[scalar_shift],&self.flux[flux_shift],d,self.order)

                    scalar_flux_divergence_adv(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[flux_shift],
                                            &PV.tendencies[scalar_shift],Gr.dims.dx[d],d)
                scalar_count += 1

        return
