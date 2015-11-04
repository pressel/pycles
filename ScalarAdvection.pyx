#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
cimport DiagnosticVariables
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats

from FluxDivergence cimport scalar_flux_divergence

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

    cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')

        #Initialize output fields
        for i in xrange(PV.nv):
            if PV.var_type[i] == 1:
                NS.add_profile(PV.index_name[i] + '_flux_z',Gr,Pa)


        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs,PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t d, i, vel_shift,scalar_shift, scalar_count = 0, flux_shift
        cdef Py_ssize_t order1 = 1

        for i in xrange(PV.nv): #Loop over the prognostic variables
            if PV.var_type[i] == 1: #Only compute advection if variable i is a scalar
                scalar_shift = i * Gr.dims.npg
                #No rescaling of fluxes
                for d in xrange(Gr.dims.dims): #Loop over the cardinal direction
                    #The flux has a different shift since it is only for the scalars
                    flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d * Gr.dims.npg

                    #Make sure that we get the velocity components in the correct order
                    #Also check for a scalar-specific velocity (e.g. hydrometeor sedimentation)
                    sc_vel_name = PV.velocity_names_directional[d] + '_' + DV.index_name[i]
                    if sc_vel_name in DV.name_index:

                        vel_shift = DV.get_varshift(Gr, sc_vel_name)
                        compute_advective_fluxes_a(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&DV.values[vel_shift],
                                                   &PV.values[scalar_shift],&self.flux[flux_shift],d,order1)
                    else:
                        vel_shift = PV.velocity_directions[d]*Gr.dims.npg
                        compute_advective_fluxes_a(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&PV.values[vel_shift],
                                                   &PV.values[scalar_shift],&self.flux[flux_shift],d,self.order)

                    scalar_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[flux_shift],
                                            &PV.tendencies[scalar_shift],Gr.dims.dx[d],d)


                scalar_count += 1

        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t scalar_count =  0, i, d = 2, flux_shift, k
            double[:] tmp
            double [:] tmp_interp = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        for i in xrange(PV.nv):
            if PV.var_type[i] == 1:
                flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d * Gr.dims.npg
                tmp = Pa.HorizontalMean(Gr, &self.flux[flux_shift])
                for k in xrange(Gr.dims.gw,Gr.dims.nlg[2]-Gr.dims.gw):
                    tmp_interp[k] = 0.5*(tmp[k-1]+tmp[k])
                NS.write_profile(PV.index_name[i] + '_flux_z', tmp_interp[Gr.dims.gw:-Gr.dims.gw], Pa)
                scalar_count += 1



        return
