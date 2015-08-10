cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState

from FluxDivergence cimport scalar_flux_divergence

import numpy as np
cimport numpy as np

import cython

cdef extern from "advection_interpolation.h":
    inline double interp_2(double phi, double phip1) nogil
    inline double interp_4(double phim1, double phi, double phip1, double phip2) nogil
cdef extern from "scalar_advection.h":
    void compute_advective_fluxes(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *velocity, double *scalar, double* flux, int d, int scheme) nogil
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

                    compute_advective_fluxes(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&PV.values[vel_shift],
                                             &PV.values[scalar_shift],&self.flux[flux_shift],d,self.order)

                    scalar_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[flux_shift],
                                            &PV.tendencies[scalar_shift],Gr.dims.dx[d],d)
                scalar_count += 1


                #tendency_array = PV.get_tendency_array('s',Gr)
                #import pylab as plt
                #plt.contour(tendency_array[:,7,:],12)
                #plt.show()
                #import sys; sys.exit()
                #print(np.max(np.array(PV.tendencies)),np.min(np.array(PV.tendencies)))

        return

# cdef compute_advective_fluxes_cython(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *velocity, double *scalar, double* flux, int d, int scheme):
#
#     if scheme == 2:
#         second_order(dims,alpha0,alpha0_half,velocity,scalar,flux,d)
#     elif scheme == 4:
#         fourth_order(dims,alpha0,alpha0_half,velocity,scalar,flux,d)
#
#     return
#
#
# @cython.boundscheck(False)  #Turn off numpy array index bounds checking
# @cython.wraparound(False)   #Turn off numpy array wrap around indexing
# @cython.cdivision(True)
# cdef second_order_cython(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *velocity, double *scalar, double* flux, int d):
#
#     cdef:
#         int imin = 0
#         int jmin = 0
#         int kmin = 0
#
#         int imax = dims.nlg[0] -1
#         int jmax = dims.nlg[1] -1
#         int kmax = dims.nlg[2] -1
#
#         int istride = dims.nlg[1] * dims.nlg[2];
#         int jstride = dims.nlg[2];
#
#         int ishift, jshift, ijk, i,j,k
#
#         int [3] p1 = [istride, jstride, 1]
#         int sp1 = p1[d]
#
#     if d == 2:
#         for i in xrange(imin,imax):
#             ishift = i*istride
#             for j in xrange(jmin,jmax):
#                 jshift = j*jstride
#                 for k in xrange(kmin,kmax):
#                     ijk = ishift + jshift + k
#                     flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1]) * velocity[ijk]/alpha0[k]
#     else:
#         for i in xrange(imin,imax):
#             ishift = i*istride
#             for j in xrange(jmin,jmax):
#                 jshift = j*jstride
#                 for k in xrange(kmin,kmax):
#                     ijk = ishift + jshift + k
#                     flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1])*velocity[ijk]/alpha0_half[k]
#
#     return
#
#
# @cython.boundscheck(False)  #Turn off numpy array index bounds checking
# @cython.wraparound(False)   #Turn off numpy array wrap around indexing
# @cython.cdivision(True)
# cdef fourth_order_cython(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *velocity, double *scalar, double* flux, int d):
#
#     cdef:
#         int imin = 0
#         int jmin = 0
#         int kmin = 0
#
#         int imax = dims.nlg[0] -1
#         int jmax = dims.nlg[1] -1
#         int kmax = dims.nlg[2] -1
#
#         int istride = dims.nlg[1] * dims.nlg[2];
#         int jstride = dims.nlg[2];
#
#         int ishift, jshift, ijk, i,j,k
#
#         int [3] p1 = [istride, jstride, 1]
#         int sm1 = -p1[d]
#         int sp1 = p1[d]
#         int sp2 = 2 * p1[d]
#
#
#     if d == 2:
#         for i in xrange(imin,imax):
#             ishift = i*istride
#             for j in xrange(jmin,jmax):
#                 jshift = j*jstride
#                 for k in xrange(kmin,kmax):
#                     ijk = ishift + jshift + k
#                     flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2]) * velocity[ijk]/alpha0[k]
#     else:
#         for i in xrange(imin,imax):
#             ishift = i*istride
#             for j in xrange(jmin,jmax):
#                 jshift = j*jstride
#                 for k in xrange(kmin,kmax):
#                     ijk = ishift + jshift + k
#                     flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]/alpha0_half[k]
#
#     return




