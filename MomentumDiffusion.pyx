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
    cdef void compute_diffusive_flux(Grid.DimStruct *dims, double* vgrad1, double* vgrad2, double* viscosity, double* flux)
    cdef void compute_entropy_source(Grid.DimStruct *dims, double* viscosity, double* strain_rate_mag, double* temperature, double* entropy_tendency)

cdef class MomentumDiffusion:

    def __init__(self, DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa):
        DV.add_variables('viscosity','--','sym',Pa)
        return

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        self.flux = np.zeros((Gr.dims.dims*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')

        return

    cpdef update(self, Grid.Grid Gr,ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, Kinematics.Kinematics Ke):

        cdef:
            long i1
            long i2
            long shift_v1
            long shift_vgrad1
            long shift_vgrad2
            long shift_flux
            long count = 0
            long visc_shift = DV.get_varshift(Gr,'viscosity')
            long temp_shift = DV.get_varshift(Gr,'temperature')
            long s_shift = PV.get_varshift(Gr,'s')

        for i1 in xrange(Gr.dims.dims):
            shift_v1 = PV.velocity_directions[i1] * Gr.dims.npg
            for i2 in xrange(Gr.dims.dims):
                shift_vgrad1 = Ke.get_grad_shift(Gr,i1,i2)
                shift_vgrad2 = Ke.get_grad_shift(Gr,i1,i2)
                shift_flux = count * Gr.dims.npg

                #First we compute the flux
                compute_diffusive_flux(&Gr.dims,&Ke.vgrad[shift_vgrad1],&Ke.vgrad[shift_vgrad2],&DV.values[visc_shift],&self.flux[shift_flux])
                momentum_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[shift_flux],&PV.values[shift_v1],Gr.dims.dx[i1],i1,i2)


                count += 1

        compute_entropy_source(&Gr.dims, &DV.values[visc_shift], &Ke.strain_rate_mag[0], &DV.values[temp_shift],&PV.tendencies[s_shift])






        return
#
#         cdef:
#             long i1
#             long i2
#             long shift_u1
#             long shift_u2
#             long shift_flux
#             long flux_count = 0
#
#
#         cdef int visc_shift = DV.get_nv('viscosity')
#
#
#         '''The indexing here refers to the first (i1) and second (i2) tensor components of the diffusive flux tensor
#         the loops over i1 and i2 are so that we don't have to compute one of the symmetric off diagonal flux tensor
#         components
#         '''
#
#         for i1 in xrange(Gr.dims.dims):
#             shift_u1 = PV.velocity_directions[i1] * Gr.dims.npg
#             for i2 in xrange(Gr.dims.dims-1,i1-1,-1):
#                 shift_flux = flux_count * Gr.dims.npg
#                 shift_u2 = PV.velocity_directions[i2] * Gr.dims.npg
#                 compute_diffusive_flux(&Gr.dims, &Rs.alpha0[0], &Rs.alpha0_half[0],&DV.values[visc_shift],
#                                        &PV.values[shift_u1],&PV.values[shift_u2],&self.flux[shift_flux],
#                                        Gr.dims.dx[i1],Gr.dims.dx[i2],i1,i2,2)
#
#                 if(i1 == i2):
#                     ''' Compute the diagonal flux divergences'''
#                     momentum_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[shift_flux],&PV.values[shift_u1],Gr.dims.dx[i1],i1,i2)
#                 else:
#                     ''' Compute the off diagonal flux divergencs'''
#                     momentum_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[shift_flux],&PV.values[shift_u1],Gr.dims.dx[i1],i1,i2)
#                     momentum_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[shift_flux],&PV.values[shift_u2],Gr.dims.dx[i2],i2,i1)
#
#
#                 flux_count += 1
#
#
#
#
#
#         return
#
#
# cdef compute_diffusive_flux(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
#                             double *viscosity, double *u1, double *u2 ,double *flux, double dx1, double dx2, int i1, int i2, int scheme):
#
#     if scheme == 2:
#         second_order(dims, alpha0, alpha0_half,
#                             viscosity, u1, u2, flux, dx1, dx2, i1, i2)
#
#     return
#
# @cython.boundscheck(False)  #Turn off numpy array index bounds checking
# @cython.wraparound(False)   #Turn off numpy array wrap around indexing
# @cython.cdivision(True)
# cdef second_order(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
#                             double *viscosity, double *u1, double *u2, double *flux, double dx1, double dx2, int i1 , int i2):
#
#     cdef:
#
#         int imin = 1
#         int jmin = 1
#         int kmin = 1
#
#         int imax = dims.nlg[0]-1
#         int jmax = dims.nlg[1]-1
#         int kmax = dims.nlg[2]-1
#
#         int istride = dims.nlg[1] * dims.nlg[2]
#         int jstride = dims.nlg[2]
#
#         int ishift, jshift
#
#         int i,j,k,ijk
#
#         int sm1_1 = -1
#         int sm1_2 = -1
#
#         int sp1_1 = 1
#         int sp1_2 = 1
#
#         double dxi_1 = 1.0/dx1
#         double dxi_2 = 2.0/dx2
#
#     if i1 == 0:
#         sm1_1 = -istride
#         sp1_1 = istride
#     elif i2 == 1:
#         sm1_1 = -jstride
#         sp1_1 = jstride
#
#     if i2 == 0:
#         sm1_2 = -istride
#         sp1_2 = istride
#     elif i2 == 1:
#         sm1_2 = -jstride
#         sp1_2 = jstride
#
#
#     #Diagonal terms but not in z direction
#     if i1 == i2 and i1 != 2:
#         for i in xrange(imin,imax):
#             ishift = i*istride
#             for j in xrange(jmin,jmax):
#                 jshift = j * jstride
#                 for k in xrange(kmin,kmax):
#                     ijk = ishift + jshift + k
#                     flux[ijk] = -2.0 * (u1[ijk+sp1_1] - u1[ijk])*dxi_1*viscosity[ijk + sp1_1] / alpha0_half[k]
#     #Diagonal term in z direction
#     elif i1 == i2:
#         for i in xrange(imin,imax):
#             ishift = i*istride
#             for j in xrange(jmin,jmax):
#                 jshift = j * jstride
#                 for k in xrange(kmin,kmax):
#                     ijk = ishift + jshift + k
#                     flux[ijk] = -2.0 * (u1[ijk+sp1_1] - u1[ijk])*dxi_1*viscosity[ijk + sp1_1] / alpha0_half[k + 1]
#     #Off diagonal term in z direction
#     elif i1 == 2 or i2 == 2:
#          for i in xrange(imin,imax):
#             ishift = i*istride
#             for j in xrange(jmin,jmax):
#                 jshift = j * jstride
#                 for k in xrange(kmin,kmax):
#                     ijk = ishift + jshift + k
#                     flux[ijk] = -(((u1[ijk] - u1[ijk +sm1_1]) * dxi_1 +
#                                 (u2[ijk] - u2[ijk+sm1_2])*dxi_2)
#                                  * (viscosity[ijk + sm1_1 + sm1_2] + viscosity[ijk + sm1_1] + viscosity[ijk + sm1_2] + viscosity[ijk])*0.25)/alpha0[k+1]
#     #Off diagonal terms not in z direction
#     else:
#         for i in xrange(imin,imax):
#             ishift = i*istride
#             for j in xrange(jmin,jmax):
#                 jshift = j * jstride
#                 for k in xrange(kmin,kmax):
#                     ijk = ishift + jshift + k
#                     flux[ijk] =  -(((u1[ijk] - u1[ijk +sm1_1]) * dxi_1 +
#                                 (u2[ijk] - u2[ijk+sm1_2])*dxi_2)
#                                  * (viscosity[ijk + sm1_1 + sm1_2] + viscosity[ijk + sm1_1] + viscosity[ijk + sm1_2] + viscosity[ijk])*0.25) / alpha0_half[k]
#
#     return