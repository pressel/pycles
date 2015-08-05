cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
from FluxDivergence cimport momentum_flux_divergence

import numpy as np
cimport numpy as np

import cython

cdef extern from "advection_interpolation.h":
    inline double interp_2(double phi, double phip1) nogil
    inline double interp_4(double phim1, double phi, double phip1, double phip2) nogil
cdef extern from "momentum_advection.h":
    void second_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil
    void fourth_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil
    void sixth_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil
    void eighth_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil
    void weno_third_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil
    void weno_fifth_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil
    void weno_seventh_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil
    void weno_ninth_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil
    void weno_eleventh_order_c(Grid.DimStruct *dims, double *rho0, double *rho_half, double *vel_advected, double *vel_advecting,
        double* flux, long d_advected, long d_advecting) nogil

cdef class MomentumAdvection:
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
        self.flux = np.zeros((PV.nv_velocities*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')
        return



    cpdef update(self, Grid.Grid Gr,ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):

        cdef:
            long i_advecting       #Direction of advecting velocity
            long i_advected        #Direction of momentum component
            long shift_advected    #Shift to beginning of momentum (velocity) component in the PV.values array
            long shift_advecting   #Shift to beginning of advecting velocity componentin the PV.values array
            long shift_flux        #Shift to the i_advecting, i_advected flux component in the self.flux array

        for i_advected in xrange(Gr.dims.dims):
            #Compute the shift to the starting location of the advected velocity in the PV values array
            shift_advected = PV.velocity_directions[i_advected] * Gr.dims.npg
            for i_advecting in xrange(Gr.dims.dims):

                #Compute the shift to the starting location of the advecting velocity in the PV values array
                shift_advecting = PV.velocity_directions[i_advecting] * Gr.dims.npg

                #Compute the shift to the starting location of the advecting flux.
                shift_flux = i_advected * Gr.dims.dims * Gr.dims.npg + i_advecting * Gr.dims.npg


                #Compute the fluxes
                compute_advective_fluxes(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],
                                         &PV.values[shift_advected],&PV.values[shift_advecting],&self.flux[shift_flux],
                                         i_advected,i_advecting,4)
                #Compute flux divergence
                momentum_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[shift_flux],
                                            &PV.tendencies[shift_advected],Gr.dims.dx[i_advecting],i_advected,i_advecting)


        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef double [:,:,:] get_flux(self, int i_advected, int i_advecting, Grid.Grid Gr):
        cdef:
            int shift_flux = i_advected * Gr.dims.dims * Gr.dims.npg + i_advecting * Gr.dims.npg
            int i,j,k,ijk,ishift, jshift
            int istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            int jstride = Gr.dims.nlg[2]

            int imin = 0
            int jmin = 0
            int kmin = 0

            int imax = Gr.dims.nlg[0]
            int jmax = Gr.dims.nlg[1]
            int kmax = Gr.dims.nlg[2]

            cdef double [:,:,:] return_flux = np.empty((Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2]),dtype=np.double,order='c')
            cdef double [:] flux = self.flux

        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        return_flux[i,j,k] = flux[shift_flux + ishift + jshift + k ]

        return return_flux

cdef void compute_advective_fluxes(Grid.DimStruct *dims, double *rho0, double *rho0_half,double *vel_advected,
                                  double *vel_advecting, double *flux, long d_advected, long d_advecting, int scheme):

    if scheme == 2:
        second_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    elif scheme == 3:
        weno_third_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    elif scheme == 4:
        fourth_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    elif scheme == 5:
        weno_fifth_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    elif scheme == 6:
        sixth_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    elif scheme == 7:
        weno_seventh_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    elif scheme == 8:
        eighth_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    elif scheme == 9:
        weno_ninth_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    elif scheme == 11:
        weno_eleventh_order_c(dims,rho0,rho0_half,vel_advected,vel_advecting,flux,d_advected, d_advecting)
    return


cdef void second_order(Grid.DimStruct *dims, double *rho0, double *rho0_half,double *vel_advected,
                                  double *vel_advecting, double *flux, long d_advected, long d_advecting):
    cdef:

        int imin = 0
        int jmin = 0
        int kmin = 0

        int imax = dims.nlg[0]-1
        int jmax = dims.nlg[1]-1
        int kmax = dims.nlg[2]-1



        int istride = dims.nlg[1] * dims.nlg[2]
        int jstride = dims.nlg[2]

        int ishift, jshift

        int i,j,k,ijk

        int [3] p1 = [istride, jstride, 1]
        int p1_ed =  p1[d_advecting]
        int p1_ing = p1[d_advected]

    if d_advected != 2 and d_advecting != 2:   #These are the horizontal fluxes of horizontal momentum
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+p1_ing]) *
                                 interp_2(vel_advected[ijk],vel_advected[ijk+p1_ed])) * rho0_half[k]
    elif d_advected == 2 and d_advecting == 2: #These are the vertical fluxes of vertical momentum
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+p1_ing])
                                 * interp_2(vel_advected[ijk],vel_advected[ijk+p1_ed])) * rho0_half[k+1]
    else:  #These are the vertical fluxes of horizontal momentum and horizontal fluxes of vertical momentum
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+p1_ing])
                                 * interp_2(vel_advected[ijk],vel_advected[ijk+p1_ed])) * rho0[k]

    return

cdef void fourth_order(Grid.DimStruct *dims, double *rho0, double *rho0_half,double *vel_advected,
                                  double *vel_advecting, double *flux, long d_advected, long d_advecting):
    cdef:

        int imin = 1
        int jmin = 1
        int kmin = 1

        int imax = dims.nlg[0]-1
        int jmax = dims.nlg[1]-1
        int kmax = dims.nlg[2]-1

        int istride = dims.nlg[1] * dims.nlg[2]
        int jstride = dims.nlg[2]

        int ishift, jshift

        int i,j,k,ijk

        int [3] p1 = [istride, jstride, 1]

        int m1_ing = -p1[d_advected]
        int p1_ing = p1[d_advected]
        int p2_ing = 2*p1[d_advected]

        int m1_ed = -p1[d_advecting]
        int p1_ed = p1[d_advecting]
        int p2_ed = 2*p1[d_advecting]


    #Fix the alphas here
    if d_advected != 2 and d_advecting != 2:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    flux[ijk] = (interp_4(vel_advecting[ijk+m1_ing],vel_advecting[ijk],vel_advecting[ijk+p1_ing],vel_advecting[ijk+p2_ing]) *
                                 interp_4(vel_advected[ijk+m1_ed],vel_advected[ijk],vel_advected[ijk+p1_ed],vel_advected[ijk+p2_ed])) * rho0_half[k]
    elif d_advected == 2 and d_advecting == 2:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    flux[ijk] = (interp_4(vel_advecting[ijk+m1_ing],vel_advecting[ijk],vel_advecting[ijk+p1_ing],vel_advecting[ijk+p2_ing]) *
                                 interp_4(vel_advected[ijk+m1_ed],vel_advected[ijk],vel_advected[ijk+p1_ed],vel_advected[ijk+p2_ed])) * rho0_half[k+1]
    else:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j * jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    flux[ijk] = (interp_4(vel_advecting[ijk+m1_ing],vel_advecting[ijk],vel_advecting[ijk+p1_ing],vel_advecting[ijk+p2_ing]) *
                                 interp_4(vel_advected[ijk+m1_ed],vel_advected[ijk],vel_advected[ijk+p1_ed],vel_advected[ijk+p2_ed]))* rho0[k]

    return
