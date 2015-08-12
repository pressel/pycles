cimport Grid
import cython

@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
cdef scalar_flux_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                            double *flux, double *tendency, double dx, int d):

    cdef:
        int imin = 0
        int jmin = 0
        int kmin = 0

        int imax = dims.nlg[0] -1
        int jmax = dims.nlg[1] -1
        int kmax = dims.nlg[2] -1

        int istride = dims.nlg[1] * dims.nlg[2];
        int jstride = dims.nlg[2];

        int ishift, jshift, ijk, i,j,k

        double dxi = 1.0/dx


        #Compute the strides given the dimensionality
        int [3] p1 = [istride, jstride, 1]
        int sm1 =  -p1[d]


    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tendency[ijk] = tendency[ijk] - alpha0_half[k]*(flux[ijk] - flux[ijk+sm1])*dxi


    return


@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
cdef momentum_flux_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half
                              ,double *flux, double *tendency, double dx, long d_advected, long d_advecting):

    cdef:
        int imin = dims.gw
        int jmin = dims.gw
        int kmin = dims.gw

        int imax = dims.nlg[0] - dims.gw
        int jmax = dims.nlg[1] - dims.gw
        int kmax = dims.nlg[2] - dims.gw

        int istride = dims.nlg[1] * dims.nlg[2]
        int jstride = dims.nlg[2]

        int ishift, jshift, ijk, i,j,k

        double dxi = 1.0/dx


    #Compute the strides given the dimensionality
        int [3] p1 = [istride, jstride, 1]
        int m1 =  -p1[d_advecting]

    #apply some logic to make sure the correct specific volume is used
    with nogil:
        if d_advecting != 2:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        tendency[ijk] = tendency[ijk] - (flux[ijk] - flux[ijk + m1])*alpha0_half[k]*dxi
        else:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        tendency[ijk] = tendency[ijk] - (flux[ijk] - flux[ijk + m1])*alpha0[k]*dxi

    return