cimport Grid
import cython

@cython.boundscheck(False)  #Turn off numpy array index bounds checking
@cython.wraparound(False)   #Turn off numpy array wrap around indexing
@cython.cdivision(True)
cdef void scalar_flux_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                            double *flux, double *tendency, double dx, Py_ssize_t d):

    cdef:
        Py_ssize_t imin = 0
        Py_ssize_t jmin = 0
        Py_ssize_t kmin = 0
        Py_ssize_t imax = dims.nlg[0] -1
        Py_ssize_t jmax = dims.nlg[1] -1
        Py_ssize_t kmax = dims.nlg[2] -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2];
        Py_ssize_t jstride = dims.nlg[2];
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = 1.0/dx
        #Compute the strides given the dimensionality
        Py_ssize_t [3] p1 = [istride, jstride, 1]
        Py_ssize_t sm1 =  -p1[d]

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
cdef void momentum_flux_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half
                              ,double *flux, double *tendency, Py_ssize_t d_advected, Py_ssize_t d_advecting):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] - dims.gw
        Py_ssize_t jmax = dims.nlg[1] - dims.gw
        Py_ssize_t kmax = dims.nlg[2] - dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = 1.0/dims.dx[d_advecting]

    #Compute the strides given the dimensionality
        Py_ssize_t [3] p1 = [istride, jstride, 1]
        Py_ssize_t sm1 =  -p1[d_advecting]

    #apply some logic to make sure the correct specific volume is used
    with nogil:
        if d_advected != 2:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        tendency[ijk] = tendency[ijk] - (flux[ijk] - flux[ijk + sm1])*alpha0_half[k]*dxi
        else:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        tendency[ijk] = tendency[ijk] - (flux[ijk] - flux[ijk + sm1])*alpha0[k]*dxi
    return