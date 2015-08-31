cimport Grid


cdef extern from 'flux_divergence.h':
    void scalar_flux_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                                double *flux, double *tendency, double dx, Py_ssize_t d) nogil


    void momentum_flux_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                                  double *flux, double *tendency, Py_ssize_t d_advected, Py_ssize_t d_advecting) nogil
