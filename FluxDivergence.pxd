cimport Grid


cdef extern from 'flux_divergence.h':
    void scalar_flux_divergence(Grid.DimStruct *dims, float *alpha0, float *alpha0_half,
                                float *flux, float *tendency, float dx, Py_ssize_t d) nogil


    void momentum_flux_divergence(Grid.DimStruct *dims, float *alpha0, float *alpha0_half,
                                  float *flux, float *tendency, Py_ssize_t d_advected, Py_ssize_t d_advecting) nogil
