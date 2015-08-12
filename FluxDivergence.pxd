cimport Grid

cdef scalar_flux_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                            double *flux, double *tendency, double dx, int d)

cdef momentum_flux_divergence(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                              double *flux, double *tendency, long d_advected, long d_advecting)
