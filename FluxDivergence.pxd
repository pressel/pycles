cimport Grid

cdef scalar_flux_divergence_adv(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                            double *flux, double *tendency, double dx, int d)

cdef scalar_flux_divergence_diff(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                            double *flux, double *tendency, double dx, int d)

cdef momentum_flux_divergence_adv(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                              double *flux, double *tendency, double dx, long d_advected, long d_advecting)

cdef momentum_flux_divergence_diff(Grid.DimStruct *dims, double *alpha0, double *alpha0_half,
                              double *flux, double *tendency, double dx, long d_advected, long d_advecting)
