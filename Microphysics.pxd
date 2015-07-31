cdef:
    double lambda_constant(double T) nogil

    double latent_heat_constant(double T, double T) nogil

cdef class No_Microphysics_Dry:
    #Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type

cdef class No_Microphysics_SA:
    #Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type


cdef inline double lambda_constant(double T) nogil:
    return 1.0

cdef inline double latent_heat_constant(double T,double Lambda) nogil:
    return 2.501e6
