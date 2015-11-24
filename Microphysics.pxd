cdef:
    double lambda_constant(double T) nogil

    double latent_heat_constant(double T, double T) nogil

cdef class No_Microphysics_Dry:
    # Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type

cdef class No_Microphysics_SA:
    # Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type

cdef class No_Microphysics_DrySGS:
    # Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type

cdef class No_Microphysics_SA_SGS:
    # Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type

cdef inline double lambda_constant(double T) nogil:
    return 1.0

cdef inline double latent_heat_constant(double T, double Lambda) nogil:
    return 2.501e6

cdef inline double latent_heat_variable(double T, double Lambda) nogil:
    cdef:
        double TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC *
            TC - 0.00006 * TC * TC * TC) * 1000.0
