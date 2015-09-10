cdef extern from "thermodynamic_functions.h":
    inline double theta_c(const double p0, const double T) nogil
    inline double thetali_c(const double p0, const double T, const double qt, const double ql, const double qi, const double L) nogil
    inline double exner_c(const double p0) nogil
    inline double pd_c(const double p0, const double qt, const double qv) nogil
    inline double pv_c(const double p0, const double qt, const double qv) nogil
    inline double density_temperature_c(const double T, const double qt, const double qv) nogil
    inline double theta_rho_c(const double p0, const double T, const double qt, const double qv) nogil
    inline double cpm_c(const double qt) nogil
    inline double thetas_c(const double s, const double qt) nogil
    inline double thetas_t_c(const double p0, const double T, const double qt, const double qv,
                            const double qc, const double L) nogil
    inline double entropy_from_thetas_c(const double thetas, const double qt) nogil
    inline double buoyancy_c(const double alpha0, const double alpha) nogil
    inline double alpha_c(const double p0, const double T, const double qt, const double qv) nogil
    inline double qv_star_c(const double p0, const double qt, const double pv) nogil
