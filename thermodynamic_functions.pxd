cdef extern from "thermodynamic_functions.h":
    double theta_c(const double p0, const double T) nogil
    double thetali_c(const double p0, const double T, const double qt, const double ql, const double qi, const double L) nogil
    double exner_c(const double p0) nogil
    double pd_c(const double p0, const double qt, const double qv) nogil
    double pv_c(const double p0, const double qt, const double qv) nogil
    double density_temperature_c(const double T, const double qt, const double qv) nogil
    double theta_rho_c(const double p0, const double T, const double qt, const double qv) nogil
    double cpm_c(const double qt) nogil
    double thetas_c(const double s, const double qt) nogil
    double thetas_t_c(const double p0, const double T, const double qt, const double qv,
                            const double qc, const double L) nogil
    double entropy_from_thetas_c(const double thetas, const double qt) nogil
    double buoyancy_c(const double alpha0, const double alpha) nogil
    double alpha_c(const double p0, const double T, const double qt, const double qv) nogil
    double qv_star_c(const double p0, const double qt, const double pv) nogil
