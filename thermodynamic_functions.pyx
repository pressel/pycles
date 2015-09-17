cpdef inline double theta(const double p0, const double T) nogil:
    return theta_c(p0, T)

cpdef inline double thetali(const double p0, const double T, const double qt, const double ql, const double qi, const double L) nogil:
    return thetali_c(p0, T, qt, ql, qi, L)

cpdef inline double exner(const double p0) nogil:
    return exner_c(p0)

cpdef inline double pd(const double p0, const double qt, const double qv) nogil:
    return pd_c(p0, qt, qv)

cpdef inline double pv(const double p0, const double qt, const double qv) nogil:
    return pv_c(p0, qt, qv)

cpdef inline double density_temperature(const double T, const double qt, const double qv) nogil:
    return density_temperature_c(T, qt, qv)

cpdef inline double theta_rho(const double p0, const double T, const double qt, const double qv) nogil:
    return theta_rho_c(p0, T, qt, qv)

cpdef inline double cpm(const double qt) nogil:
    return cpm_c(qt)

cpdef inline double thetas(const double s, const double qt) nogil:
    return thetas_c(s, qt)

cpdef inline double thetas_t(const double p0, const double T, const double qt, const double qv,
                      const double qc, const double L) nogil:
    return thetas_t_c( p0,  T, qt, qv, qc, L)

cpdef inline double entropy_from_thetas_(const double thetas, const double qt) nogil:
    return entropy_from_thetas_c(thetas, qt)

cpdef inline double buoyancy(const double alpha0, const double alpha) nogil:
    return buoyancy_c(alpha0, alpha)

cpdef inline double alpha(const double p0, const double T, const double qt, const double qv) nogil:
    return alpha_c(p0, T, qt, qv)

cpdef inline double qv_star(const double p0, const double qt, const double pv) nogil:
    return qv_star_c(p0, qt, pv)