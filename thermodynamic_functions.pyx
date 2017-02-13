cpdef inline float theta(const float p0, const float T) nogil:
    return theta_c(p0, T)

cpdef inline float thetali(const float p0, const float T, const float qt, const float ql, const float qi, const float L) nogil:
    return thetali_c(p0, T, qt, ql, qi, L)

cpdef inline float exner(const float p0) nogil:
    return exner_c(p0)

cpdef inline float pd(const float p0, const float qt, const float qv) nogil:
    return pd_c(p0, qt, qv)

cpdef inline float pv(const float p0, const float qt, const float qv) nogil:
    return pv_c(p0, qt, qv)

cpdef inline float density_temperature(const float T, const float qt, const float qv) nogil:
    return density_temperature_c(T, qt, qv)

cpdef inline float theta_rho(const float p0, const float T, const float qt, const float qv) nogil:
    return theta_rho_c(p0, T, qt, qv)

cpdef inline float cpm(const float qt) nogil:
    return cpm_c(qt)

cpdef inline float thetas(const float s, const float qt) nogil:
    return thetas_c(s, qt)

cpdef inline float thetas_t(const float p0, const float T, const float qt, const float qv,
                      const float qc, const float L) nogil:
    return thetas_t_c( p0,  T, qt, qv, qc, L)

cpdef inline float entropy_from_thetas_(const float thetas, const float qt) nogil:
    return entropy_from_thetas_c(thetas, qt)

cpdef inline float buoyancy(const float alpha0, const float alpha) nogil:
    return buoyancy_c(alpha0, alpha)

cpdef inline float alpha(const float p0, const float T, const float qt, const float qv) nogil:
    return alpha_c(p0, T, qt, qv)

cpdef inline float qv_star(const float p0, const float qt, const float pv) nogil:
    return qv_star_c(p0, qt, pv)