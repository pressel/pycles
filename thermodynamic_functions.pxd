cdef extern from "thermodynamic_functions.h":
    inline float theta_c(const float p0, const float T) nogil
    inline float thetali_c(const float p0, const float T, const float qt, const float ql, const float qi, const float L) nogil
    inline float exner_c(const float p0) nogil
    inline float pd_c(const float p0, const float qt, const float qv) nogil
    inline float pv_c(const float p0, const float qt, const float qv) nogil
    inline float density_temperature_c(const float T, const float qt, const float qv) nogil
    inline float theta_rho_c(const float p0, const float T, const float qt, const float qv) nogil
    inline float cpm_c(const float qt) nogil
    inline float thetas_c(const float s, const float qt) nogil
    inline float thetas_t_c(const float p0, const float T, const float qt, const float qv,
                            const float qc, const float L) nogil
    inline float entropy_from_thetas_c(const float thetas, const float qt) nogil
    inline float buoyancy_c(const float alpha0, const float alpha) nogil
    inline float alpha_c(const float p0, const float T, const float qt, const float qv) nogil
    inline float qv_star_c(const float p0, const float qt, const float pv) nogil
