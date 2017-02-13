cpdef inline float sd(float pd, float T) nogil:
    return sd_c(pd, T)
cpdef inline float sv(float pv, float T) nogil:
    return sv_c(pv, T)
cpdef inline float sc(float L, float T) nogil:
    return sc_c(L, T)
cpdef inline float s_tendency(float p0, float qt, float qv, float T,
                                     float qt_tendency, float T_tendency) nogil:
    return s_tendency_c(p0, qt, qv, T, qt_tendency, T_tendency)

