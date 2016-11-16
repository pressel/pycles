cpdef inline double sd(double pd, double T) nogil:
    return sd_c(pd, T)
cpdef inline double sv(double pv, double T) nogil:
    return sv_c(pv, T)
cpdef inline double sc(double L, double T) nogil:
    return sc_c(L, T)
cpdef inline double s_tendency(double p0, double qt, double qv, double T,
                                     double qt_tendency, double T_tendency) nogil:
    return s_tendency_c(p0, qt, qv, T, qt_tendency, T_tendency)

