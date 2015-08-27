cpdef inline double sd(double pd, double T) nogil:
    return sd_c(pd, T)
cpdef inline double sv(double pv, double T) nogil:
    return sv_c(pv, T)
cpdef inline double sc(double L, double T) nogil:
    return sc_c(L, T)
