cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil
    inline double sc_c(double L, double T) nogil
    inline double s_tendency_c(double p0, double qt, double qv, double T,
                               double qt_tendency, double T_tendency) nogil
