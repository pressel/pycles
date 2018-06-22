cdef extern from "entropies.h":
    double sd_c(double pd, double T) nogil
    double sv_c(double pv, double T) nogil
    double sc_c(double L, double T) nogil
    double s_tendency_c(double p0, double qt, double qv, double T,
                               double qt_tendency, double T_tendency) nogil
