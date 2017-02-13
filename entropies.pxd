cdef extern from "entropies.h":
    inline float sd_c(float pd, float T) nogil
    inline float sv_c(float pv, float T) nogil
    inline float sc_c(float L, float T) nogil
    inline float s_tendency_c(float p0, float qt, float qv, float T,
                               float qt_tendency, float T_tendency) nogil
