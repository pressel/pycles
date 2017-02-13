#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"

inline float  sd_c(float  pd, float  T){
    return sd_tilde + cpd*log(T/T_tilde) -Rd*log(pd/p_tilde);
}

inline float  sv_c(float  pv, float  T){
    return sv_tilde + cpv*log(T/T_tilde) - Rv * log(pv/p_tilde);
}

inline float  sc_c(float  L, float  T){
    return -L/T;
}

inline float  s_tendency_c(float  p0, float  qt, float  qv, float  T, float  qt_tendency, float  T_tendency){
    const float  pv = pv_c(p0, qt, qv);
    const float  pd = pd_c(p0, qt, qv);
    return cpm_c(qt) * T_tendency / T +  (sv_c(pv,T) - sd_c(pd,T)) * qt_tendency;
}



