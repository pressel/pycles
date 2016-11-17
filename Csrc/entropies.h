#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"

inline double sd_c(double pd, double T){
    return sd_tilde + cpd*log(T/T_tilde) -Rd*log(pd/p_tilde);
}

inline double sv_c(double pv, double T){
    return sv_tilde + cpv*log(T/T_tilde) - Rv * log(pv/p_tilde);
}

inline double sc_c(double L, double T){
    return -L/T;
}

inline double s_tendency_c(double p0, double qt, double qv, double T, double qt_tendency, double T_tendency){
    const double pv = pv_c(p0, qt, qv);
    const double pd = pd_c(p0, qt, qv);
    return cpm_c(qt) * T_tendency / T +  (sv_c(pv,T) - sd_c(pd,T)) * qt_tendency;
}



