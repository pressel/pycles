#pragma once
#include "parameters.h"

inline double sd_c(double pd, double T){
    return sd_tilde + cpd*log(T/T_tilde) -Rd*log(pd/p_tilde);
}

inline double sv_c(double pv, double T){
    return sv_tilde + cpv*log(T/T_tilde) - Rv * log(pv/p_tilde);
}

inline double sc_c(double L, double T){
    return -L/T;
}

