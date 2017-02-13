#pragma once
#include "parameters.h"
#include <math.h>

inline float  exner_c(const float  p0){
    return pow((p0/p_tilde),kappa);
}

inline float  theta_c(const float  p0, const float  T){
    // Dry potential temperature
    return T / exner_c(p0);
}

inline float  thetali_c(const float  p0, const float  T, const float  qt, const float  ql, const float  qi, const float  L){
    // Liquid ice potential temperature consistent with Triopoli and Cotton (1981)
    return theta_c(p0, T) * exp(-L*(ql/(1.0 - qt) + qi/(1.0 -qt))/(T*cpd));
}

inline float  pd_c(const float  p0,const float  qt, const float  qv){
    return p0*(1.0-qt)/(1.0 - qt + eps_vi * qv);
}

inline float  pv_c(const float  p0, const float  qt, const float  qv){
    return p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv);
}

inline float  density_temperature_c(const float  T, const float  qt, const float  qv){
    return T * (1.0 - qt + eps_vi * qv);
}

inline float  theta_rho_c(const float  p0, const float  T, const float  qt, const float  qv){
    return density_temperature_c(T,qt,qv)/exner_c(p0);
}

inline float  cpm_c(const float  qt){
    return (1.0-qt) * cpd + qt * cpv;
}

inline float  thetas_c(const float  s, const float  qt){
    return T_tilde*exp((s-(1.0-qt)*sd_tilde - qt*sv_tilde)/cpm_c(qt));
}

inline float  thetas_t_c(const float  p0, const float  T, const float  qt, const float  qv, const float  qc, const float  L){
    const float  qd = 1.0 - qt;
    const float  pd = pd_c(p0,qt,qt-qc);
    const float  pv = pv_c(p0,qt,qt-qc);
    const float  cpm = cpm_c(qt);
    return T * pow(p_tilde/pd,qd * Rd/cpm)*pow(p_tilde/pv,qt*Rv/cpm)*exp(-L * qc/(cpm*T));
}

inline float  entropy_from_thetas_c(const float  thetas, const float  qt){
    return cpm_c(qt) * log(thetas/T_tilde) + (1.0 - qt)*sd_tilde + qt * sv_tilde;
}

inline float  buoyancy_c(const float  alpha0, const float  alpha){
    return g * (alpha - alpha0)/alpha0;
}

inline float  qv_star_c(const float  p0, const float  qt, const float  pv){
    return eps_v * (1.0 - qt) * pv / (p0 - pv) ;
}

inline float  alpha_c(float  p0, float  T, float   qt, float  qv){
    return (Rd * T)/p0 * (1.0 - qt + eps_vi * qv);
}
