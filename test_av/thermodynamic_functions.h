#pragma once
#include "parameters.h"
#include <math.h>

inline double exner_c(const double p0){
    return pow((p0/p_tilde),kappa);
}

inline double theta_c(const double p0, const double T){
    // Dry potential temperature
    return T / exner_c(p0);
}

inline double thetali_c(const double p0, const double T, const double qt, const double ql, const double qi, const double L){
    // Liquid ice potential temperature consistent with Triopoli and Cotton (1981)
    return theta_c(p0, T) * exp(-L*(ql/(1.0 - qt) + qi/(1.0 -qt))/(T*cpd));
}

inline double pd_c(const double p0,const double qt, const double qv){
    return p0*(1.0-qt)/(1.0 - qt + eps_vi * qv);
}

inline double pv_c(const double p0, const double qt, const double qv){
    return p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv);
}

inline double density_temperature_c(const double T, const double qt, const double qv){
    return T * (1.0 - qt + eps_vi * qv);
}

inline double theta_rho_c(const double p0, const double T, const double qt, const double qv){
    return density_temperature_c(T,qt,qv)/exner_c(p0);
}

inline double cpm_c(const double qt){
    return (1.0-qt) * cpd + qt * cpv;
}

inline double thetas_c(const double s, const double qt){
    return T_tilde*exp((s-(1.0-qt)*sd_tilde - qt*sv_tilde)/cpm_c(qt));
}

inline double thetas_t_c(const double p0, const double T, const double qt, const double qv, const double qc, const double L){
    const double qd = 1.0 - qt;
    const double pd = pd_c(p0,qt,qt-qc);
    const double pv = pv_c(p0,qt,qt-qc);
    const double cpm = cpm_c(qt);
    return T * pow(p_tilde/pd,qd * Rd/cpm)*pow(p_tilde/pv,qt*Rv/cpm)*exp(-L * qc/(cpm*T));
}

inline double entropy_from_thetas_c(const double thetas, const double qt){
    return cpm_c(qt) * log(thetas/T_tilde) + (1.0 - qt)*sd_tilde + qt * sv_tilde;
}

inline double buoyancy_c(const double alpha0, const double alpha){
    return g * (alpha - alpha0)/alpha0;
}

inline double qv_star_c(const double p0, const double qt, const double pv){
    return eps_v * (1.0 - qt) * pv / (p0 - pv) ;
}

inline double alpha_c(double p0, double T, double  qt, double qv){
    return (Rd * T)/p0 * (1.0 - qt + eps_vi * qv);
}
