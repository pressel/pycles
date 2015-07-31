#pragma once
#include "parameters.h"
#include <math.h>


inline double exner_c(const double p0){
    return pow((p_tilde/p0),kappa);
};

inline double theta_c(const double p0, const double T){
    return T * pow((p_tilde/p0),kappa) ;
};

inline double pd_c(const double p0,const double qt, const double qv){
    return p0*(1.0-qt)/(1.0 - qt + eps_vi * qv);
};

inline double pv_c(const double p0, const double qt, const double qv){
    return p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv);
};

inline double virtual_temperature_c(const double T, const double qt, const double qv){
    return T * (1.0 - qt + eps_vi * qv);
}

inline double thetav_c(const double p0, const double T, const double qt, const double qv){
    return virtual_temperature_c(T,qt,qv)*exner_c(p0);
}

inline double cpm_c(const double qt){
    return (1.0-qt) * cpd + qt * cpv;
}

inline double thetas_c(const double s, const double qt){
    return T_tilde*exp((s-(1-qt)*sd_tilde - qt*sv_tilde)/cpm_c(qt));
}

inline double buoyancy_c(const double alpha0, const double alpha){
    return g * (alpha - alpha0)/alpha0;
}

inline double qv_star_c(double p0, double qt, double pv){
    return eps_v * (1.0 - qt) * pv / (p0 - pv) ;
}

inline double qt_from_pv(double p0, double pv){
    double a = eps_v * pv/(p0 - pv);
    return a / (1.0 + a);
}

inline double alpha_c(double p0, double T, double  qt, double qv){
    return (Rd * T)/p0 * (1.0 - qt + eps_vi * qv);
};