#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "entropies.h"

inline double compute_ustar_c(double windspeed, double buoyancy_flux, double z0, double z1){

    const double am = 4.8;
    const double bm = 19.3;
    const double c1 = -0.50864521488493919; // = pi/2 - 3*log(2)
    const double lnz = log(z1/fabs(z0));
    double ustar = windspeed * kappa / lnz;

    if(fabs(buoyancy_flux)>1.0e-10){
        for (long i=0; i<6; i++){
            double lmo = -pow(ustar,3.0)/(buoyancy_flux * kappa);
            double zeta = z1/lmo;
            if(zeta > 0.0){
                ustar = kappa * windspeed/(lnz + am*zeta);
            }
            else{
                double x = pow(1.0 - bm * zeta, 0.25);
                double psi1 = 2.0 * log(1.0+x) + log(1.0 + x*x) -2.0*atan(x) + c1;
                ustar = windspeed * kappa/(lnz - psi1);
            }
        }
    }

    return ustar;
}

inline double entropyflux_from_thetaflux_qtflux(double thetaflux, double qtflux, double p0_b, double T_b, double qt_b, double qv_b){
    const double exner_b = exner_c(p0_b);
    const double pd_b = pd_c(p0_b, qt_b, qv_b);
    const double pv_b = pv_c(p0_b, qt_b, qv_b);
    const double sd_b = sd_c(pd_b, T_b);
    const double sv_b = sv_c(pv_b, T_b);
    const double cp_b = cpm_c(qt_b);


    double entropyflux = cp_b*thetaflux*exner_b/T_b + qtflux*(sv_b-sd_b);


    return entropyflux;
}