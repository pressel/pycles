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


void exchange_coefficients_byun(double Ri, double zb, double z0, double cm, double ch){

    //Monin-Obukhov similarity based on
    //Daewon W. Byun, 1990: On the Analytical Solutions of Flux-Profile Relationships for the Atmospheric Surface Layer. J. Appl. Meteor., 29, 652–657.
    //doi: http://dx.doi.org/10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2

    const double Pr0 = 0.74;
    const double beta_m = 4.7;
    const double beta_h = beta_m/Pr0;
    const double gamma_m = 15.0;
    const double gamma_h = 9.0;
    const double Ri_crit = 0.25;
    const double vkb = 0.35; //Von Karman constant from Businger 1971 used by Byun
    const double C_neu = vkb/logz
    const double logz = log(zb/z0);
    const double zfactor = zb/(zb-z0)*logz;
    double zeta, lmo, zeta0, psi_m, psi_h

    double sb = Ri/Pr0;

    if(Ri <= 0.0 ){
        // Unstable or neutral case
        const double qb = 1.0/9.0 * (1.0 /(gamma_m * gamma_m) + 3.0 * gamma_h/gamma_m * sb * sb);
        const double pb = 1.0/54.0 * (-2.0/(gamma_m*gamma_m*gamma_m) + 9.0/gamma_m * (-gamma_h/gamma_m + 3.0)*sb * sb);
        const double crit = qb * qb *qb - pb * pb;
        if(crit >=0.0){
            const double angle = acos(pb/pow(qb,1.5));
            zeta = zfactor * (1.0/(3.0*gamma_m)-(tb+qb/tb));
        }
        else{
            const double tb = pow((sqrt(-crit) + fabs(pb)),0.3333);
            zeta = zfactor * (1.0/(3.0*gamma_m)-(tb+qb/tb));
        }
        lmo = zb/zeta;
        zeta0 = z0/lmo;
        const double x = pow((1.0 - gamma_m * zeta),0.25);
        const double x0 = pow((1.0 - gamma_m * zeta0), 0.25);
        const double y = sqrt(1.0 - gamma_h * zeta );
        const double y = sqrt(1.0 - gamma_h * zeta0 );
        psi_m = 2.0 * log((1.0 + x)/(1.0 + x0)) + log((1.0 + x*x)/(1.0 + x0 * x0))-2.0*atan(x)+2.0*atan(x0);
        psi_h = 2.0 * log((1.0 + y)/(1.0+y0));
    }
    else{
        //Stable case
        const double Ri_cut = 1.0/(logz+1.0/Ri_crit);
        //distinguish between stable and very stable cases
        if(Ri > Ri_cut){
            zeta = logz * Ri/(1.0 - Ri_cut/Ri_crit);
        }
        else{
            zeta = zfactor/(2.0*beta_h*(beta_m*Ri -1.0))*((1.0-2.0*beta_h*Ri)-sqrt(1.0+4.0*(beta_h - beta_m)*sb));
        }
        lmo = zb/zeta;
        zeta0 = z0/zeta;
        if(zeta > 1.0){
            psi_m = 1.0 - beta_m - zeta;
            psi_h = 1.0 - beta_h - zeta;
        }
        else{
            psi_m = 1.0 - beta_m - zeta;
            psi_h = 1.0 - beta_h - zeta;
        }
    }
    const double cu = min(vkb/(logz-psi_m),2.0*C_neu);
    const double cth = min(vkb/(logz-psi_h)/Pr0,4.5*C_neu);
    cm = cu * cu;
    ch = cu * cth;

    return;
}
