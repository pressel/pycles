#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "entropies.h"

inline float  psi_m_unstable(float  zeta, float  zeta0){
    const float  x = pow((1.0 - gamma_m * zeta),0.25);
    const float  x0 = pow((1.0 - gamma_m * zeta0), 0.25);
    float  psi_m = 2.0 * log((1.0 + x)/(1.0 + x0)) + log((1.0 + x*x)/(1.0 + x0 * x0))-2.0*atan(x)+2.0*atan(x0);

    return psi_m;
}

inline float  psi_h_unstable(float  zeta, float  zeta0){
    const float  y = sqrt(1.0 - gamma_h * zeta );
    const float  y0 = sqrt(1.0 - gamma_h * zeta0 );

    float  psi_h = 2.0 * log((1.0 + y)/(1.0+y0));

    return psi_h;
}

inline float  psi_m_stable(float  zeta, float  zeta0){
    float  psi_m = -beta_m * (zeta - zeta0);
    return psi_m;
}


inline float  psi_h_stable(float  zeta, float  zeta0){
    float  psi_h = -beta_h * (zeta - zeta0);
    return psi_h;
}

float  compute_ustar(float  windspeed, float  buoyancy_flux, float  z0, float  zb){
    float  lmo, zeta, zeta0, psi_m, ustar;
    float  ustar0, ustar1, ustar_new, f0, f1, delta_ustar;
    float  logz = log(zb/z0);

    //use neutral condition as first guess
    ustar0 = windspeed * vkb/logz  ;
    if(fabs(buoyancy_flux) > 1.0e-20){
        lmo = -ustar0 * ustar0 * ustar0/(buoyancy_flux * vkb);
        zeta = zb/lmo;
        zeta0 = z0/lmo;
        if(zeta >= 0.0){
            f0 = windspeed - ustar0/vkb*(logz - psi_m_stable(zeta,zeta0));
            ustar1 = windspeed*vkb/(logz - psi_m_stable(zeta,zeta0));
            lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * vkb);
            zeta = zb/lmo;
            zeta0 = z0/lmo;
            f1 = windspeed - ustar1/vkb*(logz - psi_m_stable(zeta,zeta0));
            ustar = ustar1;
            delta_ustar = ustar1 -ustar0;
            do{
                ustar_new = ustar1 - f1 * delta_ustar/(f1-f0);
                f0 = f1;
                ustar0 = ustar1;
                ustar1 = ustar_new;
                lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * vkb);
                zeta = zb/lmo;
                zeta0 = z0/lmo;
                f1 = windspeed - ustar1/vkb*(logz - psi_m_stable(zeta,zeta0));
                delta_ustar = ustar1 -ustar0;
            }while(fabs(delta_ustar) > 1e-10);
         }
        else{
            f0 = windspeed - ustar0/vkb*(logz - psi_m_unstable(zeta,zeta0));
            ustar1 = windspeed*vkb/(logz - psi_m_unstable(zeta,zeta0));
            lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * vkb);
            zeta = zb/lmo;
            zeta0 = z0/lmo;
            f1 = windspeed - ustar1/vkb*(logz - psi_m_unstable(zeta,zeta0));
            ustar = ustar1;
            delta_ustar = ustar1 -ustar0;
            do{
                ustar_new = ustar1 - f1 * delta_ustar/(f1-f0);
                f0 = f1;
                ustar0 = ustar1;
                ustar1 = ustar_new;
                lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * vkb);
                zeta = zb/lmo;
                zeta0 = z0/lmo;
                f1 = windspeed - ustar1/vkb*(logz - psi_m_unstable(zeta,zeta0));
                delta_ustar = ustar1 -ustar0;
            }while(fabs(delta_ustar) > 1e-10);
         }
    }
    else{
        ustar = ustar0;
    }

    return ustar;
}

inline float  entropyflux_from_thetaflux_qtflux(float  thetaflux, float  qtflux, float  p0_b, float  T_b, float  qt_b, float  qv_b){
    const float  exner_b = exner_c(p0_b);
    const float  pd_b = pd_c(p0_b, qt_b, qv_b);
    const float  pv_b = pv_c(p0_b, qt_b, qv_b);
    const float  sd_b = sd_c(pd_b, T_b);
    const float  sv_b = sv_c(pv_b, T_b);
    const float  cp_b = cpm_c(qt_b);

    float  entropyflux = cp_b*thetaflux*exner_b/T_b + qtflux*(sv_b-sd_b);

    return entropyflux;
}

void exchange_coefficients_byun(float  Ri, float  zb, float  z0, float * cm, float * ch, float * lmo){

    //Monin-Obukhov similarity based on
    //Daewon W. Byun, 1990: On the Analytical Solutions of Flux-Profile Relationships for the Atmospheric Surface Layer. J. Appl. Meteor., 29, 652–657.
    //doi: http://dx.doi.org/10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2


    const float  logz = log(zb/z0);
    const float  zfactor = zb/(zb-z0)*logz;
    float  zeta, zeta0, psi_m, psi_h, lmo_;
//    float  psi_h_stable, psi_m_stable, psi_h_unstable, psi_m_unstable;

    float  sb = Ri/Pr0;

    if(Ri <= 0.0 ){
        // Unstable or neutral case
        const float  qb = 1.0/9.0 * (1.0 /(gamma_m * gamma_m) + 3.0 * gamma_h/gamma_m * sb * sb);
        const float  pb = 1.0/54.0 * (-2.0/(gamma_m*gamma_m*gamma_m) + 9.0/gamma_m * (-gamma_h/gamma_m + 3.0)*sb * sb);
        const float  crit = qb * qb *qb - pb * pb;
        if(crit >=0.0){
            const float  angle = acos(pb/sqrt(qb * qb * qb));
            zeta = zfactor * (-2.0 * sqrt(qb) * cos(angle/3.0)+1.0/(3.0*gamma_m));
        }
        else{
            const float  tb = cbrt(sqrt(-crit) + fabs(pb));
            zeta = zfactor * (1.0/(3.0*gamma_m)-(tb + qb/tb));
        }
        lmo_ = zb/zeta;
        zeta0 = z0/lmo_;
        psi_m = psi_m_unstable(zeta, zeta0);
        psi_h = psi_h_unstable(zeta,zeta0);
    }
    else{
        zeta = zfactor/(2.0*beta_h*(beta_m*Ri -1.0))*((1.0-2.0*beta_h*Ri)-sqrt(1.0+4.0*(beta_h - beta_m)*sb));
        lmo_ = zb/zeta;
        zeta0 = z0/lmo_;
        psi_m = psi_m_stable(zeta, zeta0);
        psi_h = psi_h_stable(zeta,zeta0);
    }
    const float  cu = vkb/(logz-psi_m);
    const float  cth = vkb/(logz-psi_h)/Pr0;

    *cm = cu * cu;
    *ch = cu * cth;
    *lmo = lmo_;

    return;
}

void compute_windspeed(const struct DimStruct *dims, float * restrict u, float * restrict v, float * restrict speed, float  u0, float  v0, float  gustiness ){
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t istride_2d = dims->nlg[1];

    const ssize_t imin = 1;
    const ssize_t jmin = 1;
    const ssize_t kmin = 1;

    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    const ssize_t gw = dims->gw;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            const ssize_t ij = i * istride_2d + j;
            const ssize_t ijk = ishift + jshift + gw ;
            const float  u_interp = interp_2(u[ijk-istride],u[ijk]) + u0;
            const float  v_interp = interp_2(v[ijk-jstride],v[ijk]) + v0;
            speed[ij] = fmax(sqrt(u_interp*u_interp + v_interp*v_interp),gustiness);
        }
    }

    return;
}