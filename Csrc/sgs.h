#pragma once
#include "parameters.h"

void smagorinsky_update(const struct DimStruct *dims, double* restrict visc, double* restrict diff,
double* restrict buoy_freq, double* restrict strain_rate_mag, double cs, double prt){

    double delta = cbrt(dims->dx[0]*dims->dx[1]*dims->dx[2]);

    for (size_t i=0; i<dims->npg; i++){
        double s2 = strain_rate_mag[i]*strain_rate_mag[i];
        visc[i] = cs*cs*delta*delta*s2;
        if(buoy_freq[i] > 0.0){
            double fb = sqrt(fmax(1.0 - buoy_freq[i]/(prt*s2),0.0));
            visc[i] = visc[i] * fb;
        }
        diff[i] = visc[i]/prt;
    }
    return;
}

double tke_ell(double cn, double e, double buoy_freq, double delta){
    double ell;
    if(buoy_freq> 1.0e-10){
        ell = fmax(fmin(cn*sqrt(fmax(e,0.0)/buoy_freq),delta),1.0e-10);
    }
    else{
        ell = delta;
    }

    return ell;
}
void tke_viscosity_diffusivity(const struct DimStruct *dims, double* restrict e, double* restrict buoy_freq,
double* restrict visc, double* restrict diff, double cn, double ck){
    const double delta = cbrt(dims->dx[0]*dims->dx[1]*dims->dx[2]);
    double ell = delta;


    for (size_t i=0; i<dims->npg; i++){
        ell = tke_ell(cn, e[i], buoy_freq[i], delta);
        visc[i] = ck * ell * sqrt(fmax(e[i],0.0));
        const double prt = delta/(delta + 2.0 * ell);
        diff[i] = visc[i]/prt;

    }

    return;
}

void tke_dissipation(const struct DimStruct *dims, double* restrict e, double* restrict e_tendency,
double* restrict buoy_freq, double cn,  double ck){
    const double delta = pow(dims->dx[0]*dims->dx[1]*dims->dx[2],1.0/3.0);
    double ell = delta;


    for (size_t i=0; i<dims->npg; i++){
        ell = tke_ell(cn, e[i], buoy_freq[i], delta);
        const double ceps= 1.9 * ck + (0.93 - 1.9 * ck) * ell/delta;
        e_tendency[i] += -ceps * pow(fmax(e[i],0.0),1.5) /ell;
    }

    return;
}

void tke_shear_production(const struct DimStruct *dims,  double* restrict e_tendency, double* restrict visc, double* restrict strain_rate_mag ){
    for (size_t i=0; i<dims->npg; i++){
        e_tendency[i] += visc[i] * strain_rate_mag[i] * strain_rate_mag[i];
    }

    return;
}

void tke_buoyant_production(const struct DimStruct *dims,  double* restrict e_tendency, double* restrict diff, double* restrict buoy_freq ){
    for (size_t i=0; i<dims->npg; i++){
        e_tendency[i] += -diff[i] * buoy_freq[i];
    }

    return;
}


//J. Mailhot and R. Benoit, 1982: A Finite-Element Model of the Atmospheric Boundary Layer Suitable
//for Use with Numerical Weather Prediction Models. J. Atmos. Sci., 39, 2249–2266.
//doi: http://dx.doi.org/10.1175/1520-0469(1982)039<2249:AFEMOT>2.0.CO;2
void tke_surface(const struct DimStruct *dims, double* e, double* lmo, double* ustar, double h_bl, double zb){
    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];
    const size_t istride_2d = dims->nlg[1];

    const size_t imin = 1;
    const size_t jmin = 1;
    const size_t kmin = 1;

    const size_t imax = dims->nlg[0]-1;
    const size_t jmax = dims->nlg[1]-1;
    const size_t kmax = dims->nlg[2]-1;

    const size_t gw = dims->gw;
    const double onethird = 1.0/3.0;

    for(size_t i=imin;i<imax;i++){
        const size_t ishift = i*istride ;
        for(size_t j=jmin;j<jmax;j++){
            const size_t jshift = j*jstride;
            const size_t ij = i * istride_2d + j;
            const size_t ijk = ishift + jshift + gw ;
            if(zb/lmo[ij] >= 0.0){
                e[ijk] = 3.75 * ustar[ij] * ustar[ij];
            }
            else{
                const double wstar = ustar[ij] * pow(-h_bl/lmo[ij]/vkb,onethird);
                e[ijk] =  (3.75 + pow(-zb/lmo[ij],2.0*onethird))  * ustar[ij] * ustar[ij] + 0.2 * wstar * wstar;
             }
        }
    }

    return;
}