#pragma once

void smagorinsky_update(const struct DimStruct *dims, double* restrict visc, double* restrict diff,
double* restrict buoy_freq, double* restrict strain_rate_mag, double cs, double prt){

    double delta = pow(dims->dx[0]*dims->dx[1]*dims->dx[2],1.0/3.0);

    for (size_t i=0; i<dims->npg; i++){
        double s2 = strain_rate_mag[i]*strain_rate_mag[i];
        visc[i] = cs*cs*delta*delta*s2;
        diff[i] = visc[i]/prt;
        if(buoy_freq[i] > 0.0){
            double fb = fmax(1.0 - buoy_freq[i]/(prt*s2),0.0);
            visc[i] = visc[i] * fb;
            diff[i] = diff[i] * fb;
        }

    }
    return;
}

void tke_viscosity_diffusivity(const struct DimStruct *dims, double* restrict e, double* restrict buoy_freq,
double* restrict visc, double* restrict diff, double cn, double ck){
    const double delta = pow(dims->dx[0]*dims->dx[1]*dims->dx[2],1.0/3.0);
    double ell = delta;


    for (size_t i=0; i<dims->npg; i++){
        if(buoy_freq[i]> 0.0){
            ell = fmin(cn*sqrt(e[i]/buoy_freq[i]),delta);
        }
        else{
            ell = delta;
        }
        visc[i] = ck * ell * sqrt(e[i]);
        const double prt = delta/(delta + 2.0 * ell);
        diff[i] = visc[i]/prt;

    }

    return;
}

void tke_dissipation(const struct DimStruct *dims, double* restrict e, double* restrict e_tendency, double* restrict buoy_freq, double cn,  double ck){
    const double delta = pow(dims->dx[0]*dims->dx[1]*dims->dx[2],1.0/3.0);
    double ell = delta;


    for (size_t i=0; i<dims->npg; i++){
        if(buoy_freq[i]> 0.0){
            ell = fmin(cn*sqrt(e[i]/buoy_freq[i]),delta);
        }
        else{
            ell = delta;
        }
        const double ceps= 1.9 * ck + (0.93 - 1.9 * ck) * ell/delta;


    }

    return;
}