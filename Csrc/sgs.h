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

double tke_ell(double cn, double e, double buoy_freq, double delta){
    double ell;
    if(buoy_freq> 0.0){
        ell = fmin(cn*sqrt(e/buoy_freq),delta);
    }
    else{
        ell = delta;
    }

    return ell;
}
void tke_viscosity_diffusivity(const struct DimStruct *dims, double* restrict e, double* restrict buoy_freq,
double* restrict visc, double* restrict diff, double cn, double ck){
    const double delta = pow(dims->dx[0]*dims->dx[1]*dims->dx[2],1.0/3.0);
    double ell = delta;


    for (size_t i=0; i<dims->npg; i++){
        ell = tke_ell(cn, e[i], buoy_freq[i], delta);
        visc[i] = ck * ell * sqrt(e[i]);
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
        e_tendency[i] += -ceps * pow(e[i],1.5) /ell;
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