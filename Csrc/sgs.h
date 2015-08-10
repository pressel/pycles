#pragma once

void smagorinsky_update(const struct DimStruct *dims, double* restrict visc, double* restrict diff,
double* restrict buoy_freq, double* restrict strain_rate_mag, double cs, double prt){

    double delta = pow(dims->dx[0]*dims->dx[1]*dims->dx[2],1.0/3.0);

    for (long i=0; i<dims->npg; i++){
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