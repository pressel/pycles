#pragma once

void smagorinsky_update(const struct DimStruct *dims, double* restrict visc, double* restrict diff,
double* restrict buoy_freq, double* restrict strain_rate_mag){

    for long(i=0; i<dims->npg; i++){
        strain_rate_mag[i] = 0.0;
    }
    return;
}