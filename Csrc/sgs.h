
void smagorinsky_update(const struct DimStruct *dims, double* restrict visc, double* restrict diff,
double* restrict buoy_freq, double* restrict strain_rate_mag, double cs, double prt){

    double delta = cbrt(dims->dx[0]*dims->dx[1]*dims->dx[2]);

    for (size_t i=0; i<dims->npg; i++){
        visc[i] = cs*cs*delta*delta*strain_rate_mag[i];
        if(buoy_freq[i] > 0.0){
            double fb = sqrt(fmax(1.0 - buoy_freq[i]/(prt*strain_rate_mag[i]*strain_rate_mag[i]),0.0));
            visc[i] = visc[i] * fb;  
        }
        diff[i] = visc[i]/prt;   
    }
    return;
}
