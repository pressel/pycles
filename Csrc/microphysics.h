#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "entropies.h"
#define KT  2.5e-2 // J/m/s/K
#define DVAPOR 3.0e-5 // m^2/s
#define DENSITY_LIQUID  1000.0 // density of liquid water, kg/m^3
#define MICRO_EPS  1.0e-13
#define C_STOKES_VEL 1.19e8 //(m s)^-1, Rogers 1979, Ackerman 2009
#define SIGMA_G 1.5 //1.2 // geometric standard deviation of droplet psdf.  Ackerman 2009

// Here, only functions that can be used commonly by any microphysical scheme
// convention: begin function name with "microphysics"
// Scheme-specific functions should be place in a scheme specific .h file, function name should indicate which scheme

float  microphysics_mean_mass(float  n, float  q, float  min_mass, float  max_mass){
    // n = number concentration of species x, 1/kg
    // q = specific mass of species x kg/kg
    // min_mass, max_mass = limits of allowable masses, kg
    // return: mass = mean particle mass in kg
    float  mass = fmin(fmax(q/fmax(n, MICRO_EPS),min_mass),max_mass); // MAX/MIN: when l_=0, x_=xmin
    return mass;
}

float  microphysics_diameter_from_mass(float  mass, float  prefactor, float  exponent){
    // find particle diameter from scaling rule of form
    // Dm = prefactor * mass ** exponent
    float  diameter = prefactor * pow(mass, exponent);
    return diameter;
}

float  microphysics_g(struct LookupStruct *LT, float  (*lam_fp)(float ), float  (*L_fp)(float , float ), float  temperature){
    float  lam = lam_fp(temperature);
    float  L = L_fp(temperature,lam);
    float  pv_sat = lookup(LT, temperature);
    float  g_therm = 1.0/(Rv*temperature/DVAPOR/pv_sat + L/KT/temperature * (L/Rv/temperature - 1.0));
    return g_therm;

}

float  microphysics_saturation_ratio(struct LookupStruct *LT,  float  temperature, float   p0, float  qt){
    float  pv_sat = lookup(LT, temperature);
    float  qv_sat = qv_star_c(p0, qt, pv_sat);
    float  saturation_ratio = qt/qv_sat - 1.0;
    return saturation_ratio;
}




float  compute_wetbulb(struct LookupStruct *LT,const float  p0, const float  s, const float  qt, const float  T){


    float  Twet = T;
    float  T_1 = T;
    float  pv_star_1  = lookup(LT, T_1);
    float  qv_star_1 = qv_star_c(p0,qt,pv_star_1);
    ssize_t iter = 0;
    /// If not saturated
    if(qt >= qv_star_1){
        Twet = T_1;
    }
    else{
        qv_star_1 = pv_star_1/(eps_vi * (p0 - pv_star_1) + pv_star_1);
        float  pd_1 = p0 - pv_star_1;
        float  s_1 = sd_c(pd_1,T_1) * (1.0 - qv_star_1) + sv_c(pv_star_1,T_1) * qv_star_1;
        float  f_1 = s - s_1;
        float  T_2 = T_1 - 0.5;
        float  delta_T  = fabs(T_2 - T_1);

        do{
            float  pv_star_2 = lookup(LT, T_2);
            float  qv_star_2 = pv_star_2/(eps_vi * (p0 - pv_star_2) + pv_star_2);
            float  pd_2 = p0 - pv_star_2;
            float  s_2 = sd_c(pd_2,T_2) * (1.0 - qv_star_2) + sv_c(pv_star_2,T_2) * qv_star_2;
            float  f_2 = s - s_2;
            float  T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1);
            T_1 = T_2;
            T_2 = T_n;
            f_1 = f_2;
            delta_T  = fabs(T_2 - T_1);
            iter += 1;
        } while(iter < 1);    //(delta_T >= 1.0e-3);
        Twet  = T_1;
    }

    return Twet;
}



void microphysics_wetbulb_temperature(struct DimStruct *dims, struct LookupStruct *LT, float * restrict p0, float * restrict s,
                                      float * restrict qt,  float * restrict T,  float * restrict Twet ){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                Twet[ijk] = compute_wetbulb(LT, p0[k], s[ijk], qt[ijk],  T[ijk]);

            } // End k loop
        } // End j loop
    } // End i loop
    return;
 }

//See Ackerman et al 2009 (DYCOMS-RF02 IC paper) Eq. 7
 void microphysics_stokes_sedimentation_velocity(const struct DimStruct *dims, float * restrict density, float  ccn,
                                      float * restrict ql, float * restrict qt_velocity){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];
    const float  distribution_factor = exp(5.0 * log(SIGMA_G) * log(SIGMA_G));
    const float  number_factor = C_STOKES_VEL * cbrt((0.75/pi/DENSITY_LIQUID/ccn) * (0.75/pi/DENSITY_LIQUID/ccn));


    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin-1; k<kmax+1; k++){
                const ssize_t ijk = ishift + jshift + k;
                float  ql_tmp = fmax(ql[ijk],0.0);

                qt_velocity[ijk] = -number_factor * distribution_factor *  cbrt(density[k]* density[k] *ql_tmp* ql_tmp);

            }
        }
    }


     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;

                qt_velocity[ijk] = interp_2(qt_velocity[ijk], qt_velocity[ijk+1]) ;
            }
        }
    }


    return;

}