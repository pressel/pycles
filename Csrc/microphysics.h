#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"

#define KT  2.5e-2 // J/m/s/K
#define Dv 3.0e-5 // m^2/s
#define density_liquid  1000.0 // density of liquid water, kg/m^3
#define micro_eps  1.0e-13

// Here, only functions that can be used commonly by any microphysical scheme
// convention: begin function name with "microphysics"
// Scheme-specific functions should be place in a scheme specific .h file, function name should indicate which scheme

double microphysics_mean_mass(double n, double q, double min_mass, double max_mass){
    // n = number concentration of species x, 1/kg
    // q = specific mass of species x kg/kg
    // min_mass, max_mass = limits of allowable masses, kg
    // return: mass = mean particle mass in kg
    double mass = fmin(fmax(q/fmax(n, micro_eps),min_mass),max_mass); // MAX/MIN: when l_=0, x_=xmin
    return mass;
}

double microphysics_diameter_from_mass(double mass, double prefactor, double exponent){
    // find particle diameter from scaling rule of form
    // Dm = prefactor * mass ** exponent
    double diameter = prefactor * pow(mass, exponent);
    return diameter;
}

double microphysics_g(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), double temperature){
    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    double pv_sat = lookup(LT, temperature);

    double g_therm = 1.0/(Rv*temperature/Dv/pv_sat + L/KT/temperature * (L/Rv/temperature - 1.0));
    return g_therm;

}

double microphysics_saturation_ratio(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), double temperature, double  p0, double qt, double qv){
    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    double pv_sat = lookup(LT, temperature);
    double pv = pv_c(p0, qt, qv);
    double saturation_ratio = pv/pv_sat - 1.0;
    return saturation_ratio;
}