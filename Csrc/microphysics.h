#pragma once
#include "parameters.h"

// Here, only functions that can be used commonly by any microphysical scheme
// convention: begin function name with "microphysics"
// Scheme-specific functions should be place in a scheme specific .h file, function name should indicate which scheme

inline double microphysics_mean_mass(double n, double q, double min_mass, double max_mass){
    // n = number concentration of species x, 1/kg
    // q = specific mass of species x kg/kg
    // min_mass, max_mass = limits of allowable masses, kg
    // return: mass = mean particle mass in kg
    double mass = fmin(fmax(q/(n + machine_eps),min_mass),max_mass); // MAX/MIN: when l_=0, x_=xmin
    return mass;
}

inline double microphysics_diameter_from_mass(double mass, double prefactor, double exponent){
    // find particle diameter from scaling rule of form
    // D = prefactor * mass ** exponent
    double diameter = prefactor * pow(mass, exponent);
    return diameter;
}

