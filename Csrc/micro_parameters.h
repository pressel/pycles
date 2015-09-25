#pragma once
#include <stdio.h>
#include <math.h>

double rho_liq = 1000.0;
double visc_air = 2.0e-5;
double lhf = 3.34e5;
double small = 1.0e-10;
double cvl = 4190.0;
double cvi = 2106.0;
double lf0= 3.34e5;


struct hm_parameters{
    double a;
    double b;
    double c;
    double d;
    double gb1;
    double gbd1;
    double gd3;
    double gd6;
    double gamstar;
    double alpha_acc;
    double d_min;
    double d_max;
};

struct hm_properties{
    double mf;
    double diam;
    double vel;
    double lam;
    double n0;
};

struct ret_acc{
    double dyr;
    double dys;
    double dyl;
    double dyi;
};

//hm_parameters rain_param;
//hm_parameters snow_param;
//hm_parameters ice_param;
//hm_parameters liquid_param;
/*
hm_parameters rain_param = {
    .a = pi/6.0*rho_liq,
    .b = 3.0,
    .c = 130.0,
    .d = 0.5,
    .gb1 = gamma(1.0 + rain_param.b),
    .gbd1 = gamma(1.0 + rain_param.b + rain_param.d),
    .gd3 = gamma(3.0 + rain_param.d),
    .gd6 = gamma(6.0 + rain_param.d),
    .gamstar = pow(rain_param.gb1, 1.0/rain_param.b),
    .alpha_acc = 1.0,
    .d_min = 50.0e-6,
    .d_max = 2000.0e-6};

hm_parameters snow_param = {
    .a = 2.5e-2,
    .b = 2.0,
    .c = 4.0,
    .d = 0.25,
    .gb1 = gamma(1.0 + snow_param.b),
    .gbd1 = gamma(1.0 + snow_param.b + snow_param.d),
    .gd3 = gamma(3.0 + snow_param.d),
    .gamstar = pow(snow_param.gb1, 1.0/snow_param.b),
    .alpha_acc = 0.3,
    .d_min = 30.0e-6,
    .d_max = 2000.0e-6};

hm_parameters liquid_param = {
    .a = pi/6.0*rho_liq,
    .b = 3.0,
    .gb1 = gamma(1.0 + liquid_param.b),
    .gbd1 = gamma(1.0 + liquid_param.b + liquid_param.d),
    .gd3 = gamma(3.0 + liquid_param.d),
    .gamstar = pow(liquid_param.gb1, 1.0/liquid_param.b),
    .alpha_acc = 1.0,
    .d_min = 2.0e-6,
    .d_max = 30.0e-6};

double rho_ice = 900.0

hm_parameters ice_param = {
    .a = pi/6.0*rho_ice,
    .b = 3.0,
    .gb1 = gamma(1.0 + ice_param.b),
    .gbd1 = gamma(1.0 + ice_param.b + ice_param.d),
    .gd3 = gamma(3.0 + ice_param.d),
    .gamstar = pow(ice_param.gb1, 1.0/ice_param.b),
    .alpha_acc = 1.0,
    .d_min = 12.5e-6,
    .d_max = 650.0e-6};
*/

