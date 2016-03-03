#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "momentum_advection_weno.h"
#include "momentum_advection_central.h"
#include "flux_divergence.h"
#include<stdio.h>


void compute_advective_tendencies_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
                                double* restrict alpha0, double* restrict alpha0_half,
                                double* restrict vel_advected, double* restrict vel_advecting,
                                double* restrict tendency, ssize_t d_advected, ssize_t d_advecting, int scheme){

    switch(scheme){
        case 2:
            second_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 3:
            weno_third_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 4:
            fourth_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 5:
            weno_fifth_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 6:
           sixth_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 7:
            weno_seventh_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 8:
            eighth_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 9:
            weno_ninth_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 11:
            weno_eleventh_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        default:
            // Default to second order scheme.
            second_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
    };
}
