#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "momentum_advection_weno.h"
#include "momentum_advection_weno_pt.h"
#include "momentum_advection_central.h"
#include "momentum_advection_central_ws.h"
#include "momentum_advection_central_pt.h"
#include "flux_divergence.h"
#include<stdio.h>


void compute_advective_tendencies_m(struct DimStruct *dims, float * restrict rho0, float * restrict rho0_half,
                                float * restrict alpha0, float * restrict alpha0_half,
                                float * restrict vel_advected, float * restrict vel_advecting,
                                float * restrict tendency, ssize_t d_advected, ssize_t d_advecting, int scheme){

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
        case 14:
            fourth_order_ws_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 16:
            sixth_order_ws_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 24:
            fourth_order_m_pt(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 25:
            weno_fifth_order_m_pt(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 26:
            sixth_order_m_pt(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 27:
            weno_seventh_order_m_pt(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 28:
            eighth_order_m_pt(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 29:
            weno_ninth_order_m_pt(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        default:
            // Default to second order scheme.
            second_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
    };
}
