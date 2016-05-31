#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "momentum_advection_weno.h"
#include "momentum_advection_weno_pt.h"
#include "momentum_advection_central.h"
#include "momentum_advection_central_ws.h"
#include "momentum_advection_central_pt.h"
#include "flux_divergence.h"

#include "momentum_advection_central_ql.h"
#include "momentum_advection_central_ws_ql.h"
#include "momentum_advection_weno_ql.h"
#include<stdio.h>


void compute_advective_tendencies_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
                                double* restrict alpha0, double* restrict alpha0_half,
                                double* restrict vel_advected, double* restrict vel_advecting,
                                double* restrict tendency, ssize_t d_advected, ssize_t d_advecting, int scheme){


    switch(scheme){
        case 2:
//            if( d_advected==1 && d_advecting==1 ){
//                printf("2nd order full Momentum Transport \n");
//                }
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
            // This is an application of fourth order Wicker-Skamarock to momentum but using a lower order interpolation
            // for advecting velocity.
            fourth_order_ws_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 16:
            // This is an application of sixth order Wicker-Skamarock to momentum but using a lower order interpolation
            // for advecting velocity.
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

        // the following schemes are quasi-linear modifications of the cases 2 and 4
        case 102:
//            if( d_advected==1 && d_advecting==1 ){
//                printf("2nd order QL Momentum Transport \n");
//                }
            second_order_m_ql(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 104:
//            if (d_advected==1 && d_advecting==1){
//                printf("4th order QL Momentum Transport \n");}
            fourth_order_m_ql(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 105:
            weno_fifth_order_m_ql(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
        case 114:
            fourth_order_ws_m_ql(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;



        case 205:
            weno_fifth_order_m_decomp(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;

        default:
            // Default to second order scheme.
            second_order_m(dims, rho0, rho0_half, alpha0, alpha0_half, vel_advected, vel_advecting,
                tendency, d_advected, d_advecting);
            break;
    };
}
