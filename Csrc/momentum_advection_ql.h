#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include<stdio.h>

void second_order_m_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, ssize_t d_advected, ssize_t d_advecting){

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        const ssize_t imin = 0;
        const ssize_t jmin = 0;
        const ssize_t kmin = 0;

        const ssize_t imax = dims->nlg[0]-1;
        const ssize_t jmax = dims->nlg[1]-1;
        const ssize_t kmax = dims->nlg[2]-1;

        const ssize_t stencil[3] = {istride,jstride,1};
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp1_ing = stencil[d_advected];

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing])
                            *interp_2(vel_advected[ijk],vel_advected[ijk + sp1_ed]) )*rho0_half[k];
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing])
                            *interp_2(vel_advected[ijk],vel_advected[ijk + sp1_ed]) )*rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing])
                            *interp_2(vel_advected[ijk],vel_advected[ijk + sp1_ed]) )*rho0[k];
                    }
                }
            }
        }
        return;
    }


void sixth_order_m_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, ssize_t d_advected, ssize_t d_advecting){

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        const ssize_t imin = 2;
        const ssize_t jmin = 2;
        const ssize_t kmin = 2;

        const ssize_t imax = dims->nlg[0]-3;
        const ssize_t jmax = dims->nlg[1]-3;
        const ssize_t kmax = dims->nlg[2]-3;

        const ssize_t stencil[3] = {istride,jstride,1};
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2 * sp1_ed ;
        const ssize_t sp3_ed = 3 * sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;
        const ssize_t sm2_ed = -2*sp1_ed;

        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2 * sp1_ing;
        const ssize_t sp3_ing = 3 * sp1_ing;
        const ssize_t sm1_ing = -sp1_ing;
        const ssize_t sm2_ing = -2*sp1_ing;

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = (interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed])) * rho0_half[k];
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = (interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]))* rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = (interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]))* rho0[k];
                    }
                }
            }
        }
        return;
    }


void compute_eddy_fluxes_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
                                double* restrict flux, ssize_t d_advected, ssize_t d_advecting, int scheme){



    switch(scheme){
        case 2:
            second_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 3:
            weno_third_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 4:
            fourth_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 5:
            weno_fifth_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 6:
            sixth_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 7:
            weno_seventh_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 8:
            eighth_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 9:
            weno_ninth_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 11:
            weno_eleventh_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 12:
            // This is an application of fourth order Wicker-Skamarock to momentum but using a lower order interpolation
            // for advecting velocity.
            fourth_order_ws_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        case 13:
            // This is an application of sixth order Wicker-Skamarock to momentum but using a lower order interpolation
            // for advecting velocity.
            sixth_order_ws_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
        default:
            // Default to second order scheme.
            second_order_m(dims, rho0, rho0_half, vel_advected, vel_advecting,
                flux, d_advected, d_advecting);
            break;
    };
}
