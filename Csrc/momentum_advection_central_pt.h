#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "flux_divergence.h"
#include<stdio.h>

void fourth_order_m_pt(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
        double* restrict alpha0, double* restrict alpha0_half,
        double* restrict vel_advected, double* restrict vel_advecting,
        double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

    // Dynamically allocate flux array
    double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        const ssize_t imin = 1;
        const ssize_t jmin = 1;
        const ssize_t kmin = 1;

        const ssize_t imax = dims->nlg[0]-2;
        const ssize_t jmax = dims->nlg[1]-2;
        const ssize_t kmax = dims->nlg[2]-2;

        const ssize_t stencil[3] = {istride,jstride,1};
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2 * sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;

        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2 * sp1_ing;
        const ssize_t sm1_ing = -sp1_ing;

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = (interp_4_pt(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing]) *
                                 interp_4_pt(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0_half[k];
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
                    flux[ijk] = (interp_4_pt(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing]) *
                                 interp_4_pt(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0_half[k+1];
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
                    flux[ijk] = (interp_4_pt(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing]) *
                                 interp_4_pt(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed]))* rho0[k];
                    }
                }
            }
        }
    momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                            tendency, d_advected, d_advecting);
    free(flux);
    return;
}

void sixth_order_m_pt(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
        double* restrict alpha0, double* restrict alpha0_half,
        double* restrict vel_advected, double* restrict vel_advecting,
        double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

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
                    flux[ijk] = (interp_6_pt(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6_pt(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed])) * rho0_half[k];
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
                    flux[ijk] = (interp_6_pt(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6_pt(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]))* rho0_half[k+1];
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
                    flux[ijk] = (interp_6_pt(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6_pt(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]))* rho0[k];
                    }
                }
            }
        }
    free(flux);
    return;
}

void eighth_order_m_pt(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
        double* restrict alpha0, double* restrict alpha0_half,
        double* restrict vel_advected, double* restrict vel_advecting,
        double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        const ssize_t imin = 3;
        const ssize_t jmin = 3;
        const ssize_t kmin = 3;

        const ssize_t imax = dims->nlg[0]-4;
        const ssize_t jmax = dims->nlg[1]-4;
        const ssize_t kmax = dims->nlg[2]-4;

        const ssize_t stencil[3] = {istride,jstride,1};
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2 * sp1_ed ;
        const ssize_t sp3_ed = 3 * sp1_ed ;
        const ssize_t sp4_ed = 4 * sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;
        const ssize_t sm2_ed = -2*sp1_ed;
        const ssize_t sm3_ed = -3*sp1_ed;

        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2 * sp1_ing;
        const ssize_t sp3_ing = 3 * sp1_ing;
        const ssize_t sp4_ing = 4 * sp1_ing;
        const ssize_t sm1_ing = -sp1_ing;
        const ssize_t sm2_ing = -2*sp1_ing;
        const ssize_t sm3_ing = -3*sp1_ing;

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = (interp_8_pt(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing],vel_advecting[ijk+sp4_ing]) *
                                 interp_8_pt(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed])) * rho0_half[k];
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
                    flux[ijk] = (interp_8_pt(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing],vel_advecting[ijk+sp4_ing]) *
                                 interp_8_pt(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed])) * rho0_half[k+1];
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
                    flux[ijk] = (interp_8_pt(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing],vel_advecting[ijk+sp4_ing]) *
                                 interp_8_pt(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed])) * rho0[k];
                    }
                }
            }
        }
    momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                            tendency, d_advected, d_advecting);
    free(flux);
    return;
}