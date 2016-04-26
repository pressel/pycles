#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "flux_divergence.h"
#include "cc_statistics.h"
#include<stdio.h>

// !!!!!! control if doing the right thing !!!!!!
void fourth_order_ws_m_ql   (struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        if (d_advected==1 && d_advecting==1){
            printf("4th order WS QL Momentum Transport \n");}

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_int_ed = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_int_ing = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_mean_ed = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_mean_ing = (double *)malloc(sizeof(double) * dims->nlg[2]);

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


        // (1) interpolate velocity fields
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    vel_int_ing[ijk] = interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]);
                    vel_int_ed[ijk] = interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed]);
                }
            }
        }

        // (2) average interpolated velocity fields
        horizontal_mean(dims, &vel_int_ing[0], &vel_mean_ing[0]);
        horizontal_mean(dims, &vel_int_ed[0], &vel_mean_ed[0]);


        // (3) compute eddy flux:
        if (d_advected != 2 && d_advecting !=2){                    // exclude w.u, w.v, w.w, u.w, v.w (advection by or of vertical velocity)
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        eddy_flux[ijk] = (vel_int_ing[ijk] - vel_mean_ing[k]) * (vel_int_ed[ijk] - vel_mean_ed[k]) * rho0_half[k];
                        flux[ijk] = (vel_int_ing[ijk] * vel_int_ed[ijk]) * rho0_half[k];
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){               // w.w
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        eddy_flux[ijk] = (vel_int_ing[ijk] - vel_mean_ing[k]) * (vel_int_ed[ijk] - vel_mean_ed[k]) * rho0_half[k+1];
                        flux[ijk] = (vel_int_ing[ijk] * vel_int_ed[ijk]) * rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(ssize_t i=imin;i<imax;i++){                         // u.w, v.w, w.u, w.v
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        eddy_flux[ijk] = (vel_int_ing[ijk] - vel_mean_ing[k]) * (vel_int_ed[ijk] - vel_mean_ed[k]) * rho0[k];
                        flux[ijk] = (vel_int_ing[ijk] * vel_int_ed[ijk]) * rho0[k];
                    }
                }
            }
        }

        momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                                tendency, d_advected, d_advecting);

        free(eddy_flux);
        free(mean_eddy_flux);
        free(vel_int_ing);
        free(vel_int_ed);
        free(vel_mean_ed);
        free(vel_mean_ing);

        free(flux);
        return;
    }






// !!!!!! NOT Finished !!!!!!
void fourth_order_ws_m_decomp(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        if (d_advected==1 && d_advecting==1){
            printf("4th order WS decomposition Momentum Transport \n");}

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *flux_old = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

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
                    flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0_half[k];
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
                    flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0_half[k+1];
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
                    flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0[k];
                    }
                }
            }
        }

        momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                                tendency, d_advected, d_advecting);

        free(eddy_flux);
        free(mean_eddy_flux);
        free(vel_int_ing);
        free(vel_int_ed);
        free(vel_mean_ed);
        free(vel_mean_ing);

        free(flux);
        return;
    }