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
        double *flux_old = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
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
//                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
//                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0_half[k];
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
//                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
//                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0_half[k+1];
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
//                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
//                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0[k];
                    }
                }
            }
        }

//        int a = 0;
//        int c = 0;
//        double b;
//        for(ssize_t i=imin;i<imax;i++){
//            const ssize_t ishift = i*istride;
//            for(ssize_t j=jmin;j<jmax;j++){
//                const ssize_t jshift = j*jstride;
//                for(ssize_t k=kmin;k<kmax;k++){
//                    const ssize_t ijk = ishift + jshift + k;
//                    b = vel_int_ing[ijk] - interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]);
//                    if(b > 1e-5){a += 1;}
//                    b = flux[ijk] - flux_old[ijk];
//                    if(b > 1e-5){c += 1;}
//                }
//            }
//        }
//        if (a > 0){printf("Achtung!!! a > 0\n");}
//        if (c > 0){printf("Achtung!!! Flux != old Flux (c > 0)\n");}

        // (4) compute mean eddy flux
        horizontal_mean(dims, &eddy_flux[0], &mean_eddy_flux[0]);


        // (5) compute QL flux: flux = flux - eddy_flux + mean_eddy_flux
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = flux[ijk] - eddy_flux[ijk] + mean_eddy_flux[k];
//                    flux[ijk] = flux[ijk];
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






void fourth_order_ws_m_decomp(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        if(d_advected==2 && d_advecting==2){
            printf("4th order Momentum Transport decomp\n");}

        // Dynamically allocate flux array
        double *flux_old = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        double *vel_ed_mean = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_ing_mean = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_ed_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_ing_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_ed_fluc = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_ing_fluc = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        ssize_t imin = 0;
        ssize_t jmin = 0;
        ssize_t kmin = 0;

        ssize_t imax = dims->nlg[0];
        ssize_t jmax = dims->nlg[1];
        ssize_t kmax = dims->nlg[2];

        ssize_t i,j,k;

        // (1) compute mean velocities
        horizontal_mean(dims, &vel_advecting[0], &vel_ing_mean_[0]);
        horizontal_mean(dims, &vel_advected[0], &vel_ed_mean_[0]);

        for(i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;
                    vel_ing_mean[ijk] = vel_ing_mean_[k];
                    vel_ed_mean[ijk] = vel_ed_mean_[k];

                    vel_ing_fluc[ijk] = vel_advecting[ijk] - vel_ing_mean[ijk];
                    vel_ed_fluc[ijk] = vel_advected[ijk] - vel_ed_mean[ijk];
                }
            }
        }

        int ok_ing = 0;
        int ok_ed = 0;
        int ok_nan_ed = 0;
        int ok_nan_ing = 0;
        int ok_nan_ed_mean = 0;
        int ok_nan_ing_mean = 0;
        for(i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;
                    double diff = vel_advecting[ijk] - (vel_ing_fluc[ijk] + vel_ing_mean[ijk]);
                    if(fabs(diff)>0.00001){ok_ing = ok_ing + 1;}
                    diff = vel_advected[ijk] - (vel_ed_fluc[ijk] + vel_ed_mean[ijk]);
                    if(fabs(diff)>0.00001){ok_ed = ok_ed + 1;}

                    if(isnan(vel_ing_fluc[ijk])){ok_nan_ing = ok_nan_ing + 1;}
                    if(isnan(vel_ed_fluc[ijk])){ok_nan_ed = ok_nan_ed + 1;}
                    if(isnan(vel_ing_mean[ijk])){ok_nan_ing_mean = ok_nan_ing_mean + 1;}
                    if(isnan(vel_ed_mean[ijk])){ok_nan_ed_mean = ok_nan_ed_mean + 1;}
                }
            }
        }
        if(ok_ing > 1){printf("problem decomposition advecting: count = %d\n",ok_ing);}
        if(ok_ed > 1){printf("problem decomposition advected: count = %d\n",ok_ed);}
        if(ok_nan_ing > 1){printf("problem nan advecting fluc: count = %d\n",ok_nan_ing);}
        if(ok_nan_ed > 1){printf("problem nan advected fluc: count = %d\n",ok_nan_ed);}
        if(ok_nan_ing_mean > 1){printf("problem nan advecting mean: count = %d\n",ok_nan_ing_mean);}
        if(ok_nan_ed_mean > 1){printf("problem nan advected mean: count = %d\n",ok_nan_ed_mean);}

        // (2) compute flux
        const ssize_t stencil[3] = {istride,jstride,1};
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2 * sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;

        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2 * sp1_ing;
        const ssize_t sm1_ing = -sp1_ing;

        imin = 1;
        jmin = 1;
        kmin = 1;

        imax = dims->nlg[0]-2;
        jmax = dims->nlg[1]-2;
        kmax = dims->nlg[2]-2;

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        const double flux_eddy = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0_half[k];
                        const double flux_m1 = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0_half[k];
                        const double flux_m2 = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0_half[k];
                        const double flux_mean = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0_half[k];

                        flux[ijk] = flux_eddy + flux_m1 + flux_m2 + flux_mean;
                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
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
                        const double flux_eddy = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0_half[k+1];
                        const double flux_m1 = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0_half[k+1];
                        const double flux_m2 = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0_half[k+1];
                        const double flux_mean = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0_half[k+1];

                        flux[ijk] = flux_eddy + flux_m1 + flux_m2 + flux_mean;

                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
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
                        const double flux_eddy = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0[k+1];
                        const double flux_m1 = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0[k+1];
                        const double flux_m2 = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0[k+1];
                        const double flux_mean = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0[k+1];

                        flux[ijk] = flux_eddy + flux_m1 + flux_m2 + flux_mean;

                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0[k];
                    }
                }
            }
        }
        int ok_nan = 0;
        for(i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;

                    if(isnan(flux[ijk])){ok_nan = ok_nan + 1;}
                }
            }
        }
        if(ok_nan > 1){printf("problem nan flux: count = %d\n",ok_nan);}
//        else{printf("no nans in MA fluxes\n");}

        momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                                tendency, d_advected, d_advecting);
        free(flux);
        free(flux_old);
        free(vel_ed_mean);
        free(vel_ing_mean);
        free(vel_ed_mean_);
        free(vel_ing_mean_);
        free(vel_ed_fluc);
        free(vel_ing_fluc);
        return;
    }



void fourth_order_ws_m_decomp_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        if(d_advected==2 && d_advecting==2){
            printf("4th order Momentum Transport decomp QL\n");}

        // Dynamically allocate flux array
        double *flux_old = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
//        double *flux_ql = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        double *vel_ed_mean = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_ing_mean = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_ed_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_ing_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_ed_fluc = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_ing_fluc = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        ssize_t imin = 0;
        ssize_t jmin = 0;
        ssize_t kmin = 0;

        ssize_t imax = dims->nlg[0];
        ssize_t jmax = dims->nlg[1];
        ssize_t kmax = dims->nlg[2];

        ssize_t i,j,k;

        // (1) compute mean velocities
        horizontal_mean(dims, &vel_advecting[0], &vel_ing_mean_[0]);
        horizontal_mean(dims, &vel_advected[0], &vel_ed_mean_[0]);

        for(i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;
                    vel_ing_mean[ijk] = vel_ing_mean_[k];
                    vel_ed_mean[ijk] = vel_ed_mean_[k];

                    vel_ing_fluc[ijk] = vel_advecting[ijk] - vel_ing_mean[ijk];
                    vel_ed_fluc[ijk] = vel_advected[ijk] - vel_ed_mean[ijk];
                }
            }
        }


        // (2) compute flux
        const ssize_t stencil[3] = {istride,jstride,1};
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2 * sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;

        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2 * sp1_ing;
        const ssize_t sm1_ing = -sp1_ing;

        imin = 1;
        jmin = 1;
        kmin = 1;

        imax = dims->nlg[0]-2;
        jmax = dims->nlg[1]-2;
        kmax = dims->nlg[2]-2;

        double *flux_eddy_mean = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *flux_mean_eddy = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *mean_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        eddy_flux[ijk] = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0_half[k];
                        flux_mean_eddy[ijk] = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0_half[k];
                        flux_eddy_mean[ijk] = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0_half[k];
                        mean_flux[k] = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0_half[k];

//                        flux[ijk] = flux_eddy + flux_m1 + flux_m2 + flux_mean;
//                        flux[ijk] = eddy_flux[ijk] + flux_mean_eddy[ijk] + flux_eddy_mean[ijk] + mean_flux[k];
                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
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
                        eddy_flux[ijk] = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0_half[k+1];
                        flux_mean_eddy[ijk] = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0_half[k+1];
                        flux_eddy_mean[ijk] = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0_half[k+1];
                        mean_flux[k] = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0_half[k+1];

//                        flux[ijk] = flux_eddy + flux_m1 + flux_m2 + flux_mean;
//                        flux[ijk] = eddy_flux[ijk] + flux_mean_eddy[ijk] + flux_eddy_mean[ijk] + mean_flux[k];
                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
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
                        eddy_flux[ijk] = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0[k+1];
                        flux_mean_eddy[ijk] = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_fluc[ijk+sm1_ed],vel_ed_fluc[ijk],vel_ed_fluc[ijk+sp1_ed],vel_ed_fluc[ijk+sp2_ed]) * rho0[k+1];
                        flux_eddy_mean[ijk] = interp_2(vel_ing_fluc[ijk],vel_ing_fluc[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0[k+1];
                        mean_flux[k] = interp_2(vel_ing_mean[ijk],vel_ing_mean[ijk+sp1_ing]) *
                                                interp_4(vel_ed_mean[ijk+sm1_ed],vel_ed_mean[ijk],vel_ed_mean[ijk+sp1_ed],vel_ed_mean[ijk+sp2_ed]) * rho0[k+1];

//                        flux[ijk] = flux_eddy + flux_m1 + flux_m2 + flux_mean;
//                        flux[ijk] = eddy_flux[ijk] + flux_mean_eddy[ijk] + flux_eddy_mean[ijk] + mean_flux[k];
                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0[k];
                    }
                }
            }
        }


        horizontal_mean(dims, &eddy_flux[0], &mean_eddy_flux[0]);
        for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;

                        flux[ijk] = mean_flux[k] + flux_mean_eddy[ijk] + flux_eddy_mean[ijk] + mean_eddy_flux[k];
                    }
                }
            }

        int ok_nan = 0;
        for(i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;

                    if(isnan(flux[ijk])){ok_nan = ok_nan + 1;}
                }
            }
        }
        if(ok_nan > 1){printf("problem nan flux: count = %d\n",ok_nan);}
//        else{printf("no nans in MA fluxes\n");}

        momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                                tendency, d_advected, d_advecting);
        free(flux);
        free(flux_old);
//        free(flux_ql);
        free(flux_eddy_mean);
        free(flux_mean_eddy);
        free(eddy_flux);
        free(mean_eddy_flux);
        free(mean_flux);

        free(vel_ed_mean);
        free(vel_ing_mean);
        free(vel_ed_mean_);
        free(vel_ing_mean_);
        free(vel_ed_fluc);
        free(vel_ing_fluc);
        return;
    }
