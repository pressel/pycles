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







// !!!!!! test if doing the correct thing !!!!!!
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

//        const ssize_t imin = 1;
//        const ssize_t jmin = 1;
//        const ssize_t kmin = 1;
//        const ssize_t imax = dims->nlg[0]-2;
//        const ssize_t jmax = dims->nlg[1]-2;
//        const ssize_t kmax = dims->nlg[2]-2;
        ssize_t imin = 0;
        ssize_t jmin = 0;
        ssize_t kmin = 0;
        ssize_t imax = dims->nlg[0];
        ssize_t jmax = dims->nlg[1];
        ssize_t kmax = dims->nlg[2];

        const ssize_t stencil[3] = {istride,jstride,1};
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2 * sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;

        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2 * sp1_ing;
        const ssize_t sm1_ing = -sp1_ing;

        // (1) average advecting and advected velocity
        double *vel_advected_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_advected_mean = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_advecting_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_advecting_mean = (double *)malloc(sizeof(double) * dims->nlg[2]);

        horizontal_mean(dims, &vel_advecting[0], &vel_advecting_mean[0]);
        horizontal_mean(dims, &vel_advected[0], &vel_advected_mean[0]);

        // (2) compute eddy fields
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    int ijk = ishift + jshift + k;
                    vel_advecting_fluc[ijk] = vel_advecting[ijk] - vel_advecting_mean[k];
                    vel_advected_fluc[ijk] = vel_advected[ijk] - vel_advected_mean[k];

                    if(isnan(vel_advected_fluc[ijk])) {
                        printf("Nan in vel_advected_fluc\n");
                    }
                    if(isnan(vel_advecting_fluc[ijk])) {
                        printf("Nan in vel_advecting_fluc\n");
                    }
                }
            }
        }
        int ok = 0;
        for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const int ijk = ishift + jshift + k;
                        double diff = vel_advecting[ijk]-(vel_advecting_mean[k] + vel_advecting_fluc[ijk]);
                        if(fabs(diff)>0.0000001){
                            ok = 1;
                            printf("decomposition advecting , ijk= %d, diff = %f, vel = %f \n", ijk, diff, vel_advecting[ijk]);}
                        diff = vel_advected[ijk]-(vel_advected_mean[k] + vel_advected_fluc[ijk]);
                        if(fabs(diff)>0.0000001){
                            ok = 1;
                            printf("decomposition advected, ijk= %d, diff = %f, vel = %f \n", ijk, diff, vel_advected[ijk]);}
                }
            }
        }

        // (3) Compute Fluxes
        // mix_flux_one = <u_ing> u_ed'
        // mix_flux_two = u_ing' <u_ed>
        // eddy_flux = u_ing' u_ed'
        // mean_flux = <u_ing> <u_ed>
        imin = 1;
        jmin = 1;
        kmin = 1;
        imax = dims->nlg[0]-2;
        jmax = dims->nlg[1]-2;
        kmax = dims->nlg[2]-2;
        double *mix_flux_one = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mix_flux_two = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mean_flux = (double *)malloc(sizeof(double)*dims->nlg[2]);        // ??? 1D profile sufficient!?

        double vel_ing_mean = 0.0;
        double vel_ed_mean = 0.0;
        double vel_ing_fluc;
        double vel_ed_fluc;

        if (d_advected !=2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        vel_ing_fluc = interp_2(vel_advecting_fluc[ijk],vel_advecting_fluc[ijk+sp1_ing]);
                        vel_ed_fluc = interp_4(vel_advected_fluc[ijk+sm1_ed],vel_advected_fluc[ijk],vel_advected_fluc[ijk+sp1_ed],vel_advected_fluc[ijk+sp2_ed]);
                        vel_ing_mean = interp_2(vel_advecting_mean[k],vel_advecting_mean[k]);
                        vel_ed_mean = interp_4(vel_advected_mean[k],vel_advected_mean[k],vel_advected_mean[k],vel_advected_mean[k]);
//                        vel_ed_mean = vel_advected_mean[k];    // interpolation of mean profiles in x-, y-direction has no effect
//                        vel_ing_mean = vel_advecting_mean[k];    // interpolation of mean profiles in x-, y-direction has no effect

                        // (a) mix_flux_one = <u_ing> u_ed'
                        // (b) mix_flux_two = u_ing'<u_ed>
                        // (c) mean_flux = <u_ing><u_ed>
                        // (d) eddy_flux = u_ing' u_ed'
                        mix_flux_one[ijk] = vel_ing_mean * vel_ed_fluc * rho0_half[k] ;
                        mix_flux_two[ijk] = vel_ing_fluc * vel_ed_mean * rho0_half[k] ;
                        mean_flux[k] = vel_ing_mean * vel_ed_mean * rho0_half[k] ;
                        eddy_flux[ijk] = vel_ing_fluc * vel_ed_fluc * rho0_half[k] ;

                        flux[ijk] = mean_flux[k] + mix_flux_one[ijk] + mix_flux_two[ijk] + eddy_flux[ijk];
                        if(isnan(flux[ijk])) {printf("Nan in flux, d1!=2, d2!=2\n");}

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
                        vel_ing_fluc = interp_2(vel_advecting_fluc[ijk],vel_advecting_fluc[ijk+sp1_ing]);
                        vel_ed_fluc = interp_4(vel_advected_fluc[ijk+sm1_ed],vel_advected_fluc[ijk],vel_advected_fluc[ijk+sp1_ed],vel_advected_fluc[ijk+sp2_ed]);
                        vel_ing_mean = interp_2(vel_advecting_mean[k],vel_advecting_mean[k]);
                        vel_ed_mean = interp_4(vel_advected_mean[k],vel_advected_mean[k],vel_advected_mean[k],vel_advected_mean[k]);
//                        vel_ed_mean = vel_advected_mean[k];    // interpolation of mean profiles in x-, y-direction has no effect
//                        vel_ing_mean = vel_advecting_mean[k];    // interpolation of mean profiles in x-, y-direction has no effect

                        // (a) mix_flux_one = <u_ing> u_ed'
                        // (b) mix_flux_two = u_ing'<u_ed>
                        // (c) mean_flux = <u_ing><u_ed>
                        // (d) eddy_flux = u_ing' u_ed'
                        mix_flux_one[ijk] = vel_ing_mean * vel_ed_fluc * rho0_half[k+1] ;
                        mix_flux_two[ijk] = vel_ing_fluc * vel_ed_mean * rho0_half[k+1] ;
                        mean_flux[k] = vel_ing_mean * vel_ed_mean * rho0_half[k+1] ;
                        eddy_flux[ijk] = vel_ing_fluc * vel_ed_fluc * rho0_half[k+1] ;

                        flux[ijk] = mean_flux[k] + mix_flux_one[ijk] + mix_flux_two[ijk] + eddy_flux[ijk];
                        if(isnan(flux[ijk])) {printf("Nan in flux, d1=2, d2=2\n");}

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
                        vel_ing_fluc = interp_2(vel_advecting_fluc[ijk],vel_advecting_fluc[ijk+sp1_ing]);
                        vel_ed_fluc = interp_4(vel_advected_fluc[ijk+sm1_ed],vel_advected_fluc[ijk],vel_advected_fluc[ijk+sp1_ed],vel_advected_fluc[ijk+sp2_ed]);
                        vel_ing_mean = interp_2(vel_advecting_mean[k],vel_advecting_mean[k]);
                        vel_ed_mean = interp_4(vel_advected_mean[k],vel_advected_mean[k],vel_advected_mean[k],vel_advected_mean[k]);
//                        vel_ed_mean = vel_advected_mean[k];    // interpolation of mean profiles in x-, y-direction has no effect
//                        vel_ing_mean = vel_advecting_mean[k];    // interpolation of mean profiles in x-, y-direction has no effect

                        // (a) mix_flux_one = <u_ing> u_ed'
                        // (b) mix_flux_two = u_ing'<u_ed>
                        // (c) mean_flux = <u_ing><u_ed>
                        // (d) eddy_flux = u_ing' u_ed'
                        mix_flux_one[ijk] = vel_ing_mean * vel_ed_fluc * rho0[k] ;
                        mix_flux_two[ijk] = vel_ing_fluc * vel_ed_mean * rho0[k] ;
                        mean_flux[k] = vel_ing_mean * vel_ed_mean * rho0[k] ;
                        eddy_flux[ijk] = vel_ing_fluc * vel_ed_fluc * rho0[k] ;

                        flux[ijk] = mean_flux[k] + mix_flux_one[ijk] + mix_flux_two[ijk] + eddy_flux[ijk];
                        if(isnan(flux[ijk])) {printf("Nan in flux, d1=2, d2!=2\n");}

                        flux_old[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0[k];
                    }
                }
            }
        }

        momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                                tendency, d_advected, d_advecting);

        free(flux);
        free(flux_old);
        free(mix_flux_one);
        free(mix_flux_two);
        free(eddy_flux);
        free(mean_flux);
        free(vel_advecting_fluc);
        free(vel_advected_fluc);
        free(vel_advecting_mean);
        free(vel_advected_mean);


        return;
    }