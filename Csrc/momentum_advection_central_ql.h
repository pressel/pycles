#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "flux_divergence.h"

#include "cc_statistics.h"
#include<stdio.h>

// !!!!!!!! Need to Fix Flux computation (subtracting eddy flux) !!!!!
// !!!!!!!! need to be able to output eddy_flux???

// QUESTIONS / NOTES:
// need to be able to output eddy_flux???
// how to calculate horizontal mean???
// line 36 pp.: sp1_ed = stencil[d_advecting]; sp1_ing = stencil[d_advected]; --> confused about 'ed' vs. 'ing'

// dimension of flux, i.e. advection of which momentum in which direction:
//      --> second_order_m is called in MomentumAdvection.pyx (by calling compute_advective_tendencies_m), for a certain flux (advected and advecting velocities defined)

// IDEA:
// (1) interpolate velocity fields (dist. btw. advecting = advected vs. advecting != advected)
// (2) average interpolated velocity fields:
//          Pa.HorizontalMean(Gr, &DV.values[var_shift]), used e.g. in AuxiliaryStatistics
// (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
// (4) compute mean eddy flux
// (5) compute QL flux: flux = flux - eddy_flux + mean_eddy_flux

void second_order_m_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

//        if (d_advected==1 && d_advecting==1){
//            printf("2nd order QL Momentum Transport \n");}

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);     // malloc allocates size of uninitialized storage; in this case allocates memory for (nlg[0]+nlg[1]+nlg[2])*sizeof(double))
        double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_int_ed = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_int_ing = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        // double *vel_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_mean_ed = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_mean_ing = (double *)malloc(sizeof(double) * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];        // istride = "jump for getting to next i index"
        // ssize_t: predefined type of signed integer (= signed integral type) (eqvuivalent to Py_ssize_t in python, cython)
        const ssize_t jstride = dims->nlg[2];                       // jstride = "jump for getting to next j index"

        const ssize_t imin = 0;
        const ssize_t jmin = 0;
        const ssize_t kmin = 0;

        const ssize_t imax = dims->nlg[0]-1;
        const ssize_t jmax = dims->nlg[1]-1;
        const ssize_t kmax = dims->nlg[2]-1;

        const ssize_t stencil[3] = {istride,jstride,1};             // array, containing 3 elements
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp1_ing = stencil[d_advected];                // d_advecting, d_advected: given as parameters to this function


        // (1) interpolate velocity fields
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    vel_int_ing[ijk] = interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]);
                    if (d_advected != d_advecting){
                        vel_int_ed[ijk] = interp_2(vel_advected[ijk],vel_advected[ijk+sp1_ed]);}
                    else{
                        vel_int_ed[ijk] = interp_2(vel_advected[ijk],vel_advected[ijk+sp1_ed]);}
//                        vel_int_ed[ijk] = vel_int_ing[ijk];}
                }
            }
        }


        // (2) average interpolated velocity fields
        //vel_mean_ing = Pa.HorizontalMean(Gr, &vel_int_advecting);
//        const ssize_t dims->gw;
//        int d_ad = d_advected;
//        int d_ing = d_advecting;
//        printf("flux: %d, %d; vel_int[gw+2] = %f\n", d_ad, d_ing, values[0]);
        horizontal_mean(dims, &vel_int_ing[0], &vel_mean_ing[0]);
        if (d_advected != d_advecting){
            horizontal_mean(dims, &vel_int_ed[0], &vel_mean_ed[0]);
            }
        else {
            horizontal_mean(dims, &vel_int_ed[0], &vel_mean_ed[0]);
//            vel_mean_ed = vel_mean_ing ;
            }


        // (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
        if (d_advected != 2 && d_advecting !=2){                    // exclude w.u, w.v, w.w, u.w, v.w (advection by or of vertical velocity)
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        // vel_fluc = vel_int - vel_mean
                        // eddy_flux[ijk] = vel_fluc[ijk]*vel_fluc[ijk]*rho0_half[k];       // need to be able to output eddy_flux???
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
                        // vel_fluc = vel_int - vel_meanc
                        // eddy_flux[ijk] = vel_fluc[ijk]*vel_fluc[ijk]*rho0_half[k+1];       // need to be able to output eddy_flux???
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
                        // vel_fluc = vel_int - vel_mean
                        // eddy_flux[ijk] = vel_fluc[ijk]*vel_fluc[ijk]*rho0[k];       // need to be able to output eddy_flux???
                        eddy_flux[ijk] = (vel_int_ing[ijk] - vel_mean_ing[k]) * (vel_int_ed[ijk] - vel_mean_ed[k]) * rho0[k];
                        flux[ijk] = (vel_int_ing[ijk] * vel_int_ed[ijk]) * rho0[k];
                    }
                }
            }
        }

        // (4) compute mean eddy flux
        horizontal_mean(dims, &eddy_flux[0], &mean_eddy_flux[0]);
        //mean_eddy_flux = Pa.HorizontalMean(Gr, &eddy_flux);


        // (5) compute QL flux: flux = flux - eddy_flux + mean_eddy_flux
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = flux[ijk] - eddy_flux[ijk] + mean_eddy_flux[k];
//                    flux[ijk] = flux[ijk];
//                    flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing])
//                            *interp_2(vel_advected[ijk],vel_advected[ijk + sp1_ed]) )*rho0[k];
                }
            }
        }


        momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                                tendency, d_advected, d_advecting);

        //Free dynamically allocated array

        free(eddy_flux);
        free(mean_eddy_flux);
        free(vel_int_ing);
        free(vel_int_ed);
        free(vel_mean_ed);
        free(vel_mean_ing);     // gives error message

        free(flux);
        return;
    }





void fourth_order_m_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        if (d_advected==1 && d_advecting==1){
            printf("4th order QL Momentum Transport \n");}

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);     // malloc allocates size of uninitialized storage; in this case allocates memory for (nlg[0]+nlg[1]+nlg[2])*sizeof(double))
        double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_int_ed = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_int_ing = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        // double *vel_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_mean_ed = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_mean_ing = (double *)malloc(sizeof(double) * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];        // istride = "jump for getting to next i index"
        // ssize_t: predefined type of signed integer (= signed integral type) (eqvuivalent to Py_ssize_t in python, cython)
        const ssize_t jstride = dims->nlg[2];                       // jstride = "jump for getting to next j index"

        const ssize_t imin = 1;
        const ssize_t jmin = 1;
        const ssize_t kmin = 1;

        const ssize_t imax = dims->nlg[0]-2;
        const ssize_t jmax = dims->nlg[1]-2;
        const ssize_t kmax = dims->nlg[2]-2;

        const ssize_t stencil[3] = {istride,jstride,1};             // array, containing 3 elements
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2 * sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;

        const ssize_t sp1_ing = stencil[d_advected];                // d_advecting, d_advected: given as parameters to this function
        const ssize_t sp2_ing = 2 * sp1_ing;
        const ssize_t sm1_ing = -sp1_ing;



        // (1) interpolate velocity fields
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    vel_int_ing[ijk] = interp_4(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing]);
                    vel_int_ed[ijk] = interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed]);
                }
            }
        }


        // (2) average interpolated velocity fields
        //vel_mean_ing = Pa.HorizontalMean(Gr, &vel_int_advecting);
        horizontal_mean(dims, &vel_int_ing[0], &vel_mean_ing[0]);
        horizontal_mean(dims, &vel_int_ed[0], &vel_mean_ed[0]);


        // (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
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

        //Free dynamically allocated array
        free(eddy_flux);
        free(mean_eddy_flux);
        free(vel_int_ing);
        free(vel_int_ed);
        free(vel_mean_ed);
        free(vel_mean_ing);

        free(flux);
        return;
    }