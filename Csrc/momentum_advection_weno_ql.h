#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "flux_divergence.h"
#include<stdio.h>
#include<math.h>
#include <stdlib.h>


void weno_fifth_order_m_decomp(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){
        if (d_advected==1 && d_advecting==1){
            printf("WENO5 decompositioned Momentum Transport \n");}

        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *flux_old = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        ssize_t imin = 0;
        ssize_t jmin = 0;
        ssize_t kmin = 0;
        ssize_t imax = dims->nlg[0];
        ssize_t jmax = dims->nlg[1];
        ssize_t kmax = dims->nlg[2];

        const ssize_t stencil[3] = {istride,jstride,1};

        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2*sp1_ed ;
        const ssize_t sp3_ed = 3*sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;
        const ssize_t sm2_ed = -2*sp1_ed ;

        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2*sp1_ing ;
        const ssize_t sm1_ing = -sp1_ing ;


        // (1) average advecting and advected velocity
        double *vel_advected_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_advected_mean = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_advected_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_advecting_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_advecting_mean = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_advecting_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);

//        horizontal_mean(dims, &vel_advecting[0], &vel_advecting_mean[0]);
        horizontal_mean(dims, &vel_advecting[0], &vel_advecting_mean_[0]);
//        horizontal_mean(dims, &vel_advected[0], &vel_advected_mean[0]);
        horizontal_mean(dims, &vel_advected[0], &vel_advected_mean_[0]);

        // (2) compute eddy fields
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    int ijk = ishift + jshift + k;
                    vel_advecting_mean[ijk] = vel_advecting_mean_[k];
                    vel_advected_mean[ijk] = vel_advected_mean_[k];

                    vel_advecting_fluc[ijk] = vel_advecting[ijk] - vel_advecting_mean[ijk];
                    vel_advected_fluc[ijk] = vel_advected[ijk] - vel_advected_mean[ijk];
//                    vel_advecting_fluc[ijk] = vel_advecting[ijk] - vel_advecting_mean_[k];
//                    vel_advected_fluc[ijk] = vel_advected[ijk] - vel_advected_mean_[k];

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
                        double diff = vel_advecting[ijk]-(vel_advecting_mean[ijk] + vel_advecting_fluc[ijk]);
                        if(fabs(diff)>0.0000001){
                            ok = 1;
                            printf("!!! decomposition advecting , ijk= %d, diff = %f, vel = %f \n", ijk, diff, vel_advecting[ijk]);}

                        diff = vel_advected[ijk]-(vel_advected_mean[ijk] + vel_advected_fluc[ijk]);
                        if(fabs(diff)>0.0000001){
                            ok = 1;
                            printf("!!! decomposition advected, ijk= %d, diff = %f, vel = %f \n", ijk, diff, vel_advected[ijk]);}

                        diff = vel_advecting[ijk]-(vel_advecting_mean_[k] + vel_advecting_fluc[ijk]);
                        if(fabs(diff)>0.0000001){
                            ok = 1;
                            printf("decomposition advecting , ijk= %d, diff = %f, vel = %f \n", ijk, diff, vel_advecting[ijk]);}

                        diff = vel_advected[ijk]-(vel_advected_mean_[k] + vel_advected_fluc[ijk]);
                        if(fabs(diff)>0.0000001){
                            ok = 1;
                            printf("decomposition advected, ijk= %d, diff = %f, vel = %f \n", ijk, diff, vel_advected[ijk]);}
                }
            }
        }
//        if(ok==0){printf("good decomposition \n");}
        free(vel_advected_mean_);
        free(vel_advecting_mean_);


        // (3) Compute Fluxes
        // mix_flux_one = <u_ing> u_ed'
        // mix_flux_two = u_ing' <u_ed>
        // eddy_flux = u_ing' u_ed'
        // mean_flux = <u_ing> <u_ed>
        imin = 2;
        jmin = 2;
        kmin = 2;
        imax = dims->nlg[0]-3;
        jmax = dims->nlg[1]-3;
        kmax = dims->nlg[2]-3;
        double *mix_flux_one = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mix_flux_two = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mean_flux = (double *)malloc(sizeof(double)*dims->nlg[2]);        // ??? 1D profile sufficient!?

//        double phip;
//        double phim;
//        double phip_fluc;      //???? do I need const double phip ??? Difference to declaring it within loop?
//        double phim_fluc;
//        double phip_mean;
//        double phim_mean;
//        double vel_adv;

        if (d_advected !=2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
//                        printf("d_adv != 2, d_ing != 2\n");

                        const double phip_fluc = interp_weno5(vel_advected_fluc[ijk + sm2_ed],
                                                 vel_advected_fluc[ijk + sm1_ed],
                                                 vel_advected_fluc[ijk],
                                                 vel_advected_fluc[ijk + sp1_ed],
                                                 vel_advected_fluc[ijk + sp2_ed]);
                        const double phip_mean = interp_weno5(vel_advected_mean[ijk + sm2_ed],
                                                 vel_advected_mean[ijk + sm1_ed],
                                                 vel_advected_mean[ijk],
                                                 vel_advected_mean[ijk + sp1_ed],
                                                 vel_advected_mean[ijk + sp2_ed]);
//                        phip_mean = interp_weno5(vel_advected_mean_[k],vel_advected_mean_[k],vel_advected_mean_[k],vel_advected_mean_[k],vel_advected_mean_[k]);

                        const double phim_fluc = interp_weno5(vel_advected_fluc[ijk + sp3_ed],
                                                 vel_advected_fluc[ijk + sp2_ed],
                                                 vel_advected_fluc[ijk + sp1_ed],
                                                 vel_advected_fluc[ijk],
                                                 vel_advected_fluc[ijk + sm1_ed]);
                        const double phim_mean = interp_weno5(vel_advected_mean[ijk + sp3_ed],
                                                 vel_advected_mean[ijk + sp2_ed],
                                                 vel_advected_mean[ijk + sp1_ed],
                                                 vel_advected_mean[ijk],
                                                 vel_advected_mean[ijk + sm1_ed]);
//                        phim_mean = interp_weno5(vel_advected_mean_[k],vel_advected_mean_[k],vel_advected_mean_[k],vel_advected_mean_[k],vel_advected_mean_[k]);

//                        const double phip_mean = vel_advected_mean_[k];  // interpolation of mean profiles in x-, y-direction has no effect
//                        const double phim_mean = vel_advected_mean_[k];  // interpolation of mean profiles in x-, y-direction has no effect

                        const double vel_adv_mean = interp_4(vel_advecting_mean[ijk + sm1_ing],
                                                             vel_advecting_mean[ijk],
                                                             vel_advecting_mean[ijk + sp1_ing],
                                                             vel_advecting_mean[ijk] + sp2_ing);
//                        const double vel_adv_mean = vel_advecting_mean_[k];
                        const double vel_adv_fluc = interp_4(vel_advecting_fluc[ijk + sm1_ing],
                                                            vel_advecting_fluc[ijk],
                                                            vel_advecting_fluc[ijk + sp1_ing],
                                                            vel_advecting_fluc[ijk + sp2_ing]);

                        // (a) mix_flux_one = <u_ing> u_ed'
                        // (b) mean_flux = <u_ing><u_ed>
                        mix_flux_one[ijk] = 0.5 * ((vel_adv_mean+fabs(vel_adv_mean))*phip_fluc + (vel_adv_mean-fabs(vel_adv_mean))*phim_fluc)*rho0_half[k] ;
                        mean_flux[k] = 0.5 * ((vel_adv_mean+fabs(vel_adv_mean))*phip_mean + (vel_adv_mean-fabs(vel_adv_mean))*phim_mean)*rho0_half[k] ;

                        // (c) mix_flux_two = u_ing'<u_ed>
                        // (d) eddy_flux = u_ing' u_ed'
                        mix_flux_two[ijk] = 0.5 * ((vel_adv_fluc+fabs(vel_adv_fluc))*phip_mean + (vel_adv_fluc-fabs(vel_adv_fluc))*phim_mean)*rho0_half[k] ;
                        eddy_flux[ijk] = 0.5 * ((vel_adv_fluc+fabs(vel_adv_fluc))*phip_fluc + (vel_adv_fluc-fabs(vel_adv_fluc))*phim_fluc)*rho0_half[k] ;

                        flux[ijk] = mean_flux[k] + mix_flux_one[ijk] + mix_flux_two[ijk] + eddy_flux[ijk];

                        if(isnan(flux[ijk])) {
                            printf("Nan in flux, d1!=2, d2!=2\n");
                        }


                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed]);

                        const double vel_adv = interp_4(vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing]);

                        flux_old[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0_half[k] ;
//                        flux[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0_half[k] ;
//                        printf("old flux: %f\n", fabs(flux_old[ijk]));

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
//                        printf("d_adv = 2, d_ing = 2\n");

                        const double phip_fluc = interp_weno5(vel_advected_fluc[ijk + sm2_ed],
                                                 vel_advected_fluc[ijk + sm1_ed],
                                                 vel_advected_fluc[ijk],
                                                 vel_advected_fluc[ijk + sp1_ed],
                                                 vel_advected_fluc[ijk + sp2_ed]);
                        const double phip_mean = interp_weno5(vel_advected_mean[ijk + sm2_ed],
                                                 vel_advected_mean[ijk + sm1_ed],
                                                 vel_advected_mean[ijk],
                                                 vel_advected_mean[ijk + sp1_ed],
                                                 vel_advected_mean[ijk + sp2_ed]);

                        const double phim_fluc = interp_weno5(vel_advected_fluc[ijk + sp3_ed],
                                                 vel_advected_fluc[ijk + sp2_ed],
                                                 vel_advected_fluc[ijk + sp1_ed],
                                                 vel_advected_fluc[ijk],
                                                 vel_advected_fluc[ijk + sm1_ed]);
                        const double phim_mean = interp_weno5(vel_advected_mean[ijk + sp3_ed],
                                                 vel_advected_mean[ijk + sp2_ed],
                                                 vel_advected_mean[ijk + sp1_ed],
                                                 vel_advected_mean[ijk],
                                                 vel_advected_mean[ijk + sm1_ed]);
//
//                        const double phip_mean = vel_advected_mean_[k];   // interpolation of mean profiles in x-, y-direction has no effect
//                        const double phim_mean = vel_advected_mean_[k];   // interpolation of mean profiles in x-, y-direction has no effect


                        const double vel_adv_mean = interp_4(vel_advecting_mean[ijk + sm1_ing],
                                                            vel_advecting_mean[ijk],
                                                            vel_advecting_mean[ijk + sp1_ing],
                                                            vel_advecting_mean[ijk] + sp2_ing);
//                        const double vel_adv_mean = vel_advecting_mean_[k];
                        const double vel_adv_fluc = interp_4(vel_advecting_fluc[ijk + sm1_ing],
                                                            vel_advecting_fluc[ijk],
                                                            vel_advecting_fluc[ijk + sp1_ing],
                                                            vel_advecting_fluc[ijk + sp2_ing]);

                        // (a) mix_flux_one = <u_ing> u_ed' && mean_flux = <u_ing><u_ed>
//                        vel_adv = vel_advecting_mean[k];    // interpolation of mean profiles in x-, y-direction has no effect
                        mix_flux_one[ijk] = 0.5 * ((vel_adv_mean+fabs(vel_adv_mean))*phip_fluc + (vel_adv_mean-fabs(vel_adv_mean))*phim_fluc)*rho0_half[k+1] ;
                        mean_flux[k] = 0.5 * ((vel_adv_mean+fabs(vel_adv_mean))*phip_mean + (vel_adv_mean-fabs(vel_adv_mean))*phim_mean)*rho0_half[k+1] ;

                        // (b) mix_flux_two = u_ing'<u_ed> && eddy_flux = u_ing' u_ed'
                        mix_flux_two[ijk] = 0.5 * ((vel_adv_fluc+fabs(vel_adv_fluc))*phip_mean + (vel_adv_fluc-fabs(vel_adv_fluc))*phim_mean)*rho0_half[k+1] ;
                        eddy_flux[ijk] = 0.5 * ((vel_adv_fluc+fabs(vel_adv_fluc))*phip_fluc + (vel_adv_fluc-fabs(vel_adv_fluc))*phim_fluc)*rho0_half[k+1] ;

                        flux[ijk] = mean_flux[k] + mix_flux_one[ijk] + mix_flux_two[ijk] + eddy_flux[ijk];
                        if(isnan(flux[ijk])) {
                            printf("Nan in flux, d1=2, d2=2\n");
                        }


                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed]);

                        const double vel_adv = interp_4(vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing]);

                        flux_old[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0_half[k+1];
//                        flux[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0_half[k+1];

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
//                        printf("d_adv != 2, d_ing = 2\n");

                        const double phip_fluc = interp_weno5(vel_advected_fluc[ijk + sm2_ed],
                                                 vel_advected_fluc[ijk + sm1_ed],
                                                 vel_advected_fluc[ijk],
                                                 vel_advected_fluc[ijk + sp1_ed],
                                                 vel_advected_fluc[ijk + sp2_ed]);
                        const double phip_mean = interp_weno5(vel_advected_mean[ijk + sm2_ed],
                                                 vel_advected_mean[ijk + sm1_ed],
                                                 vel_advected_mean[ijk],
                                                 vel_advected_mean[ijk + sp1_ed],
                                                 vel_advected_mean[ijk + sp2_ed]);

                        const double phim_fluc = interp_weno5(vel_advected_fluc[ijk + sp3_ed],
                                                 vel_advected_fluc[ijk + sp2_ed],
                                                 vel_advected_fluc[ijk + sp1_ed],
                                                 vel_advected_fluc[ijk],
                                                 vel_advected_fluc[ijk + sm1_ed]);
                        const double phim_mean = interp_weno5(vel_advected_mean[ijk + sp3_ed],
                                                 vel_advected_mean[ijk + sp2_ed],
                                                 vel_advected_mean[ijk + sp1_ed],
                                                 vel_advected_mean[ijk],
                                                 vel_advected_mean[ijk + sm1_ed]);

//                        const double phip_mean = vel_advected_mean_[k];   // interpolation of mean profiles in x-, y-direction has no effect
//                        const double phim_mean = vel_advected_mean_[k];   // interpolation of mean profiles in x-, y-direction has no effect

                        const double vel_adv_mean = interp_4(vel_advecting_mean[ijk + sm1_ing],
                                                            vel_advecting_mean[ijk],
                                                            vel_advecting_mean[ijk + sp1_ing],
                                                            vel_advecting_mean[ijk] + sp2_ing);
//                        const double vel_adv_mean = vel_advecting_mean_[k];
                        const double vel_adv_fluc = interp_4(vel_advecting_fluc[ijk + sm1_ing],
                                                            vel_advecting_fluc[ijk],
                                                            vel_advecting_fluc[ijk + sp1_ing],
                                                            vel_advecting_fluc[ijk + sp2_ing]);

                        // (a) mix_flux_one = <u_ing> u_ed' && mean_flux = <u_ing><u_ed>
                        mix_flux_one[ijk] = 0.5 * ((vel_adv_mean+fabs(vel_adv_mean))*phip_fluc + (vel_adv_mean-fabs(vel_adv_mean))*phim_fluc)*rho0[k] ;
                        mean_flux[k] = 0.5 * ((vel_adv_mean+fabs(vel_adv_mean))*phip_mean + (vel_adv_mean-fabs(vel_adv_mean))*phim_mean)*rho0[k] ;

                        // (b) mix_flux_two = u_ing'<u_ed> && eddy_flux = u_ing' u_ed'
                        mix_flux_two[ijk] = 0.5 * ((vel_adv_fluc+fabs(vel_adv_fluc))*phip_mean + (vel_adv_fluc-fabs(vel_adv_fluc))*phim_mean)*rho0[k] ;
                        eddy_flux[ijk] = 0.5 * ((vel_adv_fluc+fabs(vel_adv_fluc))*phip_fluc + (vel_adv_fluc-fabs(vel_adv_fluc))*phim_fluc)*rho0[k] ;

                        flux[ijk] = mean_flux[k] + mix_flux_one[ijk] + mix_flux_two[ijk] + eddy_flux[ijk];
                        if(isnan(flux[ijk])) {
                            printf("Nan in flux, d1=2, d2!=2\n");
                        }


                        // Original Flux computation
                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed]);

                        const double vel_adv = interp_4(vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing]);

                        flux_old[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0[k];
//                        flux[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0[k];

                    }
                }
            }
        }

//        int count = 0;
//        double max = 0.0;
//        for(ssize_t i=imin;i<imax;i++){
//                const ssize_t ishift = i*istride;
//                for(ssize_t j=jmin;j<jmax;j++){
//                    const ssize_t jshift = j*jstride;
//                    for(ssize_t k=kmin;k<kmax;k++){
//                        const int ijk = ishift + jshift + k;
//                        const double diff = fabs(flux[ijk]-flux_old[ijk]);
//                        if(diff>0.05*fabs(flux_old[ijk])){count = count + 1;}
//                        if(max<diff){max=diff;}
//                 }
//             }
//         }
//         if(count > 0){printf("MA Fluxes differences: count = %d, max = %f\n", count, max);}


//        printf("flux: %f\n", flux);
        //        printf("values[gw] = %f\n", values[gw]);

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




void weno_fifth_order_m_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
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
        const ssize_t sp2_ed = 2*sp1_ed ;
        const ssize_t sp3_ed = 3*sp1_ed ;
        const ssize_t sm1_ed = -sp1_ed ;
        const ssize_t sm2_ed = -2*sp1_ed ;

        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2*sp1_ing ;
        const ssize_t sm1_ing = -sp1_ing ;


        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;

                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed]);

                        const double vel_adv = interp_4(vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing]);

                        flux[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0_half[k] ;
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

                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed]);

                        const double vel_adv = interp_4(vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing]);

                        flux[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0_half[k+1];
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

                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed]);

                        const double vel_adv = interp_4(vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing]);

                        flux[ijk] = 0.5 * ((vel_adv+fabs(vel_adv))*phip + (vel_adv-fabs(vel_adv))*phim)*rho0[k];
                    }
                }
            }
        }
        momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                                tendency, d_advected, d_advecting);
        free(flux);
        return;
    }
