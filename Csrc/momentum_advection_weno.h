#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "flux_divergence.h"
#include<stdio.h>

void weno_third_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
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
        const ssize_t sm1_ed = -stencil[d_advecting];
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2*stencil[d_advecting];
        const ssize_t sp1_ing = stencil[d_advected];


        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno3(vel_advected[ijk + sm1_ed],
                                                        vel_advected[ijk],
                                                        vel_advected[ijk + sp1_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno3(vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk]);

                        const double vel_adv = interp_2(vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing]);

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
                        const double phip = interp_weno3(vel_advected[ijk + sm1_ed],
                                            vel_advected[ijk],
                                            vel_advected[ijk + sp1_ed]);

                        // Up wind for negative velocity
                        const double phim = interp_weno3(vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk]);
                        const double vel_adv = interp_2(vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing]);

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
                        const double phip = interp_weno3(vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed]);

                        // Up wind for negative velocity
                        const double phim = interp_weno3(vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk]);

                        const double vel_adv = interp_2(vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing]);

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

void weno_fifth_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
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
        else if(d_advecting == 2){
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
         else if(d_advected == 2){
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
                                                        vel_advecting[ijk] ,
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing] );

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

void weno_seventh_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
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

        const ssize_t sm3_ed = -3*stencil[d_advecting];
        const ssize_t sm2_ed = -2*stencil[d_advecting];
        const ssize_t sm1_ed = -stencil[d_advecting];
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2*stencil[d_advecting];
        const ssize_t sp3_ed = 3*stencil[d_advecting];
        const ssize_t sp4_ed = 4*stencil[d_advecting];

        const ssize_t sm2_ing = -2*stencil[d_advected];
        const ssize_t sm1_ing = -stencil[d_advected];
        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2*stencil[d_advected];
        const ssize_t sp3_ing = 3*stencil[d_advected];

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;

                        //Upwind for positive velocity
                        const double phip = interp_weno7(vel_advected[ijk + sm3_ed],
                                                         vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp3_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno7(vel_advected[ijk + sp4_ed],
                                                         vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk + sm2_ed]);

                        const double vel_adv = interp_6(vel_advecting[ijk + sm2_ing],
                                                        vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing],
                                                        vel_advecting[ijk + sp3_ing]);

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
                        const double phip = interp_weno7(vel_advected[ijk + sm3_ed],
                                                         vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp3_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno7(vel_advected[ijk + sp4_ed],
                                                         vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk + sm2_ed]);

                        const double vel_adv = interp_6(vel_advecting[ijk + sm2_ing],
                                                        vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing],
                                                        vel_advecting[ijk + sp3_ing]);

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
                        const double phip = interp_weno7(vel_advected[ijk + sm3_ed],
                                                         vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp3_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno7(vel_advected[ijk + sp4_ed],
                                                         vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk + sm2_ed]);

                        const double vel_adv = interp_6(vel_advecting[ijk + sm2_ing],
                                                        vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing],
                                                        vel_advecting[ijk + sp3_ing]);

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


void weno_ninth_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        const ssize_t imin = 4;
        const ssize_t jmin = 4;
        const ssize_t kmin = 4;

        const ssize_t imax = dims->nlg[0]-5;
        const ssize_t jmax = dims->nlg[1]-5;
        const ssize_t kmax = dims->nlg[2]-5;

        const ssize_t stencil[3] = {istride,jstride,1};

        const ssize_t sm4_ed = -4*stencil[d_advecting];
        const ssize_t sm3_ed = -3*stencil[d_advecting];
        const ssize_t sm2_ed = -2*stencil[d_advecting];
        const ssize_t sm1_ed = -stencil[d_advecting];
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2*stencil[d_advecting];
        const ssize_t sp3_ed = 3*stencil[d_advecting];
        const ssize_t sp4_ed = 4*stencil[d_advecting];
        const ssize_t sp5_ed = 5*stencil[d_advecting];

        const ssize_t sm3_ing = -3*stencil[d_advected];
        const ssize_t sm2_ing = -2*stencil[d_advected];
        const ssize_t sm1_ing = -stencil[d_advected];
        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2*stencil[d_advected];
        const ssize_t sp3_ing = 3*stencil[d_advected];
        const ssize_t sp4_ing = 4*stencil[d_advected];

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;

                        //Upwind for positive velocity
                        const double phip = interp_weno9(vel_advected[ijk + sm4_ed],
                                                        vel_advected[ijk + sm3_ed],
                                                        vel_advected[ijk + sm2_ed],
                                                        vel_advected[ijk + sm1_ed],
                                                        vel_advected[ijk],
                                                        vel_advected[ijk + sp1_ed],
                                                        vel_advected[ijk + sp2_ed],
                                                        vel_advected[ijk + sp3_ed],
                                                        vel_advected[ijk + sp4_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno9(vel_advected[ijk + sp5_ed],
                                                         vel_advected[ijk + sp4_ed],
                                                         vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm3_ed]);

                        const double vel_adv = interp_8(vel_advecting[ijk + sm3_ing],
                                                        vel_advecting[ijk + sm2_ing],
                                                        vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing],
                                                        vel_advecting[ijk + sp3_ing],
                                                        vel_advecting[ijk + sp4_ing]);

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
                        const double phip = interp_weno9(vel_advected[ijk + sm4_ed],
                                                        vel_advected[ijk + sm3_ed],
                                                        vel_advected[ijk + sm2_ed],
                                                        vel_advected[ijk + sm1_ed],
                                                        vel_advected[ijk],
                                                        vel_advected[ijk + sp1_ed],
                                                        vel_advected[ijk + sp2_ed],
                                                        vel_advected[ijk + sp3_ed],
                                                        vel_advected[ijk + sp4_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno9(vel_advected[ijk + sp5_ed],
                                                         vel_advected[ijk + sp4_ed],
                                                         vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm3_ed]);

                        const double vel_adv = interp_8(vel_advecting[ijk+sm3_ing],
                                                        vel_advecting[ijk+sm2_ing],
                                                        vel_advecting[ijk+sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing],
                                                        vel_advecting[ijk + sp3_ing],
                                                        vel_advecting[ijk + sp4_ing]);

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
                        const double phip = interp_weno9(vel_advected[ijk + sm4_ed],
                                                        vel_advected[ijk + sm3_ed],
                                                        vel_advected[ijk + sm2_ed],
                                                        vel_advected[ijk + sm1_ed],
                                                        vel_advected[ijk],
                                                        vel_advected[ijk + sp1_ed],
                                                        vel_advected[ijk + sp2_ed],
                                                        vel_advected[ijk + sp3_ed],
                                                        vel_advected[ijk + sp4_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno9(vel_advected[ijk + sp5_ed],
                                                         vel_advected[ijk + sp4_ed],
                                                         vel_advected[ijk + sp3_ed],
                                                         vel_advected[ijk + sp2_ed],
                                                         vel_advected[ijk + sp1_ed],
                                                         vel_advected[ijk],
                                                         vel_advected[ijk + sm1_ed],
                                                         vel_advected[ijk + sm2_ed],
                                                         vel_advected[ijk + sm3_ed]);

                        const double vel_adv = interp_8(vel_advecting[ijk + sm3_ing],
                                                        vel_advecting[ijk + sm2_ing],
                                                        vel_advecting[ijk + sm1_ing],
                                                        vel_advecting[ijk],
                                                        vel_advecting[ijk + sp1_ing],
                                                        vel_advecting[ijk + sp2_ing],
                                                        vel_advecting[ijk + sp3_ing],
                                                        vel_advecting[ijk + sp4_ing]);

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

void weno_eleventh_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t jstride = dims->nlg[2];

        const ssize_t imin = 5;
        const ssize_t jmin = 5;
        const ssize_t kmin = 5;

        const ssize_t imax = dims->nlg[0]-6;
        const ssize_t jmax = dims->nlg[1]-6;
        const ssize_t kmax = dims->nlg[2]-6;

        const ssize_t stencil[3] = {istride,jstride,1};

        const ssize_t sm5_ed = -5*stencil[d_advecting];
        const ssize_t sm4_ed = -4*stencil[d_advecting];
        const ssize_t sm3_ed = -3*stencil[d_advecting];
        const ssize_t sm2_ed = -2*stencil[d_advecting];
        const ssize_t sm1_ed = -stencil[d_advecting];
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp2_ed = 2*stencil[d_advecting];
        const ssize_t sp3_ed = 3*stencil[d_advecting];
        const ssize_t sp4_ed = 4*stencil[d_advecting];
        const ssize_t sp5_ed = 5*stencil[d_advecting];
        const ssize_t sp6_ed = 6*stencil[d_advecting];

        const ssize_t sm4_ing = -4*stencil[d_advected];
        const ssize_t sm3_ing = -3*stencil[d_advected];
        const ssize_t sm2_ing = -2*stencil[d_advected];
        const ssize_t sm1_ing = -stencil[d_advected];
        const ssize_t sp1_ing = stencil[d_advected];
        const ssize_t sp2_ing = 2*stencil[d_advected];
        const ssize_t sp3_ing = 3*stencil[d_advected];
        const ssize_t sp4_ing = 4*stencil[d_advected];
        const ssize_t sp5_ing = 5*stencil[d_advected];

        if (d_advected != 2 && d_advecting !=2){
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;

                        //Upwind for positive velocity
                        const double phip = interp_weno11(vel_advected[ijk + sm5_ed],
                                                          vel_advected[ijk + sm4_ed],
                                                          vel_advected[ijk + sm3_ed],
                                                          vel_advected[ijk + sm2_ed],
                                                          vel_advected[ijk + sm1_ed],
                                                          vel_advected[ijk],
                                                          vel_advected[ijk + sp1_ed],
                                                          vel_advected[ijk + sp2_ed],
                                                          vel_advected[ijk + sp3_ed],
                                                          vel_advected[ijk + sp4_ed],
                                                          vel_advected[ijk + sp5_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno11(vel_advected[ijk + sp6_ed],
                                                          vel_advected[ijk + sp5_ed],
                                                          vel_advected[ijk + sp4_ed],
                                                          vel_advected[ijk + sp3_ed],
                                                          vel_advected[ijk + sp2_ed],
                                                          vel_advected[ijk + sp1_ed],
                                                          vel_advected[ijk],
                                                          vel_advected[ijk + sm1_ed],
                                                          vel_advected[ijk + sm2_ed],
                                                          vel_advected[ijk + sm3_ed],
                                                          vel_advected[ijk + sm4_ed]);

                        const double vel_adv = interp_10(vel_advecting[ijk + sm4_ing],
                                                         vel_advecting[ijk + sm3_ing],
                                                         vel_advecting[ijk + sm2_ing],
                                                         vel_advecting[ijk + sm1_ing],
                                                         vel_advecting[ijk],
                                                         vel_advecting[ijk + sp1_ing],
                                                         vel_advecting[ijk + sp2_ing],
                                                         vel_advecting[ijk + sp3_ing],
                                                         vel_advecting[ijk + sp4_ing],
                                                         vel_advecting[ijk + sp5_ing]);

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
                        const double phip = interp_weno11(vel_advected[ijk + sm5_ed],
                                                          vel_advected[ijk + sm4_ed],
                                                          vel_advected[ijk + sm3_ed],
                                                          vel_advected[ijk + sm2_ed],
                                                          vel_advected[ijk + sm1_ed],
                                                          vel_advected[ijk],
                                                          vel_advected[ijk + sp1_ed],
                                                          vel_advected[ijk + sp2_ed],
                                                          vel_advected[ijk + sp3_ed],
                                                          vel_advected[ijk + sp4_ed],
                                                          vel_advected[ijk + sp5_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno11(vel_advected[ijk + sp6_ed],
                                                          vel_advected[ijk + sp5_ed],
                                                          vel_advected[ijk + sp4_ed],
                                                          vel_advected[ijk + sp3_ed],
                                                          vel_advected[ijk + sp2_ed],
                                                          vel_advected[ijk + sp1_ed],
                                                          vel_advected[ijk],
                                                          vel_advected[ijk + sm1_ed],
                                                          vel_advected[ijk + sm2_ed],
                                                          vel_advected[ijk + sm3_ed],
                                                          vel_advected[ijk + sm4_ed]);

                        const double vel_adv = interp_10(vel_advecting[ijk + sm4_ing],
                                                         vel_advecting[ijk + sm3_ing],
                                                         vel_advecting[ijk + sm2_ing],
                                                         vel_advecting[ijk + sm1_ing],
                                                         vel_advecting[ijk],
                                                         vel_advecting[ijk + sp1_ing],
                                                         vel_advecting[ijk + sp2_ing],
                                                         vel_advecting[ijk + sp3_ing],
                                                         vel_advecting[ijk + sp4_ing],
                                                         vel_advecting[ijk + sp5_ing]);

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
                        const double phip = interp_weno11(vel_advected[ijk + sm5_ed],
                                                          vel_advected[ijk + sm4_ed],
                                                          vel_advected[ijk + sm3_ed],
                                                          vel_advected[ijk + sm2_ed],
                                                          vel_advected[ijk + sm1_ed],
                                                          vel_advected[ijk],
                                                          vel_advected[ijk + sp1_ed],
                                                          vel_advected[ijk + sp2_ed],
                                                          vel_advected[ijk + sp3_ed],
                                                          vel_advected[ijk + sp4_ed],
                                                          vel_advected[ijk + sp5_ed]);

                        // Upwind for negative velocity
                        const double phim = interp_weno11(vel_advected[ijk + sp6_ed],
                                                          vel_advected[ijk + sp5_ed],
                                                          vel_advected[ijk + sp4_ed],
                                                          vel_advected[ijk + sp3_ed],
                                                          vel_advected[ijk + sp2_ed],
                                                          vel_advected[ijk + sp1_ed],
                                                          vel_advected[ijk],
                                                          vel_advected[ijk + sm1_ed],
                                                          vel_advected[ijk + sm2_ed],
                                                          vel_advected[ijk + sm3_ed],
                                                          vel_advected[ijk + sm4_ed]);

                        const double vel_adv = interp_10(vel_advecting[ijk + sm4_ing],
                                                         vel_advecting[ijk + sm3_ing],
                                                         vel_advecting[ijk + sm2_ing],
                                                         vel_advecting[ijk + sm1_ing],
                                                         vel_advecting[ijk],
                                                         vel_advecting[ijk + sp1_ing],
                                                         vel_advecting[ijk + sp2_ing],
                                                         vel_advecting[ijk + sp3_ing],
                                                         vel_advecting[ijk + sp4_ing],
                                                         vel_advecting[ijk + sp5_ing]);
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