#pragma once
#include "grid.h"
#include "advection_interpolation.h"


void second_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 0;
        const long jmin = 0;
        const long kmin = 0;

        const long imax = dims->nlg[0]-1;
        const long jmax = dims->nlg[1]-1;
        const long kmax = dims->nlg[2]-1;

        const long stencil[3] = {istride,jstride,1};
        const long sp1_ed = stencil[d_advecting];
        const long sp1_ing = stencil[d_advected];

        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing])
                            *interp_2(vel_advected[ijk],vel_advected[ijk + sp1_ed]) )*rho0_half[k];
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing])
                            *interp_2(vel_advected[ijk],vel_advected[ijk + sp1_ed]) )*rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        flux[ijk] = (interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing])
                            *interp_2(vel_advected[ijk],vel_advected[ijk + sp1_ed]) )*rho0[k];
                    }
                }
            }
        }
        return;
    }

void fourth_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 1;
        const long jmin = 1;
        const long kmin = 1;

        const long imax = dims->nlg[0]-2;
        const long jmax = dims->nlg[1]-2;
        const long kmax = dims->nlg[2]-2;

        const long stencil[3] = {istride,jstride,1};
        const long sp1_ed = stencil[d_advecting];
        const long sp2_ed = 2 * sp1_ed ;
        const long sm1_ed = -sp1_ed ;

        const long sp1_ing = stencil[d_advected];
        const long sp2_ing = 2 * sp1_ing;
        const long sm1_ing = -sp1_ing;

        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_4(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0_half[k];
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_4(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed])) * rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_4(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing]) *
                                 interp_4(vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed]))* rho0[k];
                    }
                }
            }
        }
        return;
    }

void sixth_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 2;
        const long jmin = 2;
        const long kmin = 2;

        const long imax = dims->nlg[0]-3;
        const long jmax = dims->nlg[1]-3;
        const long kmax = dims->nlg[2]-3;

        const long stencil[3] = {istride,jstride,1};
        const long sp1_ed = stencil[d_advecting];
        const long sp2_ed = 2 * sp1_ed ;
        const long sp3_ed = 3 * sp1_ed ;
        const long sm1_ed = -sp1_ed ;
        const long sm2_ed = -2*sp1_ed;

        const long sp1_ing = stencil[d_advected];
        const long sp2_ing = 2 * sp1_ing;
        const long sp3_ing = 3 * sp1_ing;
        const long sm1_ing = -sp1_ing;
        const long sm2_ing = -2*sp1_ing;

        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed])) * rho0_half[k];
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]))* rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing]) *
                                 interp_6(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]))* rho0[k];
                    }
                }
            }
        }
        return;
    }

void eighth_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 3;
        const long jmin = 3;
        const long kmin = 3;

        const long imax = dims->nlg[0]-4;
        const long jmax = dims->nlg[1]-4;
        const long kmax = dims->nlg[2]-4;

        const long stencil[3] = {istride,jstride,1};
        const long sp1_ed = stencil[d_advecting];
        const long sp2_ed = 2 * sp1_ed ;
        const long sp3_ed = 3 * sp1_ed ;
        const long sp4_ed = 4 * sp1_ed ;
        const long sm1_ed = -sp1_ed ;
        const long sm2_ed = -2*sp1_ed;
        const long sm3_ed = -3*sp1_ed;

        const long sp1_ing = stencil[d_advected];
        const long sp2_ing = 2 * sp1_ing;
        const long sp3_ing = 3 * sp1_ing;
        const long sp4_ing = 4 * sp1_ing;
        const long sm1_ing = -sp1_ing;
        const long sm2_ing = -2*sp1_ing;
        const long sm3_ing = -3*sp1_ing;

        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_8(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing],vel_advecting[ijk+sp4_ing]) *
                                 interp_8(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed])) * rho0_half[k];
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_8(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing],vel_advecting[ijk+sp4_ing]) *
                                 interp_8(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed])) * rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                    flux[ijk] = (interp_8(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk+sp1_ing],vel_advecting[ijk+sp2_ing],vel_advecting[ijk+sp3_ing],vel_advecting[ijk+sp4_ing]) *
                                 interp_8(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed])) * rho0[k];
                    }
                }
            }
        }
        return;
    }

void weno_third_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 1;
        const long jmin = 1;
        const long kmin = 1;

        const long imax = dims->nlg[0]-2;
        const long jmax = dims->nlg[1]-2;
        const long kmax = dims->nlg[2]-2;

        const long stencil[3] = {istride,jstride,1};
        const long sm1_ed = -stencil[d_advecting];
        const long sp1_ed = stencil[d_advecting];
        const long sp2_ed = 2*stencil[d_advecting];
        const long sp1_ing = stencil[d_advected];


        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno3(vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno3(vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk]);
                        const double vel_adv = interp_2(vel_advecting[ijk],vel_advecting[ijk + sp1_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k] ;
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno3(vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed]);
                        // Up wind for negative velocity
                        const double phim = interp_weno3(vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk]);
                        const double vel_adv = interp_2(vel_advecting[ijk],vel_advecting[ijk + sp1_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno3(vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed]);
                        // Up wind for negative velocity
                        const double phim = interp_weno3(vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk]);
                        const double vel_adv = interp_2(vel_advecting[ijk],vel_advecting[ijk + sp1_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0[k];
                    }
                }
            }
        }
        return;
    }

void weno_fifth_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 2;
        const long jmin = 2;
        const long kmin = 2;

        const long imax = dims->nlg[0]-3;
        const long jmax = dims->nlg[1]-3;
        const long kmax = dims->nlg[2]-3;

        const long stencil[3] = {istride,jstride,1};

        const long sp1_ed = stencil[d_advecting];
        const long sp2_ed = 2*sp1_ed ;
        const long sp3_ed = 3*sp1_ed ;
        const long sm1_ed = -sp1_ed ;
        const long sm2_ed = -2*sp1_ed ;

        const long sp1_ing = stencil[d_advected];
        const long sp2_ing = 2*sp1_ing ;
        const long sm1_ing = -sp1_ing ;


        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed]);
                        const double vel_adv = interp_4(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k] ;
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed]);
                        const double vel_adv = interp_4(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing]);
                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno5(vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno5(vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed]);
                        const double vel_adv = interp_4(vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing]);
                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0[k];
                    }
                }
            }
        }
        return;
    }

void weno_seventh_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 3;
        const long jmin = 3;
        const long kmin = 3;

        const long imax = dims->nlg[0]-4;
        const long jmax = dims->nlg[1]-4;
        const long kmax = dims->nlg[2]-4;

        const long stencil[3] = {istride,jstride,1};

        const long sm3_ed = -3*stencil[d_advecting];
        const long sm2_ed = -2*stencil[d_advecting];
        const long sm1_ed = -stencil[d_advecting];
        const long sp1_ed = stencil[d_advecting];
        const long sp2_ed = 2*stencil[d_advecting];
        const long sp3_ed = 3*stencil[d_advecting];
        const long sp4_ed = 4*stencil[d_advecting];

        const long sm2_ing = -2*stencil[d_advected];
        const long sm1_ing = -stencil[d_advected];
        const long sp1_ing = stencil[d_advected];
        const long sp2_ing = 2*stencil[d_advected];
        const long sp3_ing = 3*stencil[d_advected];

        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno7(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno7(vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed]);
                        const double vel_adv = interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k] ;
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno7(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno7(vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed]);
                        const double vel_adv = interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing]);
                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno7(vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno7(vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed]);
                        const double vel_adv = interp_6(vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing]);
                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0[k];
                    }
                }
            }
        }
        return;
    }


void weno_ninth_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 4;
        const long jmin = 4;
        const long kmin = 4;

        const long imax = dims->nlg[0]-5;
        const long jmax = dims->nlg[1]-5;
        const long kmax = dims->nlg[2]-5;

        const long stencil[3] = {istride,jstride,1};

        const long sm4_ed = -4*stencil[d_advecting];
        const long sm3_ed = -3*stencil[d_advecting];
        const long sm2_ed = -2*stencil[d_advecting];
        const long sm1_ed = -stencil[d_advecting];
        const long sp1_ed = stencil[d_advecting];
        const long sp2_ed = 2*stencil[d_advecting];
        const long sp3_ed = 3*stencil[d_advecting];
        const long sp4_ed = 4*stencil[d_advecting];
        const long sp5_ed = 5*stencil[d_advecting];

        const long sm3_ing = -3*stencil[d_advected];
        const long sm2_ing = -2*stencil[d_advected];
        const long sm1_ing = -stencil[d_advected];
        const long sp1_ing = stencil[d_advected];
        const long sp2_ing = 2*stencil[d_advected];
        const long sp3_ing = 3*stencil[d_advected];
        const long sp4_ing = 4*stencil[d_advected];

        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno9(vel_advected[ijk+sm4_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno9(vel_advected[ijk+sp5_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm3_ed]);
                        const double vel_adv = interp_8(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing],vel_advecting[ijk + sp4_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k] ;


                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno9(vel_advected[ijk+sm4_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno9(vel_advected[ijk+sp5_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm3_ed]);
                        const double vel_adv = interp_8(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing],vel_advecting[ijk + sp4_ing]);
                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k+1];

                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno9(vel_advected[ijk+sm4_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno9(vel_advected[ijk+sp5_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm3_ed]);
                        const double vel_adv = interp_8(vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing],vel_advecting[ijk + sp4_ing]);
                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0[k];

                    }
                }
            }
        }
        return;
    }

void weno_eleventh_order_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict flux, long d_advected, long d_advecting){

        const long istride = dims->nlg[1] * dims->nlg[2];
        const long jstride = dims->nlg[2];

        const long imin = 5;
        const long jmin = 5;
        const long kmin = 5;

        const long imax = dims->nlg[0]-6;
        const long jmax = dims->nlg[1]-6;
        const long kmax = dims->nlg[2]-6;

        const long stencil[3] = {istride,jstride,1};

        const long sm5_ed = -5*stencil[d_advecting];
        const long sm4_ed = -4*stencil[d_advecting];
        const long sm3_ed = -3*stencil[d_advecting];
        const long sm2_ed = -2*stencil[d_advecting];
        const long sm1_ed = -stencil[d_advecting];
        const long sp1_ed = stencil[d_advecting];
        const long sp2_ed = 2*stencil[d_advecting];
        const long sp3_ed = 3*stencil[d_advecting];
        const long sp4_ed = 4*stencil[d_advecting];
        const long sp5_ed = 5*stencil[d_advecting];
        const long sp6_ed = 6*stencil[d_advecting];

        const long sm4_ing = -4*stencil[d_advected];
        const long sm3_ing = -3*stencil[d_advected];
        const long sm2_ing = -2*stencil[d_advected];
        const long sm1_ing = -stencil[d_advected];
        const long sp1_ing = stencil[d_advected];
        const long sp2_ing = 2*stencil[d_advected];
        const long sp3_ing = 3*stencil[d_advected];
        const long sp4_ing = 4*stencil[d_advected];
        const long sp5_ing = 5*stencil[d_advected];

        if (d_advected != 2 && d_advecting !=2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno11(vel_advected[ijk+sm5_ed],vel_advected[ijk+sm4_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp5_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno11(vel_advected[ijk+sp6_ed],vel_advected[ijk+sp5_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm4_ed]);
                        const double vel_adv = interp_10(vel_advecting[ijk+sm4_ing],vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing],vel_advecting[ijk + sp4_ing],vel_advecting[ijk + sp5_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k] ;
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno11(vel_advected[ijk+sm5_ed],vel_advected[ijk+sm4_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp5_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno11(vel_advected[ijk+sp6_ed],vel_advected[ijk+sp5_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm4_ed]);
                        const double vel_adv = interp_10(vel_advecting[ijk+sm4_ing],vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing],vel_advecting[ijk + sp4_ing],vel_advecting[ijk + sp5_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0_half[k+1];
                    }
                }
            }
        }
        else{
            for(long i=imin;i<imax;i++){
                const long ishift = i*istride;
                for(long j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                    for(long k=kmin;k<kmax;k++){
                        const long ijk = ishift + jshift + k;
                        //Upwind for positive velocity
                        const double phip = interp_weno11(vel_advected[ijk+sm5_ed],vel_advected[ijk+sm4_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm1_ed],vel_advected[ijk],
                                            vel_advected[ijk+sp1_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp5_ed]);
                        // Upwind for negative velocity
                        const double phim = interp_weno11(vel_advected[ijk+sp6_ed],vel_advected[ijk+sp5_ed],vel_advected[ijk+sp4_ed],vel_advected[ijk+sp3_ed],vel_advected[ijk+sp2_ed],vel_advected[ijk+sp1_ed],
                                            vel_advected[ijk],vel_advected[ijk+sm1_ed],vel_advected[ijk+sm2_ed],vel_advected[ijk+sm3_ed],vel_advected[ijk+sm4_ed]);
                        const double vel_adv = interp_10(vel_advecting[ijk+sm4_ing],vel_advecting[ijk+sm3_ing],vel_advecting[ijk+sm2_ing],vel_advecting[ijk+sm1_ing],vel_advecting[ijk],vel_advecting[ijk + sp1_ing],vel_advecting[ijk + sp2_ing],vel_advecting[ijk + sp3_ing],vel_advecting[ijk + sp4_ing],vel_advecting[ijk + sp5_ing]);

                        flux[ijk] = (0.5*(vel_adv+fabs(vel_adv))*phip + 0.5*(vel_adv-fabs(vel_adv))*phim)*rho0[k];
                    }
                }
            }
        }
        return;
    }


void compute_advective_fluxes_m(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict vel_advected, double* restrict vel_advecting,
                                double* restrict flux, long d_advected, long d_advecting, int scheme){

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
    };


};