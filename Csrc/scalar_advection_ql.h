#pragma once
#include "grid.h"
#include "advection_interpolation.h"

void second_order_a_ql(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

    const ssize_t imax = dims->nlg[0]-1;
    const ssize_t jmax = dims->nlg[1]-1;
    const ssize_t kmax = dims->nlg[2]-1;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1]) * velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else

    return;
}



void sixth_order_a_ql(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 2;
    const ssize_t jmin = 2;
    const ssize_t kmin = 2;

    const ssize_t imax = dims->nlg[0]-3;
    const ssize_t jmax = dims->nlg[1]-3;
    const ssize_t kmax = dims->nlg[2]-3;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_6(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3])*velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_6(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}



void compute_eddy_fluxes_a(struct DimStruct *dims, double* restrict rho0, double* rho0_half ,double* restrict velocity, double* restrict scalar,
                                double* restrict flux, int d, int scheme){
    switch(scheme){
        case 1:
            upwind_first_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 2:
            second_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 3:
            weno_third_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 4:
            fourth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 5:
            weno_fifth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 6:
            sixth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 7:
            weno_seventh_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 8:
            eighth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 9:
            weno_ninth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 11:
            weno_eleventh_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        default:
            // Make WENO5 default case. The central schemes may not be necessarily stable, however WENO5 should be.
            weno_fifth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
    };
};

