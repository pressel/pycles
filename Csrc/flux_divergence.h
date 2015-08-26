#pragma once
#include "grid.h"

void scalar_flux_divergence(struct DimStruct *dims, double *alpha0, double *alpha0_half, double *flux, double *tendency,
    double dx, size_t d){

    const size_t imin = dims->gw;
    const size_t jmin = dims->gw;
    const size_t kmin = dims->gw;

    const size_t imax = dims->nlg[0] - dims->gw;
    const size_t jmax = dims->nlg[1] - dims->gw;
    const size_t kmax = dims->nlg[2] - dims->gw;

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];
    const double dxi = 1.0/dx;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sm1 = -stencil[d];

    for(size_t i=imin; i<imax; i++){
        const size_t ishift = i * istride;
        for(size_t j=jmin; j<jmax; j++){
            const size_t jshift = j * jstride;
            for(size_t k=kmin; k<kmax; k++){
                const size_t ijk = ishift + jshift + k;
                tendency[ijk] -= alpha0_half[k] * (flux[ijk] - flux[ijk + sm1])*dxi;
            } // End k loop
        } // End j loop
    } // End i loop
}

void momentum_flux_divergence(struct DimStruct *dims, double *alpha0, double *alpha0_half, double *flux,
                                double *tendency, size_t d_advected, size_t d_advecting){

    const size_t imin = dims->gw;
    const size_t jmin = dims->gw;
    const size_t kmin = dims->gw;

    const size_t imax = dims->nlg[0] - dims->gw;
    const size_t jmax = dims->nlg[1] - dims->gw;
    const size_t kmax = dims->nlg[2] - dims->gw;

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];
    const double dxi = 1.0/dims->dx[d_advecting];

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sm1 = -stencil[d_advecting];

    if(d_advected != 2){
        for(size_t i=imin; i<imax; i++){
            const size_t ishift = i * istride;
            for(size_t j=jmin; j<jmax; j++){
                const size_t jshift = j * jstride;
                for(size_t k=kmin; k<kmax; k++){
                    const size_t ijk = ishift + jshift + k;
                    tendency[ijk] -= alpha0_half[k] * (flux[ijk] - flux[ijk + sm1])*dxi;
                } // End k loop
            } // End j loop
        } // End i loop

    } // End if
    else{
        for(size_t i=imin; i<imax; i++){
            const size_t ishift = i * istride;
            for(size_t j=jmin; j<jmax; j++){
                const size_t jshift = j * jstride;
                for(size_t k=kmin; k<kmax; k++){
                    const size_t ijk = ishift + jshift + k;
                    tendency[ijk] -= alpha0[k] * (flux[ijk] - flux[ijk + sm1])*dxi;
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else



    }