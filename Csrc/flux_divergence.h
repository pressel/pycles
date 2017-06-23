#pragma once
#include "grid.h"

void scalar_flux_divergence(struct DimStruct *dims, double *alpha0, double *alpha0_half, double *flux, double *tendency,
    double dx, ssize_t d){

    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;

    const ssize_t imax = dims->nlg[0] - dims->gw;
    const ssize_t jmax = dims->nlg[1] - dims->gw;
    const ssize_t kmax = dims->nlg[2] - dims->gw;

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const double dxi = dims->dxi[d];

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sm1 = -stencil[d];

    const double * imetl_half = dims-> imetl_half;

    if(d == 2){
        for(ssize_t i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;
                    tendency[ijk] -= alpha0_half[k] * (flux[ijk] - flux[ijk + sm1])*dxi * imetl_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
     }
     else{
        for(ssize_t i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;
                    tendency[ijk] -= alpha0_half[k] * (flux[ijk] - flux[ijk + sm1])*dxi ;
                } // End k loop
            } // End j loop
        } // End i loop
     }
}

void momentum_flux_divergence(struct DimStruct *dims, double *alpha0, double *alpha0_half, double *flux,
                                double *tendency, ssize_t d_advected, ssize_t d_advecting){

    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;

    const ssize_t imax = dims->nlg[0] - dims->gw;
    const ssize_t jmax = dims->nlg[1] - dims->gw;
    const ssize_t kmax = dims->nlg[2] - dims->gw;

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const double dxi = dims->dxi[d_advecting];

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sm1 = -stencil[d_advecting];

    const double * imetl = dims -> imetl;
    const double * imetl_half = dims -> imetl_half;

    if(d_advecting == 2){
        if(d_advected != 2){
            for(ssize_t i=imin; i<imax; i++){
                const ssize_t ishift = i * istride;
                for(ssize_t j=jmin; j<jmax; j++){
                    const ssize_t jshift = j * jstride;
                    for(ssize_t k=kmin; k<kmax; k++){
                        const ssize_t ijk = ishift + jshift + k;
                        tendency[ijk] -= alpha0_half[k] * (flux[ijk] - flux[ijk + sm1])*dxi * imetl_half[k];
                    } // End k loop
                } // End j loop
            } // End i loop

        } // End if
        else{
            for(ssize_t i=imin; i<imax; i++){
                const ssize_t ishift = i * istride;
                for(ssize_t j=jmin; j<jmax; j++){
                    const ssize_t jshift = j * jstride;
                    for(ssize_t k=kmin; k<kmax; k++){
                        const ssize_t ijk = ishift + jshift + k;
                        tendency[ijk] -= alpha0[k] * (flux[ijk] - flux[ijk + sm1])*dxi * imetl[k];
                    } // End k loop
                } // End j loop
            } // End i loop
        } // End else

        }
    else{
        if(d_advected != 2){
            for(ssize_t i=imin; i<imax; i++){
                const ssize_t ishift = i * istride;
                for(ssize_t j=jmin; j<jmax; j++){
                    const ssize_t jshift = j * jstride;
                    for(ssize_t k=kmin; k<kmax; k++){
                        const ssize_t ijk = ishift + jshift + k;
                        tendency[ijk] -= alpha0_half[k] * (flux[ijk] - flux[ijk + sm1])*dxi;
                    } // End k loop
                } // End j loop
            } // End i loop
        } // End if
        else{
            for(ssize_t i=imin; i<imax; i++){
                const ssize_t ishift = i * istride;
                for(ssize_t j=jmin; j<jmax; j++){
                    const ssize_t jshift = j * jstride;
                    for(ssize_t k=kmin; k<kmax; k++){
                        const ssize_t ijk = ishift + jshift + k;
                        tendency[ijk] -= alpha0[k] * (flux[ijk] - flux[ijk + sm1])*dxi;
                    } // End k loop
                } // End j loop
            } // End i loop
        } // End else

        }

    }
