#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "thermodynamic_functions.h"
#include "entropies.h"
void second_order_diffusion(const struct DimStruct *dims, double *rho0, double *rho0_half, double *diffusivity, double *scalar, double *flux, double dx, ssize_t d, double factor){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = dims->gw-1;
    const ssize_t jmin = dims->gw-1;
    const ssize_t kmin = dims->gw-1;

    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    const ssize_t stencil[3] = {istride,jstride,1};
    const double dxi = 1.0/dx;

    const double * imetl = dims -> imetl;

    if (d == 2){
        for(ssize_t i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = -factor * interp_2(diffusivity[ijk],diffusivity[ijk+stencil[d]]) * (scalar[ijk+stencil[d]]-scalar[ijk])*rho0[k]*dxi * imetl[k];
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
                    flux[ijk] = -factor * interp_2(diffusivity[ijk],diffusivity[ijk+stencil[d]])*(scalar[ijk+stencil[d]]-scalar[ijk])*rho0_half[k]*dxi;
                } // End k loop
            }  // End j loop
        } // End i loop
    } // End else

    // If this is the surface set the flux to be exactly zero (This may not be necessary)
    if(dims->indx_lo[2] == 0){
        for(ssize_t i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                const ssize_t ijk = ishift + jshift + dims->gw - 1;
                flux[ijk] = 0.0;

            }
         }
    }


    return;
}

void compute_diffusive_flux(const struct DimStruct *dims, double *rho0, double *rho0_half, double *diffusivity, double *scalar, double *flux, double dx, ssize_t d, ssize_t scheme, double factor){

        switch(scheme){
            case 2:
                second_order_diffusion(dims, rho0, rho0_half, diffusivity, scalar, flux, dx, d, factor);
                break;
                };
}

void compute_qt_diffusion_s_source(const struct DimStruct *dims, double *p0_half, double *alpha0, double* alpha0_half, double *flux,
                                    double* qt, double* qv, double* T, double* tendency, double (*lam_fp)(double),
                                    double (*L_fp)(double, double), double dx, ssize_t d){

    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;

    const ssize_t imax = dims->nlg[0] - dims->gw;
    const ssize_t jmax = dims->nlg[1] - dims->gw;
    const ssize_t kmax = dims->nlg[2] - dims->gw;

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const double dxi = 1.0/dx;
    const ssize_t stencil[3] = {istride,jstride,1};

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                // Compute Dry air entropy specific entropy
                double pd = pd_c(p0_half[k],qt[ijk],qv[ijk]);
                double sd = sd_c(pd,T[ijk]);

                //Compute water vapor entropy specific entrop
                double pv = pv_c(p0_half[k],qt[ijk],qv[ijk]);
                double sv = sv_c(pv,T[ijk]);

                //Compute water entropy
                double lam = lam_fp(T[ijk]);
                double L = L_fp(T[ijk],lam);
                double sw = sv - (((qt[ijk] - qv[ijk])/qt[ijk])*L/T[ijk]);

                tendency[ijk] -= (sw - sd) * alpha0_half[k] * (flux[ijk + stencil[d]] - flux[ijk])*dxi;
            }  // End k loop
        } // End j loop
    } // End i loop
}
