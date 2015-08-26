#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "thermodynamic_functions.h"
#include "entropies.h"
void second_order_diffusion(const struct DimStruct *dims, double *rho0, double *rho0_half, double *diffusivity, double *scalar, double *flux, double dx, size_t d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = dims->gw-1;
    const size_t jmin = dims->gw-1;
    const size_t kmin = dims->gw-1;

    const size_t imax = dims->nlg[0]-dims->gw;
    const size_t jmax = dims->nlg[1]-dims->gw;
    const size_t kmax = dims->nlg[2]-dims->gw;

    const size_t stencil[3] = {istride,jstride,1};
    const double dxi = 1.0/dx;

    if (d == 2){
        for(size_t i=imin; i<imax; i++){
            const size_t ishift = i * istride;
            for(size_t j=jmin; j<jmax; j++){
                const size_t jshift = j * jstride;
                for(size_t k=kmin; k<kmax; k++){
                    const size_t ijk = ishift + jshift + k;
                    flux[ijk] = -interp_2(diffusivity[ijk],diffusivity[ijk+stencil[d]]) * (scalar[ijk+stencil[d]]-scalar[ijk])*rho0[k]*dxi;
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
                    flux[ijk] = -interp_2(diffusivity[ijk],diffusivity[ijk+stencil[d]])*(scalar[ijk+stencil[d]]-scalar[ijk])*rho0_half[k]*dxi;
                } // End k loop
            }  // End j loop
        } // End i loop
    } // End else

    return;
}

void compute_diffusive_flux(const struct DimStruct *dims, double *rho0, double *rho0_half, double *diffusivity, double *scalar, double *flux, double dx, size_t d, size_t scheme){

        switch(scheme){
            case 2:
                second_order_diffusion(dims, rho0, rho0_half, diffusivity, scalar, flux, dx, d);
                break;
                };
}

void compute_qt_diffusion_s_source(const struct DimStruct *dims, double *p0_half, double *alpha0, double* alpha0_half, double *flux,
                                    double* qt, double* qv, double* T, double* tendency, double (*lam_fp)(double),
                                    double (*L_fp)(double, double), double dx, size_t d){

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

    for(size_t i=imin; i<imax; i++){
        const size_t ishift = i * istride;
        for(size_t j=jmin; j<jmax; j++){
            const size_t jshift = j * jstride;
            for(size_t k=kmin; k<kmax; k++){
                const size_t ijk = ishift + jshift + k;

                // Compute Dry air entropy specific entropy
                double pd = pd_c(p0_half[k],qt[ijk],qv[ijk]);
                double sd = sd_c(pd,T[ijk]);

                //Compute water vapor entropy specific entrop
                double pv = pv_c(p0_half[k],qt[ijk],qv[ijk]);
                double sv = sv_c(pv,T[ijk]);

                //Compute water entropy
                double lam = lam_fp(T[ijk]);
                double L = L_fp(lam,T[ijk]);
                double sw = sv - (((qt[ijk] - sv)/qt[ijk])*L/T[ijk]);

                tendency[ijk] += (sw - sd) * alpha0_half[k] * (flux[ijk] - flux[ijk + stencil[d]])*dxi;
            }  // End k loop
        } // End j loop
    } // End i loop
}