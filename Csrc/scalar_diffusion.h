#pragma once
#include "grid.h"
#include "advection_interpolation.h"

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
                }
            }
        }
    }
    else{
        for(size_t i=imin; i<imax; i++){
            const size_t ishift = i * istride;
            for(size_t j=jmin; j<jmax; j++){
                const size_t jshift = j * jstride;
                for(size_t k=kmin; k<kmax; k++){
                    const size_t ijk = ishift + jshift + k;
                    flux[ijk] = -interp_2(diffusivity[ijk],diffusivity[ijk+stencil[d]])*(scalar[ijk+stencil[d]]-scalar[ijk])*rho0_half[k]*dxi;
                }
            }
        }
    }

    return;
}