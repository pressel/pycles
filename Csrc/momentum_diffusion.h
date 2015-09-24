#pragma once
#include "grid.h"
#include <stdio.h>

void compute_diffusive_flux_m(const struct DimStruct *dims, double* restrict strain_rate,  double* restrict viscosity , double* restrict flux, double* restrict rho0, double* restrict rho0_half, size_t i1, size_t i2){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = dims->gw-1;
    const size_t jmin = dims->gw-1;
    size_t kmin = dims->gw-1;

    const size_t imax = dims->nlg[0]-dims->gw;
    const size_t jmax = dims->nlg[1]-dims->gw;
    const size_t kmax = dims->nlg[2]-dims->gw;

    const size_t stencil[3] = {istride,jstride,1};



    if(i1 != 2 && i2 != 2){
        if(i1==i2){
            for(size_t i=imin; i<imax; i++){
                const size_t ishift = i * istride;
                for(size_t j=jmin; j<jmax; j++){
                    const size_t jshift = j * jstride;
                    for(size_t k=kmin; k<kmax; k++){
                        const size_t ijk = ishift + jshift + k;
                        flux[ijk] = -2.0 * strain_rate[ijk] * viscosity[ijk + stencil[i1]] * rho0_half[k];
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
                        const double visc_interp = 0.25 * (viscosity[ijk] + viscosity[ijk + stencil[i1]]
                        + viscosity[ijk + stencil[i2]] +  viscosity[ijk + stencil[i1] + stencil[i2]] );
                        flux[ijk] = -2.0 * strain_rate[ijk] * visc_interp * rho0_half[k];
                    }
                }
            }

        }
    }
    else if(i1==2 && i2==2){
        for(size_t i=imin; i<imax; i++){
            const size_t ishift = i * istride;
            for(size_t j=jmin; j<jmax; j++){
                const size_t jshift = j * jstride;
                for(size_t k=kmin; k<kmax; k++){
                    const size_t ijk = ishift + jshift + k;
                    flux[ijk] = -2.0 * strain_rate[ijk] * viscosity[ijk + stencil[i1]] * rho0_half[k+1];
                }
            }
        }
    }
    // u and v in z or w in x and y
    else{
        for(size_t i=imin; i<imax; i++){
            const size_t ishift = i * istride;
            for(size_t j=jmin; j<jmax; j++){
                const size_t jshift = j * jstride;
                for(size_t k=kmin; k<kmax; k++){
                    const size_t ijk = ishift + jshift + k;
                    const double visc_interp = 0.25 * (viscosity[ijk] + viscosity[ijk + stencil[i1]]
                        + viscosity[ijk + stencil[i2]] +  viscosity[ijk + stencil[i1] + stencil[i2]] );
                        flux[ijk] = -2.0 * strain_rate[ijk] * visc_interp * rho0[k];
                }
            }
        }

    }

    // If this is the surface set the flux to be exactly zero (This may not be necessary)
    if(dims->indx_lo[2] == 0){
        for(size_t i=imin; i<imax; i++){
            const size_t ishift = i * istride;
            for(size_t j=jmin; j<jmax; j++){
                const size_t jshift = j * jstride;
                const size_t ijk = ishift + jshift + dims->gw;
                flux[ijk] = 0.0;

            }
         }
    }


    return;
}

void compute_entropy_source(const struct DimStruct *dims, double* restrict viscosity, double* restrict strain_rate_mag, double* restrict temperature, double* restrict entropy_tendency){
    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = dims->gw;
    const size_t jmin = dims->gw;
    const size_t kmin = dims->gw;

    const size_t imax = dims->nlg[0]-dims->gw;
    const size_t jmax = dims->nlg[1]-dims->gw;
    const size_t kmax = dims->nlg[2]-dims->gw;

    for(size_t i=imin;i<imax;i++){
        const size_t ishift = i*istride ;
        for(size_t j=jmin;j<jmax;j++){
            const size_t jshift = j*jstride;
            for(size_t k=kmin;k<kmax;k++){
                const size_t ijk = ishift + jshift + k ;
                entropy_tendency[ijk] +=  strain_rate_mag[ijk] * strain_rate_mag[ijk] * viscosity[ijk] / temperature[ijk];
            }
        }
    }
}