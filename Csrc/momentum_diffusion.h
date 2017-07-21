#pragma once
#include "grid.h"
#include <stdio.h>

void compute_diffusive_flux_m(const struct DimStruct *dims, double* restrict strain_rate,  double* restrict viscosity , double* restrict flux, double* restrict rho0, double* restrict rho0_half, ssize_t i1, ssize_t i2){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw-1;
    const ssize_t jmin = dims->gw-1;
    const ssize_t kmin = dims->gw-1;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const ssize_t stencil[3] = {istride,jstride,1};

    const double * imetl = dims -> imetl;
    const double * imetl_half = dims -> imetl_half;
    
    ///Compute flux if not flux of $\tau_{3,3}$ or $\tau{3,2}$, or  $\tau{3,1}$,  $\tau{2,3}$, or  $\tau{1,3}$ 
    if(i1 != 2 && i2 != 2){
        /// Compute flux if $\tau_{1,1} or $\tau_{2,2}$
        if(i1==i2){
            for(ssize_t i=imin; i<imax; i++){
                const ssize_t ishift = i * istride;
                for(ssize_t j=jmin; j<jmax; j++){
                    const ssize_t jshift = j * jstride;
                    for(ssize_t k=kmin; k<kmax; k++){
                        const ssize_t ijk = ishift + jshift + k;
                        /// Viscosities are located at the flux location so no interpolation necessary 
                        flux[ijk] = -2.0 * strain_rate[ijk] * viscosity[ijk + stencil[i1]] * rho0_half[k];
                    }
                }
            }
        }
        ///Compute flux if $\tau_{2,1}$ or  $\tau{1,2}$
        else{
            for(ssize_t i=imin; i<imax; i++){
                const ssize_t ishift = i * istride;
                for(ssize_t j=jmin; j<jmax; j++){
                    const ssize_t jshift = j * jstride;
                    for(ssize_t k=kmin; k<kmax; k++){
                        const ssize_t ijk = ishift + jshift + k;
                        // Here the viscosity must be interpolated 
                        const double visc_interp = 0.25 * (viscosity[ijk] + viscosity[ijk + stencil[i1]]
                        + viscosity[ijk + stencil[i2]] +  viscosity[ijk + stencil[i1] + stencil[i2]] );
                        flux[ijk] = -2.0 * strain_rate[ijk] * visc_interp * rho0_half[k];
                    }
                }
            }
        }
    }
    /// Compute flux if $\tau_{3,3}$
    else if(i1==2 && i2==2){
        for(ssize_t i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;
                    // Viscosity again at flux location so no interpolation required 
                    flux[ijk] = -2.0 * strain_rate[ijk] * viscosity[ijk + stencil[i1]] * rho0_half[k+1];
                }
            }
        }
    }
    // Compute flux if $\tau_{3,1}$, $\tau_{3,2}$, $\tau_{1,3}$, $\tau_{2,3} 
    else{
        for(ssize_t i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin; k<kmax; k++){
                    const ssize_t ijk = ishift + jshift + k;
                    // Viscosity requires interpolation 
                    const double visc_interp = 0.25 * (viscosity[ijk] + viscosity[ijk + stencil[i1]]
                        + viscosity[ijk + stencil[i2]] +  viscosity[ijk + stencil[i1] + stencil[i2]] );
                        flux[ijk] = -2.0 * strain_rate[ijk] * visc_interp * rho0[k];
                }
            }
        }
    }

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

void compute_entropy_source(const struct DimStruct *dims, double* restrict viscosity, double* restrict strain_rate_mag, double* restrict temperature, double* restrict entropy_tendency){
    // Compute the entropy source corresponding to equation (55) in Pressel et al. 2015. 
    
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;
                entropy_tendency[ijk] +=  strain_rate_mag[ijk] * strain_rate_mag[ijk] * viscosity[ijk] / temperature[ijk];
            }
        }
    }
    return; 
}
