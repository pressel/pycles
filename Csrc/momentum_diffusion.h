#pragma once
#include "grid.h"

void compute_diffusive_flux(const struct DimStruct *dims, double* restrict strain_rate,  double* restrict viscosity ,double* restrict flux, long i1, long i2){

    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0];
    const long jmax = dims->nlg[1];
    const long kmax = dims->nlg[2];

    const long stencil[3] = {istride,jstride,1};

    if(i1==i2){
        for(long i=imin; i<imax; i++){
            const long ishift = i * istride;
            for(long j=jmin; j<jmax; j++){
                const long jshift = j * jstride;
                for(long k=kmin; k<kmax; k++){
                    const long ijk = ishift + jshift + k;
                    flux[ijk] = -2.0 * strain_rate[ijk] * viscosity[ijk + stencil[i1]];

                }
            }
        }
    }
    else{
        for(long i=imin; i<imax; i++){
            const long ishift = i * istride;
            for(long j=jmin; j<jmax; j++){
                const long jshift = j * jstride;
                for(long k=kmin; k<kmax; k++){
                    const long ijk = ishift + jshift + k;
                    const double visc_interp = 0.25 * (viscosity[ijk] + viscosity[ijk + stencil[i1]]
                    + viscosity[ijk + stencil[i2]] +  viscosity[ijk + stencil[i1] + stencil[i2]] );
                    flux[ijk] = -2.0 * strain_rate[ijk] * visc_interp;

                }
            }
        }

    }


    return;
}


void compute_entropy_source(const struct DimStruct *dims, double* restrict viscosity, double* restrict strain_rate_mag, double* restrict temperature, double* restrict entropy_tendency){
    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0];
    const long jmax = dims->nlg[1];
    const long kmax = dims->nlg[2];

    for(long i=imin;i<imax;i++){
        const long ishift = i*istride ;
        for(long j=jmin;j<jmax;j++){
            const long jshift = j*jstride;
            for(long k=kmin;k<kmax;k++){
                const long ijk = ishift + jshift + k ;
                entropy_tendency[ijk] +=  strain_rate_mag[ijk] * strain_rate_mag[ijk] * viscosity[ijk] / temperature[ijk];
            }
        }
    }


}