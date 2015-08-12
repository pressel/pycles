#pragma once
#include "grid.h"

void compute_diffusive_flux(const struct DimStruct *dims, double* restrict vgrad1, double* restrict vgrad2, double* restrict viscosity ,double* restrict flux){

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
                flux[ijk] = -(vgrad1[ijk] + vgrad2[ijk])*viscosity[ijk];
            };
        };
    };



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
                entropy_tendency[ijk] += 2.0 * strain_rate_mag[ijk] * strain_rate_mag[ijk] * viscosity[ijk] / temperature[ijk];
            }
        }
    }


}