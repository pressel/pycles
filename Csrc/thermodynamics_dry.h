#pragma once
#include "parameters.h"
#include "grid.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include<stdio.h>

inline double eos_c(double pd, double s){
    return T_tilde*(exp( (s - sd_tilde + Rd *log(pd/p_tilde))/cpd));
};

inline double eos_thli_c(double pd, double thli){
    return thli * exner_c(pd);
};

void eos_update(struct DimStruct *dims, double* restrict pd, double* restrict s, double* restrict T,
    double* restrict alpha ){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    T[ijk] = eos_c(pd[k],s[ijk]);
                    alpha[ijk] = alpha_c(pd[k],T[ijk],0.0,0.0);
                } // End k loop
        } // End j loop
    } // End i loop
    return;
};


void eos_update_thli(struct DimStruct *dims, double* restrict pd, double* restrict thli, double* restrict T, double* restrict s,
    double* restrict alpha ){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    T[ijk] = eos_thli_c(pd[k],thli[ijk]);
                    alpha[ijk] = alpha_c(pd[k],T[ijk],0.0,0.0);
                    s[ijk] = sd_c(pd[k], T[ijk]);
                } // End k loop
        } // End j loop
    } // End i loop
    return;
};


void buoyancy_update(struct DimStruct *dims, double* restrict alpha0, double* restrict alpha, double* restrict buoyancy, double* restrict wt){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0]-1;
    const ssize_t jmax = dims->nlg[1]-1;
    const ssize_t kmax = dims->nlg[2]-1;

    const double * metl = dims -> metl;

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                buoyancy[ijk] = buoyancy_c(alpha0[k],alpha[ijk]);
            } // End k loop
        } // End j loop
    } // End i loop

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                wt[ijk] += interp_2(buoyancy[ijk],buoyancy[ijk+1]);
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}

void bvf_dry(struct DimStruct* dims,  double* restrict p0, double* restrict T,double* restrict theta, double* restrict bvf){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 1;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];
    const double dzi = 1.0/dims->dx[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                theta[ijk] = theta_c(p0[k],T[ijk]);
            } // End k loop
        } // End j loop
    } // End i loop

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin+1;k<kmax-1;k++){
                const ssize_t ijk = ishift + jshift + k;
                bvf[ijk] = g/theta[ijk]*(interp_2(theta[ijk],theta[ijk+1])-interp_2(theta[ijk-1],theta[ijk]))*dzi*dims->imetl_half[k];
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}


