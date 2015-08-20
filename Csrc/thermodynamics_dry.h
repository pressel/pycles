#pragma once
#include "parameters.h"
#include "grid.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include<stdio.h>

inline double eos_c(double pd, double s){
    return T_tilde*(exp( (s - sd_tilde + Rd *log(pd/p_tilde))/cpd));
};


void eos_update(struct DimStruct *dims, double* restrict pd, double* restrict s, double* restrict T,
    double* restrict alpha ){

    long i,j,k;
    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];
    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;
    const long imax = dims->nlg[0];
    const long jmax = dims->nlg[1];
    const long kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
       const long ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const long jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k;
                    T[ijk] = eos_c(pd[k],s[ijk]);
                    alpha[ijk] = alpha_c(pd[k],T[ijk],0.0,0.0);

                };
        };
    };

    return;
};

void buoyancy_update(struct DimStruct *dims, double* restrict alpha0, double* restrict alpha, double* restrict buoyancy, double* restrict wt){

    long i,j,k;
    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];
    const long imin = 1;
    const long jmin = 1;
    const long kmin = 1;
    const long imax = dims->nlg[0]-2;
    const long jmax = dims->nlg[1]-2;
    const long kmax = dims->nlg[2]-2;

    for (i=imin; i<imax; i++){
       const long ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const long jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k;
                    buoyancy[ijk] = buoyancy_c(alpha0[k],alpha[ijk]);
                };
        };
    };


    for (i=imin; i<imax; i++){
       const long ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const long jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k;
                    wt[ijk] = wt[ijk] + interp_4(buoyancy[ijk-1],buoyancy[ijk],buoyancy[ijk+1],buoyancy[ijk+2]);
                };
        };
    };



    return;
}


void bvf_dry(struct DimStruct* dims,  double* restrict p0, double* restrict T,double* restrict theta, double* restrict bvf){

    long i,j,k;
    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];
    const long imin = 1;
    const long jmin = 1;
    const long kmin = 1;
    const long imax = dims->nlg[0]-2;
    const long jmax = dims->nlg[1]-2;
    const long kmax = dims->nlg[2]-2;
    const double dzi = 1.0/dims->dx[2];


    for (i=imin; i<imax; i++){
       const long ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const long jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k;
                    theta[ijk] = theta_c(p0[k],T[ijk]);
                };
        };
    };

    for (i=imin; i<imax; i++){
       const long ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const long jshift = j * jstride;
                for (k=kmin+1;k<kmax-1;k++){
                    const long ijk = ishift + jshift + k;
                    bvf[ijk] = g/theta[ijk]*(interp_2(theta[ijk],theta[ijk+1])-interp_2(theta[ijk-1],theta[ijk]))*dzi;
                };
        };
    };




    return;
}


