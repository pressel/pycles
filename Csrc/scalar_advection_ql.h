#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "thermodynamic_functions.h"
#include "entropies.h"

#include "cc_statistics.h"




//void second_order_a_ql(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){
void second_order_a_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){
//    if (d==1){printf("2nd order QL Scalar Transport \n");}

//compute_advective_fluxes_a(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&DV.values[vel_shift],
//                                                   &PV.values[scalar_shift],&self.flux[flux_shift],d,self.order_sedimentation)
//void compute_advective_fluxes_a(struct DimStruct *dims, double* restrict rho0, double* rho0_half ,double* restrict velocity, double* restrict scalar,
//                                double* restrict flux, int d, int scheme)
//second_order_a_ql(dims, rho0, rho0_half, velocity, scalar, flux, d);
//
// --> velocity = pointer to first element of array of advecting velocity (u_i)
// --> scalar = pointer to first element of array of scalar that is advected (phi_j)
// --> flux = pointer to first element of array of flux_ij = u_i*phi_j

    // right dimensions ???? (compare to flux in ScalarAdvection)
    double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
//    double *vel_int_ing = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
//    double *vel_mean_ing = (double *)malloc(sizeof(double) * dims->nlg[2]);
    double *vel_mean = (double *)malloc(sizeof(double) * dims->nlg[2]);
    double *phi_int = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *phi_mean_int = (double *)malloc(sizeof(double) * dims->nlg[2]);

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

    const ssize_t imax = dims->nlg[0]-1;
    const ssize_t jmax = dims->nlg[1]-1;
    const ssize_t kmax = dims->nlg[2]-1;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];


    // (1) interpolation
    //     (a) velocity fields --> not necessary, since only scalars interpolated
    //     (b) scalar field
    for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    phi_int[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1]);
                }
            }
        }


    // (2) average velocity field and interpolated scalar field
    for(ssize_t k=kmin;k<kmax;k++){
        phi_mean_int[k] = 0;
        vel_mean[k] = 0;
        mean_eddy_flux[k] = 0;
        }
    horizontal_mean_return(dims, &phi_int[0], &phi_mean_int[0]);
    horizontal_mean_return(dims, &velocity[0], &vel_mean[0]);


    // (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    eddy_flux[ijk] = (phi_int[ijk] - phi_mean_int[k]) * (velocity[ijk] - vel_mean[k]) * rho0[k];
                    flux[ijk] = phi_int[ijk] * velocity[ijk] * rho0[k];
                    // flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1]) * velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    eddy_flux[ijk] = (phi_int[ijk] - phi_mean_int[k]) * (velocity[ijk] - vel_mean[k]) * rho0_half[k];
                    flux[ijk] = phi_int[ijk] * velocity[ijk] * rho0_half[k];
                    // flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else


    // (4) compute mean eddy flux
    horizontal_mean_return(dims, &eddy_flux[0], &mean_eddy_flux[0]);


    // (5) compute QL flux: flux = flux - eddy_flux + mean_eddy_flux
    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;
                flux[ijk] = flux[ijk] - eddy_flux[ijk] + mean_eddy_flux[k];
//                flux[ijk] = flux[ijk];
            }
        }
    }


    free(eddy_flux);
    free(mean_eddy_flux);
    free(vel_mean);
    free(phi_int);
    free(phi_mean_int);

    return;
}









void fourth_order_a_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){
    if (d==1){printf("4th order QL Scalar Transport \n");}

    double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
    double *vel_mean = (double *)malloc(sizeof(double) * dims->nlg[2]);
    double *phi_int = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *phi_mean_int = (double *)malloc(sizeof(double) * dims->nlg[2]);

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 1;
    const ssize_t jmin = 1;
    const ssize_t kmin = 1;

    const ssize_t imax = dims->nlg[0]-2;
    const ssize_t jmax = dims->nlg[1]-2;
    const ssize_t kmax = dims->nlg[2]-2;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sm1 = -sp1 ;


    // (1) interpolation
    //     (a) velocity fields --> not necessary, since only scalars interpolated
    //     (b) scalar field
    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                phi_int[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2]);
            }
        }
    }


    // (2) average velocity field and interpolated scalar field
//    for(ssize_t k=kmin;k<kmax;k++){
//        phi_mean_int[k] = 0;
//        vel_mean[k] = 0;
//        mean_eddy_flux[k] = 0;
//        }
    horizontal_mean_return(dims, &phi_int[0], &phi_mean_int[0]);
    horizontal_mean_return(dims, &velocity[0], &vel_mean[0]);


    // (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    eddy_flux[ijk] = (phi_int[ijk] - phi_mean_int[k]) * (velocity[ijk] - vel_mean[k]) * rho0[k];
                    flux[ijk] = phi_int[ijk] * velocity[ijk] * rho0[k];
                    // flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    eddy_flux[ijk] = (phi_int[ijk] - phi_mean_int[k]) * (velocity[ijk] - vel_mean[k]) * rho0_half[k];
                    flux[ijk] = phi_int[ijk] * velocity[ijk] * rho0_half[k];
                    // flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else


    // (4) compute mean eddy flux
    horizontal_mean_return(dims, &eddy_flux[0], &mean_eddy_flux[0]);


    // (5) compute QL flux: flux = flux - eddy_flux + mean_eddy_flux
    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;
                flux[ijk] = flux[ijk] - eddy_flux[ijk] + mean_eddy_flux[k];
//                flux[ijk] = flux[ijk];
            }
        }
    }

    free(eddy_flux);
    free(mean_eddy_flux);
    free(vel_mean);
    free(phi_int);
    free(phi_mean_int);

    return;
}



void weno_fifth_order_a_ql(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 2;
    const ssize_t jmin = 2;
    const ssize_t kmin = 2;

    const ssize_t imax = dims->nlg[0]-3;
    const ssize_t jmax = dims->nlg[1]-3;
    const ssize_t kmax = dims->nlg[2]-3;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2]);

                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2]);

                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void weno_fifth_order_a_decomp(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 2;
    const ssize_t jmin = 2;
    const ssize_t kmin = 2;

    const ssize_t imax = dims->nlg[0]-3;
    const ssize_t jmax = dims->nlg[1]-3;
    const ssize_t kmax = dims->nlg[2]-3;

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2]);

                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk + sm2],
                                                     scalar[ijk + sm1],
                                                     scalar[ijk],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk + sp2]);

                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk + sp3],
                                                     scalar[ijk + sp2],
                                                     scalar[ijk + sp1],
                                                     scalar[ijk],
                                                     scalar[ijk + sm1]);

                    flux[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}



