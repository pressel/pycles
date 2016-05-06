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
    double *vel_mean = (double *)malloc(sizeof(double) * dims->nlg[2]);
    double *phi_int = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *phi_mean_int = (double *)malloc(sizeof(double) * dims->nlg[2]);

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

//    const ssize_t imax = dims->nlg[0]-1;
//    const ssize_t jmax = dims->nlg[1]-1;
//    const ssize_t kmax = dims->nlg[2]-1;
    ssize_t imax = dims->nlg[0];
    ssize_t jmax = dims->nlg[1];
    ssize_t kmax = dims->nlg[2];

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
    horizontal_mean(dims, &phi_int[0], &phi_mean_int[0]);
    horizontal_mean(dims, &velocity[0], &vel_mean[0]);

    // (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
    imax = dims->nlg[0]-1;
    jmax = dims->nlg[1]-1;
    kmax = dims->nlg[2]-1;
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
    horizontal_mean(dims, &eddy_flux[0], &mean_eddy_flux[0]);

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




void fourth_order_a_decomp(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict velocity, double* restrict scalar, double* restrict flux, int d){

    if(d==2){printf("4th order Scalar Transport decomp \n");}

    double *vel_mean = (double *)malloc(sizeof(double) * dims->nlg[2]);
//    double *phi_fluc = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *phi_mean = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *phi_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    ssize_t imin = 0;
    ssize_t jmin = 0;
    ssize_t kmin = 0;
    ssize_t imax = dims->nlg[0];
    ssize_t jmax = dims->nlg[1];
    ssize_t kmax = dims->nlg[2];

    ssize_t i,j,k;


    // (1) compute mean fields
    for(k=kmin; k<kmax; k++){
        phi_mean_[k] = 0.0;
    }

    horizontal_mean(dims, &velocity[0], &vel_mean[0]);
    horizontal_mean(dims, &scalar[0], &phi_mean_[0]);

    for(i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                phi_mean[ijk] = 1.0;
                phi_mean_[k] = 1.0;
//                phi_mean[ijk] = phi_mean_[k];
//                phi_fluc[ijk] = scalar[ijk] - phi_mean[ijk];
            }
        }
    }


    // (2) compute flux
//    double *eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
//    double *mix_flux_phimean = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
//    double *mix_flux_phiprime = (double *)malloc(sizeof(double) * dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
//    double *mean_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
//    double *flux_old = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sm1 = -sp1 ;

    imin = 1;
    jmin = 1;
    kmin = 1;
    imax = dims->nlg[0]-2;
    jmax = dims->nlg[1]-2;
    kmax = dims->nlg[2]-2;

    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;

//                    const double eddy_flux = interp_4(phi_fluc[ijk+sm1],phi_fluc[ijk],phi_fluc[ijk+sp1],phi_fluc[ijk+sp2])
//                                        * (velocity[ijk]-vel_mean[k]) * rho0[k];
//                    const double mix_flux_phimean = interp_4(phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2])
//                                        * (velocity[ijk]-vel_mean[k]) * rho0[k];
//                    const double mix_flux_phiprime = interp_4(phi_fluc[ijk+sm1],phi_fluc[ijk],phi_fluc[ijk+sp1],phi_fluc[ijk+sp2])
//                                        * vel_mean[k] * rho0[k];
//                    const double mean_flux = interp_4(phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2])*velocity[ijk]
//                                        * vel_mean[k] * rho0[k];
//                    flux[ijk] = eddy_flux + mix_flux_phimean + mix_flux_phiprime + mean_flux;
//                    eddy_flux[ijk] = interp_4(phi_fluc[ijk+sm1],phi_fluc[ijk],phi_fluc[ijk+sp1],phi_fluc[ijk+sp2])
//                                        * (velocity[ijk]-vel_mean[k]) * rho0[k];
//                    mix_flux_phimean[ijk] = interp_4(phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2])
//                                        * (velocity[ijk]-vel_mean[k]) * rho0[k];
//                    mix_flux_phiprime[ijk] = interp_4(phi_fluc[ijk+sm1],phi_fluc[ijk],phi_fluc[ijk+sp1],phi_fluc[ijk+sp2])
//                                        * vel_mean[k] * rho0[k];
//                    mean_flux[k] = interp_4(phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2])*velocity[ijk]
//                                        * vel_mean[k] * rho0[k];
//                    flux[ijk] = eddy_flux[ijk] + mix_flux_phimean[ijk] + mix_flux_phiprime[ijk] + mean_flux[k];

//                    flux_old[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0[k];
                    flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0[k];
//                    flux[ijk] = flux_old[ijk];
                } // End k loop
            } // End j loop
        } // End i loop
    } //end if
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
//                    const double eddy_flux = interp_4(phi_fluc[ijk+sm1],phi_fluc[ijk],phi_fluc[ijk+sp1],phi_fluc[ijk+sp2])
//                                        * (velocity[ijk]-vel_mean[k]) * rho0_half[k];
//                    const double mix_flux_phimean = interp_4(phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2])
//                                        * (velocity[ijk]-vel_mean[k]) * rho0_half[k];
//                    const double mix_flux_phiprime = interp_4(phi_fluc[ijk+sm1],phi_fluc[ijk],phi_fluc[ijk+sp1],phi_fluc[ijk+sp2])
//                                        * vel_mean[k] * rho0_half[k];
//                    const double mean_flux = interp_4(phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2])*velocity[ijk]
//                                        * vel_mean[k] * rho0_half[k];
//                    flux[ijk] = eddy_flux + mix_flux_phimean + mix_flux_phiprime + mean_flux;
//                    eddy_flux[ijk] = interp_4(phi_fluc[ijk+sm1],phi_fluc[ijk],phi_fluc[ijk+sp1],phi_fluc[ijk+sp2])
//                                        * (velocity[ijk]-vel_mean[k]) * rho0_half[k];
//                    mix_flux_phimean[ijk] = interp_4(phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2])
//                                        * (velocity[ijk]-vel_mean[k]) * rho0_half[k];
//                    mix_flux_phiprime[ijk] = interp_4(phi_fluc[ijk+sm1],phi_fluc[ijk],phi_fluc[ijk+sp1],phi_fluc[ijk+sp2])
//                                        * vel_mean[k] * rho0_half[k];
//                    mean_flux[k] = interp_4(phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2])*velocity[ijk]
//                                        * vel_mean[k] * rho0_half[k];
//                    flux[ijk] = eddy_flux[ijk] + mix_flux_phimean[ijk] + mix_flux_phiprime[ijk] + mean_flux[k];

//                    flux_old[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0_half[k];
                    flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0_half[k];
//                    flux[ijk] = flux_old[ijk];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else
    return;

//    free(eddy_flux);
//    free(mix_flux_phiprime);
//    free(mix_flux_phimean);
//    free(mean_flux);
//    free(flux_old);
//    free(phi_fluc);
    free(phi_mean);
    free(phi_mean_);
    free(vel_mean);

}






void fourth_order_a_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half, double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){
    if (d==1){printf("4th order QL Scalar Transport \n");}

    double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
    double *vel_mean = (double *)malloc(sizeof(double) * dims->nlg[2]);
    double *phi_int = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
//    double *phi_int_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *phi_int_mean = (double *)malloc(sizeof(double) * dims->nlg[2]);

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
//        phi_int_mean[k] = 0;
//        vel_mean[k] = 0;
//        mean_eddy_flux[k] = 0;
//        }
    horizontal_mean(dims, &phi_int[0], &phi_int_mean[0]);
    horizontal_mean(dims, &velocity[0], &vel_mean[0]);


    // (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
//                    phi_int_fluc[ijk] = phi_int[ijk] - phi_int_mean[k];
                    eddy_flux[ijk] = (phi_int[ijk] - phi_int_mean[k]) * (velocity[ijk] - vel_mean[k]) * rho0[k];
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
                    eddy_flux[ijk] = (phi_int[ijk] - phi_int_mean[k]) * (velocity[ijk] - vel_mean[k]) * rho0_half[k];
                    flux[ijk] = phi_int[ijk] * velocity[ijk] * rho0_half[k];
                    // flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else


    // (4) compute mean eddy flux
    horizontal_mean(dims, &eddy_flux[0], &mean_eddy_flux[0]);


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
    free(phi_int_mean);

    return;
}










void weno_fifth_order_a_decomp(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
                const double* restrict velocity, const double* restrict scalar,
                double* restrict flux, int d){
//void weno_fifth_order_a_decomp(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
//                const double* restrict velocity, const double* restrict scalar,
//                double* restrict flux, double* restrict flux_old, int d){
    if (d==1){printf("Scalar Advection: WENO 5 decomposition \n");}
//    printf("Scalar Advection: WENO 5 decomposition \n");
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    ssize_t imin = 0;
    ssize_t jmin = 0;
    ssize_t kmin = 0;
    ssize_t imax = dims->nlg[0];
    ssize_t jmax = dims->nlg[1];
    ssize_t kmax = dims->nlg[2];

    const ssize_t stencil[3] = {istride,jstride,1};
    const ssize_t sp1 = stencil[d];
    const ssize_t sp2 = 2 * sp1;
    const ssize_t sp3 = 3 * sp1;
    const ssize_t sm1 = -sp1 ;
    const ssize_t sm2 = -2*sp1;

    // (1) average velocity and scalar field
    double *vel_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *vel_mean = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *vel_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);
    double *phi_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *phi_mean = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *phi_mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);

    for(ssize_t k=kmin;k<kmax;k++){
        phi_mean[k] = 0;
        vel_mean[k] = 0;
        }
    // horizontal_mean(dims, &vel_fluc[0], &phi_mean[0]);
    horizontal_mean_const(dims, &velocity[0], &vel_mean_[0]);
    horizontal_mean_const(dims, &scalar[0], &phi_mean_[0]);

    // (2) compute eddy fields
    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                int ijk = ishift + jshift + k;
                vel_mean[ijk] = vel_mean_[k];
                phi_mean[ijk] = phi_mean_[k];

                vel_fluc[ijk] = velocity[ijk] - vel_mean[ijk];
                phi_fluc[ijk] = scalar[ijk] - phi_mean[ijk];
            }
        }
    }

    // (3) Compute Fluxes
    // mix_flux_phiprime = <u>phi'
    // mix_flux_phimean = u' <phi>
    // eddy_flux = u' phi'
    // mean_flux = <u><phi>
    double *flux_old = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *mix_flux_phiprime = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *mix_flux_phimean = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
    double *mean_flux = (double *)malloc(sizeof(double)*dims->nlg[2]);        // ??? 1D profile sufficient!?
//    double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
//    double eddy_flux;
//    double mix_flux_phiprime;
//    double mix_flux_phimean;
//    double mean_flux;

    imin = 2;
    jmin = 2;
    kmin = 2;
    imax = dims->nlg[0]-3;
    jmax = dims->nlg[1]-3;
    kmax = dims->nlg[2]-3;

    double phip = 0.0;      //???? do I need const double phip ??? Difference to declaring it within loop?
    double phim = 0.0;
    if(d==2){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    // (3a) <u>phi'
                    phip = interp_weno5(phi_fluc[ijk + sm2],
                                        phi_fluc[ijk + sm1],
                                        phi_fluc[ijk],
                                        phi_fluc[ijk + sp1],
                                        phi_fluc[ijk + sp2]);
                    phim = interp_weno5(phi_fluc[ijk + sp3],
                                        phi_fluc[ijk + sp2],
                                        phi_fluc[ijk + sp1],
                                        phi_fluc[ijk],
                                        phi_fluc[ijk + sm1]);

                    // mean_eddy_flux = rms(v')*rms(phi')*covar(v',phi') // ????? different computation of mean eddy-flux
                    eddy_flux[ijk] =  0.5 * ((vel_fluc[ijk]+fabs(vel_fluc[ijk]))*phip + (vel_fluc[ijk]-fabs(vel_fluc[ijk]))*phim)*rho0[k];
                    mix_flux_phiprime[ijk] =  0.5 * ((vel_mean[ijk]+fabs(vel_mean[ijk]))*phip + (vel_mean[ijk]-fabs(vel_mean[ijk]))*phim)*rho0[k];
//                    eddy_flux =  0.5 * ((vel_fluc[ijk]+fabs(vel_fluc[ijk]))*phip + (vel_fluc[ijk]-fabs(vel_fluc[ijk]))*phim)*rho0[k];
//                    mix_flux_phiprime =  0.5 * ((vel_mean[k]+fabs(vel_mean[k]))*phip + (vel_mean[k]-fabs(vel_mean[k]))*phim)*rho0[k];

                    phip = interp_weno5(phi_mean[ijk+sm2],phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2]);
                    phim = interp_weno5(phi_mean[ijk+sp3],phi_mean[ijk+sp2],phi_mean[ijk+sp1],phi_mean[ijk],phi_mean[ijk+sm1]);
//                    phip = interp_weno5(phi_mean[k - 2],phi_mean[k - 1],phi_mean[k],phi_mean[k + 1],phi_mean[k + 2]);
//                    phim = interp_weno5(phi_mean[k + 3],phi_mean[k + 2],phi_mean[k + 1],phi_mean[k],phi_mean[k - 1]);

                    mix_flux_phimean[ijk] =  0.5 * ((vel_fluc[ijk]+fabs(vel_fluc[ijk]))*phip + (vel_fluc[ijk]-fabs(vel_fluc[ijk]))*phim)*rho0[k];
                    mean_flux[k] =  0.5 * ((vel_mean[ijk]+fabs(vel_mean[ijk]))*phip + (vel_mean[ijk]-fabs(vel_mean[ijk]))*phim)*rho0[k];
//                    mix_flux_phimean =  0.5 * ((vel_fluc[ijk]+fabs(vel_fluc[ijk]))*phip + (vel_fluc[ijk]-fabs(vel_fluc[ijk]))*phim)*rho0[k];
//                    mean_flux =  0.5 * ((vel_mean[k]+fabs(vel_mean[k]))*phip + (vel_mean[k]-fabs(vel_mean[k]))*phim)*rho0[k];

                    flux[ijk] = mean_flux[k] + mix_flux_phiprime[ijk] + mix_flux_phimean[ijk] + eddy_flux[ijk];
//                    flux[ijk] = mean_flux + mix_flux_phiprime + mix_flux_phimean + eddy_flux;

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

                    flux_old[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
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
                    phip = interp_weno5(phi_fluc[ijk + sm2],
                                        phi_fluc[ijk + sm1],
                                        phi_fluc[ijk],
                                        phi_fluc[ijk + sp1],
                                        phi_fluc[ijk + sp2]);
                    phim = interp_weno5(phi_fluc[ijk + sp3],
                                        phi_fluc[ijk + sp2],
                                        phi_fluc[ijk + sp1],
                                        phi_fluc[ijk],
                                        phi_fluc[ijk + sm1]);

                    // mean_eddy_flux = rms(v')*rms(phi')*covar(v',phi') // ????? different computation of mean eddy-flux
                    eddy_flux[ijk] =  0.5 * ((vel_fluc[ijk]+fabs(vel_fluc[ijk]))*phip + (vel_fluc[ijk]-fabs(vel_fluc[ijk]))*phim)*rho0_half[k];
                    mix_flux_phiprime[ijk] =  0.5 * ((vel_mean[ijk]+fabs(vel_mean[ijk]))*phip + (vel_mean[ijk]-fabs(vel_mean[ijk]))*phim)*rho0_half[k];
//                    eddy_flux =  0.5 * ((vel_fluc[ijk]+fabs(vel_fluc[ijk]))*phip + (vel_fluc[ijk]-fabs(vel_fluc[ijk]))*phim)*rho0_half[k];
//                    mix_flux_phiprime =  0.5 * ((vel_mean[k]+fabs(vel_mean[k]))*phip + (vel_mean[k]-fabs(vel_mean[k]))*phim)*rho0_half[k];

                    phip = interp_weno5(phi_mean[ijk+sm2],phi_mean[ijk+sm1],phi_mean[ijk],phi_mean[ijk+sp1],phi_mean[ijk+sp2]);
                    phim = interp_weno5(phi_mean[ijk+sp3],phi_mean[ijk+sp2],phi_mean[ijk+sp1],phi_mean[ijk],phi_mean[ijk+sm1]);

                    mix_flux_phimean[ijk] =  0.5 * ((vel_fluc[ijk]+fabs(vel_fluc[ijk]))*phip + (vel_fluc[ijk]-fabs(vel_fluc[ijk]))*phim)*rho0_half[k];
                    mean_flux[k] =  0.5 * ((vel_mean[ijk]+fabs(vel_mean[ijk]))*phip + (vel_mean[ijk]-fabs(vel_mean[ijk]))*phim)*rho0_half[k];
//                    mix_flux_phimean =  0.5 * ((vel_fluc[ijk]+fabs(vel_fluc[ijk]))*phip + (vel_fluc[ijk]-fabs(vel_fluc[ijk]))*phim)*rho0_half[k];
//                    mean_flux =  0.5 * ((vel_mean[k]+fabs(vel_mean[k]))*phip + (vel_mean[k]-fabs(vel_mean[k]))*phim)*rho0_half[k];      // ?? 1D profile sufficient

                    flux[ijk] = mean_flux[k] + mix_flux_phiprime[ijk] + mix_flux_phimean[ijk] + eddy_flux[ijk];
//                    flux[ijk] = mean_flux + mix_flux_phiprime + mix_flux_phimean + eddy_flux;

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

                    flux_old[ijk] =  0.5 * ((velocity[ijk]+fabs(velocity[ijk]))*phip + (velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    free(flux_old);

    free(mix_flux_phiprime);
    free(mix_flux_phimean);
    free(eddy_flux);
    free(mean_flux);
    free(vel_fluc);
    free(vel_mean);
    free(vel_mean_);
    free(phi_fluc);
    free(phi_mean);
    free(phi_mean_);
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