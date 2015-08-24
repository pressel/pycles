#pragma once
#include "grid.h"
#include "advection_interpolation.h"

void second_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 0;
    const size_t jmin = 0;
    const size_t kmin = 0;

    const size_t imax = dims->nlg[0]-1;
    const size_t jmax = dims->nlg[1]-1;
    const size_t kmax = dims->nlg[2]-1;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1]) * velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else

    return;
}

void fourth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 1;
    const size_t jmin = 1;
    const size_t kmin = 1;

    const size_t imax = dims->nlg[0]-2;
    const size_t jmax = dims->nlg[1]-2;
    const size_t kmax = dims->nlg[2]-2;


    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];
    const size_t sp2 = 2 * sp1;
    const size_t sm1 = -sp1 ;


    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } //end if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // end else
    return;
}

void sixth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 2;
    const size_t jmin = 2;
    const size_t kmin = 2;

    const size_t imax = dims->nlg[0]-3;
    const size_t jmax = dims->nlg[1]-3;
    const size_t kmax = dims->nlg[2]-3;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];
    const size_t sp2 = 2 * sp1;
    const size_t sp3 = 3 * sp1;
    const size_t sm1 = -sp1 ;
    const size_t sm2 = -2*sp1;

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_6(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3])*velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_6(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void eighth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 3;
    const size_t jmin = 3;
    const size_t kmin = 3;

    const size_t imax = dims->nlg[0]-4;
    const size_t jmax = dims->nlg[1]-4;
    const size_t kmax = dims->nlg[2]-4;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];
    const size_t sp2 = 2 * sp1;
    const size_t sp3 = 3 * sp1;
    const size_t sp4 = 4 * sp1;
    const size_t sm1 = -sp1 ;
    const size_t sm2 = -2*sp1;
    const size_t sm3 = -3*sp1;

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_8(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4])*velocity[ijk]*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    flux[ijk] = interp_8(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4])*velocity[ijk]*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    return;
}

void upwind_first_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 1;
    const size_t jmin = 1;
    const size_t kmin = 1;

    const size_t imax = dims->nlg[0]-1;
    const size_t jmax = dims->nlg[1]-1;
    const size_t kmax = dims->nlg[2]-1;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = scalar[ijk];
                    // Up wind for negative velocity
                    const double phim =scalar[ijk+sp1];
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = scalar[ijk];
                    // Up wind for negative velocity
                    const double phim =scalar[ijk+sp1];
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void weno_third_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 1;
    const size_t jmin = 1;
    const size_t kmin = 1;

    const size_t imax = dims->nlg[0]-2;
    const size_t jmax = dims->nlg[1]-2;
    const size_t kmax = dims->nlg[2]-2;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];
    const size_t sp2 = 2 * sp1;
    const size_t sm1 = -sp1 ;

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno3(scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1]);
                    // Up wind for negative velocity
                    const double phim = interp_weno3(scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno3(scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1]);
                    // Up wind for negative velocity
                    const double phim = interp_weno3(scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}


void weno_fifth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 2;
    const size_t jmin = 2;
    const size_t kmin = 2;

    const size_t imax = dims->nlg[0]-3;
    const size_t jmax = dims->nlg[1]-3;
    const size_t kmax = dims->nlg[2]-3;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];
    const size_t sp2 = 2 * sp1;
    const size_t sp3 = 3 * sp1;
    const size_t sm1 = -sp1 ;
    const size_t sm2 = -2*sp1;

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2]);
                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2]);
                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void weno_seventh_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 3;
    const size_t jmin = 3;
    const size_t kmin = 3;

    const size_t imax = dims->nlg[0]-4;
    const size_t jmax = dims->nlg[1]-4;
    const size_t kmax = dims->nlg[2]-4;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];
    const size_t sp2 = 2 * sp1;
    const size_t sp3 = 3 * sp1;
    const size_t sp4 = 4 * sp1;
    const size_t sm1 = -sp1 ;
    const size_t sm2 = -2*sp1;
    const size_t sm3 = -3*sp1;

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno7(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3]);
                    // Up wind for negative velocity
                    const double phim = interp_weno7(scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                     //Upwind for positive velocity
                    const double phip = interp_weno7(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3]);
                    // Up wind for negative velocity
                    const double phim = interp_weno7(scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    return;
}

void weno_ninth_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 4;
    const size_t jmin = 4;
    const size_t kmin = 4;

    const size_t imax = dims->nlg[0]-5;
    const size_t jmax = dims->nlg[1]-5;
    const size_t kmax = dims->nlg[2]-5;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];
    const size_t sp2 = 2 * sp1;
    const size_t sp3 = 3 * sp1;
    const size_t sp4 = 4 * sp1;
    const size_t sp5 = 5 * sp1;
    const size_t sm1 = -sp1 ;
    const size_t sm2 = -2*sp1;
    const size_t sm3 = -3*sp1;
    const size_t sm4 = -4*sp1;

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno9(scalar[ijk+sm4],scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4]);
                    // Up wind for negative velocity
                    const double phim = interp_weno9(scalar[ijk+sp5],scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2],scalar[ijk+sm3]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno9(scalar[ijk+sm4],scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4]);
                    // Up wind for negative velocity
                    const double phim = interp_weno9(scalar[ijk+sp5],scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2],scalar[ijk+sm3]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void weno_eleventh_order_a(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    const size_t imin = 5;
    const size_t jmin = 5;
    const size_t kmin = 5;

    const size_t imax = dims->nlg[0]-6;
    const size_t jmax = dims->nlg[1]-6;
    const size_t kmax = dims->nlg[2]-6;

    const size_t stencil[3] = {istride,jstride,1};
    const size_t sp1 = stencil[d];
    const size_t sp2 = 2 * sp1;
    const size_t sp3 = 3 * sp1;
    const size_t sp4 = 4 * sp1;
    const size_t sp5 = 5 * sp1;
    const size_t sp6 = 6 * sp1;
    const size_t sm1 = -sp1 ;
    const size_t sm2 = -2*sp1;
    const size_t sm3 = -3*sp1;
    const size_t sm4 = -4*sp1;
    const size_t sm5 = -5*sp1;

    if(d==2){
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno11(scalar[ijk+sm5],scalar[ijk+sm4],scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4],scalar[ijk+sp5]);
                    // Up wind for negative velocity
                    const double phim = interp_weno11(scalar[ijk+sp6],scalar[ijk+sp5],scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2],scalar[ijk+sm3],scalar[ijk+sm4]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End if
    else{
        for(size_t i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(size_t j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(size_t k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno11(scalar[ijk+sm5],scalar[ijk+sm4],scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4],scalar[ijk+sp5]);
                    // Up wind for negative velocity
                    const double phim = interp_weno11(scalar[ijk+sp6],scalar[ijk+sp5],scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2],scalar[ijk+sm3],scalar[ijk+sm4]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                } // End k loop
            } // End j loop
        } // End i loop
    } // End else
    return;
}

void compute_advective_fluxes_a(struct DimStruct *dims, double* restrict rho0, double* rho0_half ,double* restrict velocity, double* restrict scalar,
                                double* restrict flux, int d, int scheme){
    switch(scheme){
        case 1:
            upwind_first_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 2:
            second_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 3:
            weno_third_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 4:
            fourth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 5:
            weno_fifth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 6:
            sixth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 7:
            weno_seventh_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 8:
            eighth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 9:
            weno_ninth_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 11:
            weno_eleventh_order_a(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
    };
};

void compute_flux_divergence(const struct DimStruct *dims, double* restrict alpha0, double* restrict alpha0_half, const double* restrict flux, double* restrict tendency, const double dx, int d){

    size_t i,j,k;
    const size_t imin = 1;
    const size_t jmin = 1;
    const size_t kmin = 1;

    const size_t imax = dims->nlg[0]-2;
    const size_t jmax = dims->nlg[1]-2;
    const size_t kmax = dims->nlg[2]-2;

    const size_t istride = dims->nlg[1] * dims->nlg[2];
    const size_t jstride = dims->nlg[2];

    size_t sm1 ;
    switch(d){
        case 0:
            sm1 = -istride;
        case 1:
            sm1 = -jstride;
        case 2:
            sm1 = -1;
    }

    const double dxi = 1.0/dx ;

    if(d==2){
        for(i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k;
                    tendency[ijk] = tendency[ijk] - alpha0[k]*(flux[ijk+sm1] - flux[ijk])*dxi;
                } // End k loop
            } // End j loop
         } // End i loop
     } // End if
     else{
        for(i=imin;i<imax;i++){
            const size_t ishift = i*istride ;
            for(j=jmin;j<jmax;j++){
                const size_t jshift = j*jstride;
                for(k=kmin;k<kmax;k++){
                    const size_t ijk = ishift + jshift + k;
                    tendency[ijk] = tendency[ijk] - (flux[ijk+sm1] - flux[ijk])*dxi;
                }// End k loop
             } // End j loop
         } // End i loop
     }//end else

    return;
};