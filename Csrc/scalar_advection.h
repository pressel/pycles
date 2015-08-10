#pragma once
#include "grid.h"
#include "advection_interpolation.h"

void second_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];


    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1]) * velocity[ijk]*rho0[k];
                };
            };
        };

    } // end if
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    flux[ijk] = interp_2(scalar[ijk],scalar[ijk+sp1])*velocity[ijk]*rho0_half[k];
                };
            };
        };

    }; // end else

    return;
}

void fourth_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){

    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sp2 = 2 * sp1;
    const long sm1 = -sp1 ;


    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0[k];
                };
            };
        };
    } //end if
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    flux[ijk] = interp_4(scalar[ijk+sm1],scalar[ijk],scalar[ijk+sp1],scalar[ijk+sp2])*velocity[ijk]*rho0_half[k];
                };
            };
        };
    }; // end else
    return;
}

void sixth_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){


    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sp2 = 2 * sp1;
    const long sp3 = 3 * sp1;
    const long sm1 = -sp1 ;
    const long sm2 = -2*sp1;



    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    flux[ijk] = interp_6(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3])*velocity[ijk]*rho0[k];
                };
            };
        };
    }
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    flux[ijk] = interp_6(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3])*velocity[ijk]*rho0_half[k];
                };
            };
        };
    };

    return;
}

void eighth_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){


    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sp2 = 2 * sp1;
    const long sp3 = 3 * sp1;
    const long sp4 = 4 * sp1;
    const long sm1 = -sp1 ;
    const long sm2 = -2*sp1;
    const long sm3 = -3*sp1;



    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    flux[ijk] = interp_8(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4])*velocity[ijk]*rho0[k];
                };
            };
        };
    }
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    flux[ijk] = interp_8(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4])*velocity[ijk]*rho0_half[k];
                };
            };
        };
    };

    return;
}

void upwind_first(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){


    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sm1 = -sp1 ;




    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = scalar[ijk];
                    // Up wind for negative velocity
                    const double phim =scalar[ijk+sp1];
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                };
            };
        };
    }
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = scalar[ijk];
                    // Up wind for negative velocity
                    const double phim =scalar[ijk+sp1];
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                };
            };
        };
    };

    return;
}

void weno_third_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){


    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sp2 = 2 * sp1;
    const long sp3 = 3 * sp1;
    const long sm1 = -sp1 ;
    const long sm2 = -2*sp1;



    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno3(scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1]);
                    // Up wind for negative velocity
                    const double phim = interp_weno3(scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                };
            };
        };
    }
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno3(scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1]);
                    // Up wind for negative velocity
                    const double phim = interp_weno3(scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                };
            };
        };
    };

    return;
}


void weno_fifth_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){


    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sp2 = 2 * sp1;
    const long sp3 = 3 * sp1;
    const long sm1 = -sp1 ;
    const long sm2 = -2*sp1;



    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2]);
                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                };
            };
        };
    }
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno5(scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2]);
                    // Up wind for negative velocity
                    const double phim = interp_weno5(scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                };
            };
        };
    };

    return;
}

void weno_seventh_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){


    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sp2 = 2 * sp1;
    const long sp3 = 3 * sp1;
    const long sp4 = 4 * sp1;
    const long sm1 = -sp1 ;
    const long sm2 = -2*sp1;
    const long sm3 = -3*sp1;



    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno7(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3]);
                    // Up wind for negative velocity
                    const double phim = interp_weno7(scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                };
            };
        };
    }
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                     //Upwind for positive velocity
                    const double phip = interp_weno7(scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3]);
                    // Up wind for negative velocity
                    const double phim = interp_weno7(scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                };
            };
        };
    };

    return;
}

void weno_ninth_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){


    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sp2 = 2 * sp1;
    const long sp3 = 3 * sp1;
    const long sp4 = 4 * sp1;
    const long sp5 = 5 * sp1;
    const long sm1 = -sp1 ;
    const long sm2 = -2*sp1;
    const long sm3 = -3*sp1;
    const long sm4 = -4*sp1;



    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno9(scalar[ijk+sm4],scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4]);
                    // Up wind for negative velocity
                    const double phim = interp_weno9(scalar[ijk+sp5],scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2],scalar[ijk+sm3]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                };
            };
        };
    }
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno9(scalar[ijk+sm4],scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4]);
                    // Up wind for negative velocity
                    const double phim = interp_weno9(scalar[ijk+sp5],scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2],scalar[ijk+sm3]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                };
            };
        };
    };

    return;
}

void weno_eleventh_order(const struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,const double* restrict velocity, const double* restrict scalar, double* restrict flux, int d){


    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;

    const long imax = dims->nlg[0]-1;
    const long jmax = dims->nlg[1]-1;
    const long kmax = dims->nlg[2]-1;


    const long stencil[3] = {istride,jstride,1};
    const long sp1 = stencil[d];
    const long sp2 = 2 * sp1;
    const long sp3 = 3 * sp1;
    const long sp4 = 4 * sp1;
    const long sp5 = 5 * sp1;
    const long sp6 = 6 * sp1;
    const long sm1 = -sp1 ;
    const long sm2 = -2*sp1;
    const long sm3 = -3*sp1;
    const long sm4 = -4*sp1;
    const long sm5 = -5*sp1;



    if(d==2){
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno11(scalar[ijk+sm5],scalar[ijk+sm4],scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4],scalar[ijk+sp5]);
                    // Up wind for negative velocity
                    const double phim = interp_weno11(scalar[ijk+sp6],scalar[ijk+sp5],scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2],scalar[ijk+sm3],scalar[ijk+sm4]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0[k];
                };
            };
        };
    }
    else{
        for(long i=imin;i<imax;i++){
            const long ishift = i*istride ;
            for(long j=jmin;j<jmax;j++){
                const long jshift = j*jstride;
                for(long k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k ;
                    //Upwind for positive velocity
                    const double phip = interp_weno11(scalar[ijk+sm5],scalar[ijk+sm4],scalar[ijk+sm3],scalar[ijk+sm2],scalar[ijk+sm1],scalar[ijk],
                                            scalar[ijk+sp1],scalar[ijk+sp2],scalar[ijk+sp3],scalar[ijk+sp4],scalar[ijk+sp5]);
                    // Up wind for negative velocity
                    const double phim = interp_weno11(scalar[ijk+sp6],scalar[ijk+sp5],scalar[ijk+sp4],scalar[ijk+sp3],scalar[ijk+sp2],scalar[ijk+sp1],
                                            scalar[ijk],scalar[ijk+sm1],scalar[ijk+sm2],scalar[ijk+sm3],scalar[ijk+sm4]);
                    flux[ijk] =  (0.5*(velocity[ijk]+fabs(velocity[ijk]))*phip + 0.5*(velocity[ijk]-fabs(velocity[ijk]))*phim)*rho0_half[k];
                };
            };
        };
    };

    return;
}

void compute_advective_fluxes(struct DimStruct *dims, double* restrict rho0, double* rho0_half ,double* restrict velocity, double* restrict scalar,
                                double* restrict flux, int d, int scheme){

    switch(scheme){
        case 1:
            upwind_first(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 2:
            second_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 3:
            weno_third_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 4:
            fourth_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 5:
            weno_fifth_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 6:
            sixth_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 7:
            weno_seventh_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 8:
            eighth_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 9:
            weno_ninth_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
        case 11:
            weno_eleventh_order(dims, rho0, rho0_half, velocity, scalar, flux, d);
            break;
    };


};

void compute_flux_divergence(const struct DimStruct *dims, double* restrict alpha0, double* restrict alpha0_half, const double* restrict flux, double* restrict tendency, const double dx, int d){

    long i,j,k;
    const long imin = 1;
    const long jmin = 1;
    const long kmin = 1;

    const long imax = dims->nlg[0]-2;
    const long jmax = dims->nlg[1]-2;
    const long kmax = dims->nlg[2]-2;

    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];

    long sm1 ;
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
            const long ishift = i*istride ;
                for(j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                        for(k=kmin;k<kmax;k++){
                            const long ijk = ishift + jshift + k;
                            tendency[ijk] = tendency[ijk] - alpha0[k]*(flux[ijk+sm1] - flux[ijk])*dxi;
                        };
                 };
         };
     } // end if
     else{
        for(i=imin;i<imax;i++){
            const long ishift = i*istride ;
                for(j=jmin;j<jmax;j++){
                    const long jshift = j*jstride;
                        for(k=kmin;k<kmax;k++){
                            const long ijk = ishift + jshift + k;
                            tendency[ijk] = tendency[ijk] - (flux[ijk+sm1] - flux[ijk])*dxi;
                        };
                 };
         };

     }//end else

    return;
};