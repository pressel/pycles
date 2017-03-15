#pragma once
#include "grid.h"
#include "advection_interpolation.h"


void compute_velocity_gradient(const struct DimStruct *dims, double* restrict v, double* restrict vgrad, const ssize_t d){

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
    const double dxi = dims->dxi[d];

    const double * imetl_half = dims -> imetl_half;

    if(d == 3){
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    vgrad[ijk] = (v[ijk + sp1] - v[ijk])*dxi * imetl_half[k+1];
                }
            }
        }
    }
    else{
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride ;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k ;
                    vgrad[ijk] = (v[ijk + sp1] - v[ijk])*dxi ;
                }
            }
        }
    }
    return;
}


void compute_strain_rate(const struct DimStruct *dims, double* restrict vgrad, double* restrict strain_rate){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

    const ssize_t imax = dims->nlg[0]-1;
    const ssize_t jmax = dims->nlg[1]-1;
    const ssize_t kmax = dims->nlg[2]-1;
    ssize_t count = 0;
    //Loop over the dimensions twice to compute the strain rate vector components
    for(ssize_t vi1=0;vi1<dims->dims;vi1++){
        for(ssize_t d=0;d<dims->dims;d++){
            const ssize_t shift_v1 = 3 * dims->npg * vi1 + dims->npg * d ;
            const ssize_t shift_v2 = 3 * dims->npg * d + dims-> npg * vi1;
            const ssize_t shift = count * dims->npg;
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride ;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k ;
                        strain_rate[shift + ijk] = 0.5 * (vgrad[shift_v1 + ijk] + vgrad[shift_v2 + ijk]) ;
                        }
                    }
                }
        count = count + 1 ;
        }
    }
    return;
}


void compute_strain_rate_mag(const struct DimStruct *dims, double* restrict strain_rate, double* restrict strain_rate_mag){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

    const ssize_t imax = dims->nlg[0]-1;
    const ssize_t jmax = dims->nlg[1]-1;
    const ssize_t kmax = dims->nlg[2]-1;
    ssize_t count = 0;

    const ssize_t stencil[3] = {istride,jstride,1};

    //Set all values of strain rate magnitude to zero
    for(ssize_t i=0; i<dims->npg; i++){
        strain_rate_mag[i] = 0.0;
    }

    //On-diagonal components
    for(ssize_t d=0;d<dims->dims;d++){
        const ssize_t shift_s = 4*d*dims->npg;
        for(ssize_t i=imin+1;i<imax;i++){
            const ssize_t ishift=i*istride;
            for(ssize_t j=jmin+1;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin+1;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    const ssize_t total_shift = shift_s + ijk - stencil[d];
                    strain_rate_mag[ijk] += strain_rate[total_shift]*strain_rate[total_shift];
                }
            }
        }
    }

    //Off-diagonal components
    //Here factor of 2 arises because we invoke symmetry of tensor
    for(ssize_t vi1=0;vi1<dims->dims-1;vi1++){
        for (ssize_t d=vi1;d<dims->dims;d++){
            const ssize_t shift_s = 3 * dims->npg * vi1 + dims->npg * d ;
            for(ssize_t i=imin+1;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin+1;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin+1;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        const ssize_t sp1 = shift_s + ijk;
                        const ssize_t sp2 = shift_s + ijk - stencil[vi1];
                        const ssize_t sp3 = shift_s + ijk - stencil[d] ;
                        const ssize_t sp4 = shift_s + ijk - stencil[d] - stencil[vi1] ;
                        const double s_interp = 0.25*(strain_rate[sp1]+strain_rate[sp2]+strain_rate[sp3]+strain_rate[sp4]);
                        strain_rate_mag[ijk] += 2.0*s_interp*s_interp;
                    }
                }
            }

        }
    }

    //Complete the calculation
    for(ssize_t i=0; i<dims->npg; i++){
        strain_rate_mag[i] = sqrt(2.0*strain_rate_mag[i]);
    }
    return;
}

void compute_wind_speed_angle(const struct DimStruct *dims, double* restrict u, double* restrict v,
 double* restrict wind_speed, double* restrict wind_angle, double u0, double v0){
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];

    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;

    const ssize_t imax = dims->nlg[0]-1;
    const ssize_t jmax = dims->nlg[1]-1;
    const ssize_t kmax = dims->nlg[2]-1;


    for(ssize_t i=imin;i<imax;i++){
        const ssize_t ishift = i*istride ;
        for(ssize_t j=jmin;j<jmax;j++){
            const ssize_t jshift = j*jstride;
            for(ssize_t k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k ;
                const double u_interp = interp_2(u[ijk + istride], u[ijk])+u0;
                const double v_interp = interp_2(v[ijk + jstride], v[ijk])+v0;
                wind_speed[ijk] = sqrt(u_interp * u_interp + v_interp * v_interp);
                wind_angle[ijk] = atan2(v_interp,u_interp+1.0e-20);
            }
        }
    }

    return;
}