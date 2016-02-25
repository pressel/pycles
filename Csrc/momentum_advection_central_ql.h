#pragma once
#include "grid.h"
#include "advection_interpolation.h"
#include "flux_divergence.h"
#include<stdio.h>

// QUESTIONS / NOTES:
// need to be able to output eddy_flux???
// how to calculate horizontal mean???
// line 36 pp.: sp1_ed = stencil[d_advecting]; sp1_ing = stencil[d_advected]; --> confused about 'ed' vs. 'ing'

// dimension of flux, i.e. advection of which momentum in which direction:
//      --> second_order_m is called in MomentumAdvection.pyx (by calling compute_advective_tendencies_m), for a certain flux (advected and advecting velocities defined)

// IDEA:
// (1) interpolate velocity fields (dist. btw. advecting = advected vs. advecting != advected)
// (2) average interpolated velocity fields:
//          Pa.HorizontalMean(Gr, &DV.values[var_shift]), used e.g. in AuxiliaryStatistics
// (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
// (4) compute mean eddy flux
// (5) compute QL flux: flux = flux - eddy_flux + mean_eddy_flux

void second_order_m_ql(struct DimStruct *dims, double* restrict rho0, double* restrict rho0_half,
    double* restrict alpha0, double* restrict alpha0_half,
    double* restrict vel_advected, double* restrict vel_advecting,
    double* restrict tendency, ssize_t d_advected, ssize_t d_advecting){

        // Dynamically allocate flux array
        double *flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);     // malloc allocates size of uninitialized storage; in this case allocates memory for (nlg[0]+nlg[1]+nlg[2])*sizeof(double))
        double *eddy_flux = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *mean_eddy_flux = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_int_ed = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_int_ing = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        // double *vel_fluc = (double *)malloc(sizeof(double)*dims->nlg[0] * dims->nlg[1] * dims->nlg[2]);
        double *vel_mean_ed = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *vel_mean_ing = (double *)malloc(sizeof(double) * dims->nlg[2]);

        const ssize_t istride = dims->nlg[1] * dims->nlg[2];        // istride = "jump for getting to next i index"
        // ssize_t: predefined type of signed integer (= signed integral type) (eqvuivalent to Py_ssize_t in python, cython)
        const ssize_t jstride = dims->nlg[2];                       // jstride = "jump for getting to next j index"

        const ssize_t imin = 0;
        const ssize_t jmin = 0;
        const ssize_t kmin = 0;

        const ssize_t imax = dims->nlg[0]-1;
        const ssize_t jmax = dims->nlg[1]-1;
        const ssize_t kmax = dims->nlg[2]-1;

        const ssize_t stencil[3] = {istride,jstride,1};             // array, containing 3 elements
        // ??? ed vs ing
        const ssize_t sp1_ed = stencil[d_advecting];
        const ssize_t sp1_ing = stencil[d_advected];                // d_advecting, d_advected: given as parameters to this function


        // (1) interpolate velocity fields
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    vel_int_ing[ijk] = interp_2(vel_advecting[ijk],vel_advecting[ijk+sp1_ing]);
                    if (d_advected != d_advecting)
                        vel_int_ed[ijk] = interp_2(vel_advected[ijk],vel_advected[ijk+sp1_ed]);
                    else
                        vel_int_ed[ijk] = vel_int_ing[ijk];
                }
            }
        }


        // (2) average interpolated velocity fields
        // ??? call function per k or globally?
        for(ssize_t k=kmin;k<kmax;k++){
            //vel_mean_ing = mean(vel_int_advecting)
            if (d_advected != d_advecting)
                //vel_mean_ed = mean(vel_int_advected)
                vel_mean_ed[k] = 1;
            else
                vel_mean_ed[k] = vel_mean_ing[k];
        }


        // (3) compute eddy flux: (vel - mean_vel)**2 AND compute total flux
        if (d_advected != 2 && d_advecting !=2){                    // exclude w.u, w.v, w.w, u.w, v.w (advection by or of vertical velocity)
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        // ... to be modified according to QL ...
                        // vel_fluc = vel_int - vel_mean
                        // eddy_flux[ijk] = vel_fluc[ijk]*vel_fluc[ijk]*rho0_half[k];       // need to be able to output eddy_flux???
                        eddy_flux[ijk] = (vel_int_ing[ijk] - vel_mean_ing[k]) * (vel_int_ed[ijk] - vel_mean_ed[k]) * rho0_half[k];
                        flux[ijk] = (vel_int_ing[ijk] * vel_int_ed[ijk]) * rho0_half[k];
                        // ... to be modified according to QL ...
                    }
                }
            }
        }
        else if(d_advected == 2 && d_advecting == 2){               // w.w
            for(ssize_t i=imin;i<imax;i++){
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        // ... to be modified according to QL ...
                        // vel_fluc = vel_int - vel_mean
                        // eddy_flux[ijk] = vel_fluc[ijk]*vel_fluc[ijk]*rho0_half[k];       // need to be able to output eddy_flux???
                        eddy_flux[ijk] = (vel_int_ing[ijk] - vel_mean_ing[k]) * (vel_int_ed[ijk] - vel_mean_ed[k]) * rho0_half[k];
                        flux[ijk] = (vel_int_ing[ijk] * vel_int_ed[ijk]) * rho0_half[k];
                        // ... to be modified according to QL ..
                    }
                }
            }
        }
        else{
            for(ssize_t i=imin;i<imax;i++){                         // u.w, v.w, w.u, w.v
                const ssize_t ishift = i*istride;
                for(ssize_t j=jmin;j<jmax;j++){
                    const ssize_t jshift = j*jstride;
                    for(ssize_t k=kmin;k<kmax;k++){
                        const ssize_t ijk = ishift + jshift + k;
                        // ... to be modified according to QL ...
                        // vel_fluc = vel_int - vel_mean
                        // eddy_flux[ijk] = vel_fluc[ijk]*vel_fluc[ijk]*rho0_half[k];       // need to be able to output eddy_flux???
                        eddy_flux[ijk] = (vel_int_ing[ijk] - vel_mean_ing[k]) * (vel_int_ed[ijk] - vel_mean_ed[k]) * rho0_half[k];
                        flux[ijk] = (vel_int_ing[ijk] * vel_int_ed[ijk]) * rho0_half[k];
                        // ... to be modified according to QL ..
                    }
                }
            }
        }


        // (4) compute mean eddy flux
        // ??? call function per k or globally?
        for(ssize_t k=kmin;k<kmax;k++){
            //mean_eddy_flux = mean(eddy_flux)
            mean_eddy_flux[k] = 1;
        }


        // (5) compute QL flux: flux = flux - eddy_flux + mean_eddy_flux
        for(ssize_t i=imin;i<imax;i++){
            const ssize_t ishift = i*istride;
            for(ssize_t j=jmin;j<jmax;j++){
                const ssize_t jshift = j*jstride;
                for(ssize_t k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    flux[ijk] = flux[ijk] - eddy_flux[ijk] + mean_eddy_flux[k];
                }
            }
        }


        momentum_flux_divergence(dims, alpha0, alpha0_half, flux,
                                tendency, d_advected, d_advecting);

        //Free dynamically allocated array
        free(flux);
        free(vel_int_ing);
        free(vel_int_ed);
        free(vel_mean_ed);
        free(vel_mean_ing);
        return;
    }
