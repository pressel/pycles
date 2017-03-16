#pragma once
#include "parameters.h"
#include "grid.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "lookup.h"
#include "entropies.h"
#include <stdio.h>

inline double temperature_no_ql(double pd, double pv, double s, double qt){
    return T_tilde * exp((s -
                            (1.0-qt)*(sd_tilde - Rd * log(pd/p_tilde))
                            - qt * (sv_tilde - Rv * log(pv/p_tilde)))
                            /((1.0-qt)*cpd + qt * cpv));
}

void eos_c(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                    const double p0, const double s, const double qt, double* T, double* qv, double* ql, double *qi){
    *qv = qt;
    *ql = 0.0;
    *qi = 0.0;
    double pv_1 = pv_c(p0,qt,qt );
    double pd_1 = p0 - pv_1;
    double T_1 = temperature_no_ql(pd_1,pv_1,s,qt);
    double pv_star_1 = lookup(LT, T_1);
    double qv_star_1 = qv_star_c(p0,qt,pv_star_1);

    /// If not saturated
    if(qt <= qv_star_1){
        *T = T_1;
        return;
    }
    else{
        double sigma_1 = qt - qv_star_1;
        double lam_1 = lam_fp(T_1);
        double L_1 = L_fp(T_1,lam_1);
        double s_1 = sd_c(pd_1,T_1) * (1.0 - qt) + sv_c(pv_1,T_1) * qt + sc_c(L_1,T_1)*sigma_1;
        double f_1 = s - s_1;
        double T_2 = T_1 + sigma_1 * L_1 /((1.0 - qt)*cpd + qv_star_1 * cpv);
        double delta_T  = fabs(T_2 - T_1);
        double qv_star_2;
        double sigma_2;
        double lam_2;
        do{
            double pv_star_2 = lookup(LT, T_2);
            qv_star_2 = qv_star_c(p0,qt,pv_star_2);
            double pv_2 = pv_c(p0,qt,qv_star_2);
            double pd_2 = p0 - pv_2;
            sigma_2 = qt - qv_star_2;
            lam_2 = lam_fp(T_2);
            double L_2 = L_fp(T_2,lam_2);
            double s_2 = sd_c(pd_2,T_2) * (1.0 - qt) + sv_c(pv_2,T_2) * qt + sc_c(L_2,T_2)*sigma_2;
            double f_2 = s - s_2;
            double T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1);
            T_1 = T_2;
            T_2 = T_n;
            f_1 = f_2;
            delta_T  = fabs(T_2 - T_1);
        } while(delta_T >= 1.0e-3 || sigma_2 < 0.0 );
        *T  = T_2;
        *qv = qv_star_2;
        *ql = lam_2 * sigma_2;
        *qi = (1.0 - lam_2) * sigma_2;
        return;
    }
}

void eos_update(struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double* restrict p0, double* restrict s, double* restrict qt, double* restrict T,
    double* restrict qv, double* restrict ql, double* restrict qi, double* restrict alpha ){

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
                    eos_c(LT, lam_fp, L_fp, p0[k], s[ijk],qt[ijk],&T[ijk],&qv[ijk],&ql[ijk],&qi[ijk]);
                    alpha[ijk] = alpha_c(p0[k], T[ijk],qt[ijk],qv[ijk]);

                } // End k loop
            } // End j loop
        } // End i loop
    return;
    }

void buoyancy_update_sa(struct DimStruct *dims, double* restrict alpha0, double* restrict alpha, double* restrict buoyancy, double* restrict wt){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2]-1;

    const double * metl = dims->metl;

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                buoyancy[ijk] = buoyancy_c(alpha0[k],alpha[ijk]) ;
            } // End k loop
        } // End j loop
    } // End i loop

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin+1;k<kmax-2;k++){
                const ssize_t ijk = ishift + jshift + k;
                wt[ijk] = wt[ijk] + interp_2(buoyancy[ijk],buoyancy[ijk+1]);
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}

void bvf_sa(struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double), double* restrict p0, double* restrict T, double* restrict qt, double* restrict qv, double* restrict theta_rho,double* restrict bvf){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 1;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2]-1;
    const double dzi = 1.0/dims->dx[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                theta_rho[ijk] = theta_rho_c(p0[k],T[ijk],qt[ijk],qv[ijk]);
            } // End k loop
        } // End j loop
    } // End i loop

    for(i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(k=kmin+1; k<kmax-1; k++){
                const ssize_t ijk = ishift + jshift + k;
                if(qv[ijk]<qt[ijk]){
                    //moist saturated
                    double Lv=L_fp(T[ijk],lam_fp(T[ijk]));
                    double pv_star = lookup(LT,T[ijk]);
                    double rsl = eps_v*pv_star/(p0[k]-pv_star);
                    double gamma_w = g/cpd*(1.0/(1.0-qt[ijk]))*(1.0+Lv*rsl/(Rd*T[ijk]))/(cpm_c(qt[ijk])/cpd + Lv*Lv*(eps_v+rsl)*rsl/(cpd*Rd*T[ijk]*T[ijk]));
                    double dTdz=(interp_2(T[ijk],T[ijk+1])-interp_2(T[ijk-1],T[ijk]))*dzi* dims->imetl_half[k];
                    double dqtdz = (interp_2(qt[ijk],qt[ijk+1])-interp_2(qt[ijk-1],qt[ijk]))*dzi* dims->imetl_half[k];
                    bvf[ijk] = g/T[ijk]*(dTdz+gamma_w)*(1.0 + Lv*rsl/(Rd*T[ijk]))-dqtdz/(1.0-qt[ijk]);
                }  // End if
                else{
                    //moist subsaturated
                    bvf[ijk] = g/theta_rho[ijk]*(interp_2(theta_rho[ijk],theta_rho[ijk+1])-interp_2(theta_rho[ijk-1],theta_rho[ijk]))*dzi * dims->imetl_half[k];
                } // End else
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}


void thetali_update(struct DimStruct *dims, double (*lam_fp)(double), double (*L_fp)(double, double), double* restrict p0, double* restrict T, double* restrict qt, double* restrict ql, double* restrict qi, double* restrict thetali){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
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
                double Lv=L_fp(T[ijk],lam_fp(T[ijk]));
                thetali[ijk] =  thetali_c(p0[k], T[ijk], qt[ijk], ql[ijk], qi[ijk], Lv);
            } // End k loop
        } // End j loop
    } // End i loop

    return;
}

void clip_qt(struct DimStruct *dims, double* restrict qt, double clip_value){
    size_t i;
    const size_t npg = dims->npg;
    for (i=0; i<npg; i++){
        qt[i] = fmax(qt[i], clip_value);
    }
    return;
}