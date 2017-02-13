#pragma once
#include "parameters.h"
#include "grid.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "lookup.h"
#include "entropies.h"
#include <stdio.h>

inline float  temperature_no_ql(float  pd, float  pv, float  s, float  qt){
    return T_tilde * exp((s -
                            (1.0-qt)*(sd_tilde - Rd * log(pd/p_tilde))
                            - qt * (sv_tilde - Rv * log(pv/p_tilde)))
                            /((1.0-qt)*cpd + qt * cpv));
}

void eos_c(struct LookupStruct *LT, float  (*lam_fp)(float ), float  (*L_fp)(float , float ),
                    const float  p0, const float  s, const float  qt, float * T, float * qv, float * ql, float  *qi){
    *qv = qt;
    *ql = 0.0;
    *qi = 0.0;
    float  pv_1 = pv_c(p0,qt,qt );
    float  pd_1 = p0 - pv_1;
    float  T_1 = temperature_no_ql(pd_1,pv_1,s,qt);
    float  pv_star_1 = lookup(LT, T_1);
    float  qv_star_1 = qv_star_c(p0,qt,pv_star_1);

    /// If not saturated
    if(qt <= qv_star_1){
        *T = T_1;
        return;
    }
    else{
        float  sigma_1 = qt - qv_star_1;
        float  lam_1 = lam_fp(T_1);
        float  L_1 = L_fp(T_1,lam_1);
        float  s_1 = sd_c(pd_1,T_1) * (1.0 - qt) + sv_c(pv_1,T_1) * qt + sc_c(L_1,T_1)*sigma_1;
        float  f_1 = s - s_1;
        float  T_2 = T_1 + sigma_1 * L_1 /((1.0 - qt)*cpd + qv_star_1 * cpv);
        float  delta_T  = fabs(T_2 - T_1);
        float  qv_star_2;
        float  sigma_2;
        float  lam_2;
        do{
            float  pv_star_2 = lookup(LT, T_2);
            qv_star_2 = qv_star_c(p0,qt,pv_star_2);
            float  pv_2 = pv_c(p0,qt,qv_star_2);
            float  pd_2 = p0 - pv_2;
            sigma_2 = qt - qv_star_2;
            lam_2 = lam_fp(T_2);
            float  L_2 = L_fp(T_2,lam_2);
            float  s_2 = sd_c(pd_2,T_2) * (1.0 - qt) + sv_c(pv_2,T_2) * qt + sc_c(L_2,T_2)*sigma_2;
            float  f_2 = s - s_2;
            float  T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1);
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

void eos_update(struct DimStruct *dims, struct LookupStruct *LT, float  (*lam_fp)(float ), float  (*L_fp)(float , float ),
    float * restrict p0, float * restrict s, float * restrict qt, float * restrict T,
    float * restrict qv, float * restrict ql, float * restrict qi, float * restrict alpha ){

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

void buoyancy_update_sa(struct DimStruct *dims, float * restrict alpha0, float * restrict alpha, float * restrict buoyancy, float * restrict wt){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2]-1;

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
            for (k=kmin+1;k<kmax-2;k++){
                const ssize_t ijk = ishift + jshift + k;
                wt[ijk] = wt[ijk] + interp_2(buoyancy[ijk],buoyancy[ijk+1]);
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}

void bvf_sa(struct DimStruct *dims, struct LookupStruct *LT, float  (*lam_fp)(float ), float  (*L_fp)(float , float ), float * restrict p0, float * restrict T, float * restrict qt, float * restrict qv, float * restrict theta_rho,float * restrict bvf){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 1;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2]-1;
    const float  dzi = 1.0/dims->dx[2];

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
                    float  Lv=L_fp(T[ijk],lam_fp(T[ijk]));
                    float  pv_star = lookup(LT,T[ijk]);
                    float  rsl = eps_v*pv_star/(p0[k]-pv_star);
                    float  gamma_w = g/cpd*(1.0/(1.0-qt[ijk]))*(1.0+Lv*rsl/(Rd*T[ijk]))/(cpm_c(qt[ijk])/cpd + Lv*Lv*(eps_v+rsl)*rsl/(cpd*Rd*T[ijk]*T[ijk]));
                    float  dTdz=(interp_2(T[ijk],T[ijk+1])-interp_2(T[ijk-1],T[ijk]))*dzi;
                    float  dqtdz = (interp_2(qt[ijk],qt[ijk+1])-interp_2(qt[ijk-1],qt[ijk]))*dzi;
                    bvf[ijk] = g/T[ijk]*(dTdz+gamma_w)*(1.0 + Lv*rsl/(Rd*T[ijk]))-dqtdz/(1.0-qt[ijk]);
                }  // End if
                else{
                    //moist subsaturated
                    bvf[ijk] = g/theta_rho[ijk]*(interp_2(theta_rho[ijk],theta_rho[ijk+1])-interp_2(theta_rho[ijk-1],theta_rho[ijk]))*dzi;
                } // End else
            } // End k loop
        } // End j loop
    } // End i loop
    return;
}


void thetali_update(struct DimStruct *dims, float  (*lam_fp)(float ), float  (*L_fp)(float , float ), float * restrict p0, float * restrict T, float * restrict qt, float * restrict ql, float * restrict qi, float * restrict thetali){

    ssize_t i,j,k;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];
    const float  dzi = 1.0/dims->dx[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                float  Lv=L_fp(T[ijk],lam_fp(T[ijk]));
                thetali[ijk] =  thetali_c(p0[k], T[ijk], qt[ijk], ql[ijk], qi[ijk], Lv);
            } // End k loop
        } // End j loop
    } // End i loop

    return;
}

void clip_qt(struct DimStruct *dims, float * restrict qt, float  clip_value){
    size_t i;
    const size_t npg = dims->npg;
    for (i=0; i<npg; i++){
        qt[i] = fmax(qt[i], clip_value);
    }
    return;
}