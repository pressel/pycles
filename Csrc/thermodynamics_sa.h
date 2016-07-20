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

void eos_c_refstate(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                    const double p0, const double s, const double qt, double* T, double* qv, double* ql, double *qi){
//    printf("doing reference state saturation adjustment (eos_c_refstate)\n");
    *qv = qt;
    *ql = 0.0;
    *qi = 0.0;
    double pv_1 = pv_c(p0,qt,qt );
    double pd_1 = p0 - pv_1;
    double T_1 = temperature_no_ql(pd_1,pv_1,s,qt);
    double pv_star_1 = lookup(LT, T_1);
    double qv_star_1 = qv_star_c(p0,qt,pv_star_1);
//    printf("eos_c: qt = %f, qv_star_1 = %f, qv = %f\n", qt, qv_star_1, *qv);        // in initialisation: qt > qv_star_1 (qt ~ 10*qv_star_1)

    // If not saturated
    if(qt <= qv_star_1){
//        printf("eos_c: not saturated\n");
        *T = T_1;
        // __
//        printf("no iteration\n");
        // __
        return;
    }
    else{
//        printf("eos_c: saturated\n");
        double sigma_1 = qt - qv_star_1;
        double lam_1 = lam_fp(T_1);
        double L_1 = L_fp(T_1,lam_1);       // L_fp = ThermdynamicsSA.L_fp = Thermodynamics/LatentHeat.L_fp --> LatentHeat.L_fp = LatentHeat.L
        double s_1 = sd_c(pd_1,T_1) * (1.0 - qt) + sv_c(pv_1,T_1) * qt + sc_c(L_1,T_1)*sigma_1;
        double f_1 = s - s_1;
        double T_2 = T_1 + sigma_1 * L_1 /((1.0 - qt)*cpd + qv_star_1 * cpv);
        double delta_T  = fabs(T_2 - T_1);
        double qv_star_2;
        double sigma_2;
        double lam_2;
        // __
        int count = 0;
        /*double pv_star_2 = lookup(LT, T_2);
        qv_star_2 = qv_star_c(p0,qt,pv_star_2);
        sigma_2 = qt - qv_star_2;
        lam_2 = lam_fp(T_2);*/
        // __
        do{
//            printf("start loop\n");
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
            count ++;
        } while(delta_T >= 1.0e-3 || sigma_2 < 0.0 );
//        } while((delta_T >= 1.0e-3 || sigma_2 < 0.0) && count < 6);
        *T  = T_2;
        *qv = qv_star_2;
        *ql = lam_2 * sigma_2;
        *qi = (1.0 - lam_2) * sigma_2;
        // __
//        printf("eos_c iterations: count = %d\n",count);
//        printf("ql = %f\n", *ql);
        // __
        return;
    }
}


void eos_c(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                    const double p0, const double s, const double qt, double* T, double* qv, double* ql, double *qi){
//    printf("doing saturation adjustment (eos_c)\n");
    *qv = qt;
    *ql = 0.0;
    *qi = 0.0;
    double pv_1 = pv_c(p0,qt,qt );
    double pd_1 = p0 - pv_1;
    double T_1 = temperature_no_ql(pd_1,pv_1,s,qt);
    double pv_star_1 = lookup(LT, T_1);
    double qv_star_1 = qv_star_c(p0,qt,pv_star_1);

//    printf("eos_c: qt = %f, qv_star_1 = %f, qv = %f\n", qt, qv_star_1, *qv);        // in initialisation: qt > qv_star_1 (qt ~ 10*qv_star_1)
    // If not saturated
    if(qt <= qv_star_1){
//        printf("eos_c: not saturated\n");
        *T = T_1;
        // __
        /*printf("no iteration\n");
        double sigma_1 = qt - qv_star_1;
        double lam_1 = lam_fp(T_1);         // = lambda(T_1)?
        *ql = lam_1 * sigma_1;              // from paper: q_l,1 = lambda_1(T_1)*sigma_1
                                            // *ql = lam_2 * sigma_2;
        *qi = (1.0 - lam_1) * sigma_1;      // from paper: q_i,1 = [1-lambda(T_1)] * sigma_1
                                            // *qi = (1.0 - lam_2) * sigma_2;*/
        // __
        return;
    }
    else{
//        printf("eos_c: saturated\n");
        double sigma_1 = qt - qv_star_1;
        double lam_1 = lam_fp(T_1);
        double L_1 = L_fp(T_1,lam_1);       // L_fp = ThermdynamicsSA.L_fp = Thermodynamics/LatentHeat.L_fp --> LatentHeat.L_fp = LatentHeat.L
        double s_1 = sd_c(pd_1,T_1) * (1.0 - qt) + sv_c(pv_1,T_1) * qt + sc_c(L_1,T_1)*sigma_1;
        double f_1 = s - s_1;
        double T_2 = T_1 + sigma_1 * L_1 /((1.0 - qt)*cpd + qv_star_1 * cpv);
        double delta_T  = fabs(T_2 - T_1);
        double qv_star_2;
        double sigma_2;
        double lam_2;
        // __
        int count = 0;
        // __
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
        // __
//        printf("eos_c iterations: count = %d\n",count);
//        printf("ql = %f\n", *ql);
        // __
        return;
    }
}

void eos_update(struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double* restrict p0, double* restrict s, double* restrict qt, double* restrict T,
    double* restrict qv, double* restrict ql, double* restrict qi, double* restrict alpha ){
//    printf("eos_update\n");
    ssize_t i,j,k;
    // __
    int i_,j_,k_;
    // __
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];
    // __
//    int ijk_;
//    int a = 0;       // if defined as int, no nan shown in T; if double: Bus error
    // __

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    eos_c(LT, lam_fp, L_fp, p0[k], s[ijk],qt[ijk],&T[ijk],&qv[ijk],&ql[ijk],&qi[ijk]);
                    alpha[ijk] = alpha_c(p0[k], T[ijk],qt[ijk],qv[ijk]);

//                    a = T[ijk];
//                        printf("hoi\n");}
//                        j_ = j;
//                        i_ = i;
//                        k_ = k;
//                        printf("??? T is nan at: %d, (%d, %d, %d)\n",ijk_,i_,j_,k_);}
                } // End k loop
            } // End j loop
        } // End i loop
//    printf("number of nans = %d\n",a);
//    printf("finished eos_update\n");
/*

    for (i=0; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=0;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=0;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    a = alpha[ijk]; // ok
                    if(isnan(a)){            // Bus error: 10
                        ijk_ = ijk;
                        i_ = i;
                        j_ = j;
                        k_ = k;
                        printf("??? alpha is nan at: %d, (%d, %d, %d)\n",ijk_,i_,j_,k_);}
//                    a = qt[ijk];
//                    if(isnan(a)){
//                        ijk_ = ijk;
//                        i_ = i;
//                        j_ = j;
//                        k_ = k;
//                        printf("??? qt is nan at: %d, (%d, %d, %d)\n",ijk_,i_,j_,k_);}
//                    a = qv[ijk];
//                    if(isnan(a)){
//                        ijk_ = ijk;
//                        i_ = i;
//                        j_ = j;
//                        k_ = k;
//                        printf("??? qv is nan at: %d, (%d, %d, %d)\n",ijk_,i_,j_,k_);}
                    a = T[ijk];
                    if(isnan(a)){
                        ijk_ = ijk;
                        j_ = j;
                        i_ = i;
                        k_ = k;
                        printf("??? T is nan at: %d, (%d, %d, %d)\n",ijk_,i_,j_,k_);}
//                    a = p0[k];
//                    if(isnan(a)){
//                        ijk_ = k;
//                        i_ = i;
//                        k_ = k;
//                        printf("??? T is nan at: %d, (%d, %d, %d)\n",ijk_,i_,j_,k_);}
//                        printf("??? p0 is nan at: %d, (%d, %d, %d)\n",ijk_,i,j,k);}
                } // End k loop
            } // End j loop
        } // End i loop
*/

    return;

     /*
     Bus errors occur when your processor cannot even attempt the memory access requested. A bus error is trying to
     access memory that can't possibly be there. You've used an address that's meaningless to the system, or the wrong
     kind of address for that operation.
     Segmentation faults occur when accessing memory which does not belong to your process
     (accessing memory that you're not allowed to access), they are very common and are typically the result of:
        - using a pointer to something that was deallocated.
        - using an uninitialized hence bogus pointer.
        - using a null pointer.
        - overflowing a buffer. */
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
                    double dTdz=(interp_2(T[ijk],T[ijk+1])-interp_2(T[ijk-1],T[ijk]))*dzi;
                    double dqtdz = (interp_2(qt[ijk],qt[ijk+1])-interp_2(qt[ijk-1],qt[ijk]))*dzi;
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