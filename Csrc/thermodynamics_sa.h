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
        double lam_1 = lam_fp(T_1);         // lam_fp gives the liquid fraction for mixed-phase clouds (fraction of supercooled liquid)
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
    /*
    Use saturation adjustment scheme to compute temperature T and ql given s and qt.
    :param p0: pressure [Pa]
    :param s: entropy  [K]
    :param qt:  total water specific humidity
    :return: T, ql, qv, qi

    mixed phase cloud: supercooled liquid water co-exists with ice below the freezing temperature (T_i < T < T_f)
    T_i = 233 K (homogeneous nucleation temperature)
    T_f = 273.15K (freezing point)
    lam_fp(T): gives the liquid fraction for mixed-phase clouds (fraction of supercooled liquid)
    l_fp(T,lam_fp(T)) = lam_fp*L_v + (1-lam_fp)*L_s: effective specific latent heat in mixed-phase

    Functions from Microphyiscs.pxd:
        liquid fraction:
            lam_fp(T) = 1.0 (lambda_constant)
        Latent Heat:
            L_fp(T, lambda(T)) = (2500.8 - 2.36 * TC + 0.0016 * TC**2 - 0.00006 * TC**3) * 1000.0
                with: TC = T - 273.15

    Functions from thermodynamic_functions.h:
        pv_c = p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv)

    Definitions (c.f. Pressel et al., 2016):
        saturation vapor pressure: pv_star(T)
            --> from Clausius Clapeyron (Lookup table for integration)
        saturation specific humidity: qv_star(p,qt,pv_star)
            --> ideal gas law; defined in Csrc/thermodynamics.h
        saturation excess: sigma = qt - qv_star
    */

//    printf("doing saturation adjustment (eos_c)\n");
    *qv = qt;
    *ql = 0.0;
    *qi = 0.0;
    double pv_1 = pv_c(p0,qt,qt );
    double pd_1 = p0 - pv_1;
    double T_1 = temperature_no_ql(pd_1,pv_1,s,qt);
    double pv_star_1 = lookup(LT, T_1);
    double qv_star_1 = qv_star_c(p0,qt,pv_star_1);
    // __
    int nan_T2 = 0;
    int nan_Tn = 0;
    double val_T1 = 0.0;
    double val_T2 = 0.0;
    double val_pd1 = 0.0;
    double val_pv1 = 0.0;
    // __

//    printf("eos_c: qt = %f, qv_star_1 = %f, qv = %f\n", qt, qv_star_1, *qv);        // in initialisation: qt > qv_star_1 (qt ~ 10*qv_star_1)
    // If not saturated
//    if(qt <= qv_star_1){
    if(qt <= qv_star_1 || qv_star_1 < 0.0){
//        printf("eos_c: not saturated\n");
        *T = T_1;
        return;
    }
    else{
        //printf("eos_c: saturated\n");
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
//        // the following definitions are necessary if while-loop below commented out
//        double pv_star_2 = lookup(LT, T_2);
//        if (pv_star_2>p0){
//            T_2 = 350.0;
//            pv_star_2 = lookup(LT, T_2);
//        }
        if (T_2 > 350.0){
            val_T2 = T_2;
            T_2 = 350.0;
        }
        if(isnan(T_2)){nan_T2 = 1;}
        double pv_star_2 = lookup(LT, T_2);
        int count = 0;
        // the following definitions are necessary if while-loop below commented out
        /*double pv_star_2 = lookup(LT, T_2);
        qv_star_2 = qv_star_c(p0,qt,pv_star_2);
        sigma_2 = qt - qv_star_2;
        lam_2 = lam_fp(T_2);*/
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
            count ++;
            if(isnan(T_n)){nan_Tn = 1;val_T1=T_1;val_pv1=pv_1;val_pd1=pd_1;}
        } while(delta_T >= 1.0e-3 || sigma_2 < 0.0 );
//        } while((delta_T >= 1.0e-3 || sigma_2 < 0.0) && count < 2);*/
        *T  = T_2;
        *qv = qv_star_2;
        *ql = lam_2 * sigma_2;
        *qi = (1.0 - lam_2) * sigma_2;
        // __
//        printf("eos_c iterations: count = %d\n",count);
//        printf("ql = %f\n", *ql);
        if(nan_T2==1){
            printf("nan_T2: %d\n", nan_T2);
            printf("T_1: %f\n", val_T1);
            printf("pv_1: %f\n", val_pv1);
            printf("pd_1: %f\n", val_pd1);
        }
        if(nan_Tn==1){
            printf("nan_Tn: %d\n", nan_Tn);
        }
        if(val_T2>0.0){
            printf("T2>350: %f\n",val_T2);
        }


        // __
        return;
    }
}



//void eos_update(struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
//    double* restrict p0, double* restrict s, double* restrict qt, double* restrict T,
//    double* restrict qv, double* restrict ql, double* restrict qi, double* restrict alpha ){
void eos_update(struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double* restrict p0, double* restrict s, double* restrict qt, double* restrict T,
    double* restrict qv, double* restrict ql, double* restrict qi, double* restrict alpha, int* n_nan ){
//    printf("eos_update\n");
    ssize_t i,j,k;
    // __
    int i_,j_,k_;
    int ijk_ = 1.0;
    double T_ = 0.0;
    // __
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
//                    const int ijk = ishift + jshift + k;
                    eos_c(LT, lam_fp, L_fp, p0[k], s[ijk],qt[ijk],&T[ijk],&qv[ijk],&ql[ijk],&qi[ijk]);
                    alpha[ijk] = alpha_c(p0[k], T[ijk],qt[ijk],qv[ijk]);
//                    printf("ASCII value = %d\n", ijk_);     // test print statement
//                    if(isnan(alpha[ijk])){
//                        ijk_ = 1;
//                        ijk_ = ijk;
//                        printf("!?! alpha nan in eos_update, T= %i \n",1);
////                        printf("!?! alpha nan in eos_update, T= %f!!!\n",T[ijk]);
////                        printf("problem decomposition advecting: count = %d\n",ok_ing);
//                    }
//                        ijk_ = ijk;
//                        printf("!?! alpha nan in eos_update, at: %d, T= %f!!!\n",ijk_,T[ijk]);}
//                    else{printf("!?! no nan, T= %f!!!\n",T[ijk]);}
//                  if(isnan(T[ijk])){
////                     ijk_ = ijk;
////                     T_ = T[ijk];
//                     printf("T is nan");
////                     printf("ijk=%d",ijk_);
//                     n_nan++;}
                } // End k loop
            } // End j loop
        } // End i loop
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
    // __
    int ijk_;
    // __
    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
            for (k=kmin;k<kmax;k++){
                const ssize_t ijk = ishift + jshift + k;
                if(isnan(alpha[ijk])){
                    ijk_ = ijk;
                    printf("!?! alpha is nan at: %d!!!\n",ijk_);}
                if(isnan(alpha0[k])){
                    ijk_ = k;
                    printf("!?! alpha0 is nan at: k = %d!!!\n",ijk_);}
//                else{printf("!?! no nan in eos\n");}
                buoyancy[ijk] = buoyancy_c(alpha0[k],alpha[ijk]);
                if(isnan(buoyancy[ijk])){
                    ijk_ = ijk;
                    printf("!!! buoyancy is nan at: %d!!!\n",ijk_);}
                if(isnan(alpha[ijk])){
                    ijk_ = ijk;
                    printf("!!! alpha is nan at: %d!!!\n",ijk_);}
                if(isnan(alpha0[k])){
                    ijk_ = k;
                    printf("!!! alpha is nan at: k = %d!!!\n",ijk_);}
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
