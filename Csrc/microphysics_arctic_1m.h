#pragma once
#include "grid.h"
#include "lookup.h"
#include "parameters.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include <stdio.h>
#include <math.h>
#include "parameters_micro.h"
#include "advection_interpolation.h"
#include "microphysics.h"

double rho_liq = 1000.0;
double visc_air = 2.0e-5;
double lhf = 3.34e5;
double small = 1.0e-10;
double cvl = 4190.0;
double cvi = 2106.0;
double lf0= 3.34e5;

double vapor_diffusivity(const double temperature, const double p0){
    /*Straka 2009 (5.2)*/
    double val = 2.11e-5 * pow((temperature / 273.15), 1.94) * (p0 / 101325.0);
    return val;
};

double thermal_conductivity(const double temperature){
    /*Straka 2009 (5.33)*/
    double val = 2.591e-2 * pow((temperature / 296.0), 1.5) * (416.0 / (temperature - 120.0));
    return val;
};

double pv_star_ice_c(const double temperature){
    double t0 = 273.16;
    double a3i = 22.587;
    double a4i = -0.7;
    double es0 = 611.21;
    return es0 * exp(a3i * (temperature - t0)/(temperature - a4i));
};

double microphysics_g_arctic(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double temperature, double dvap, double kt){
    double lam = lam_fp(temperature);
    double L = L_fp(temperature,lam);
    double pv_sat = lookup(LT, temperature);
    /*Straka 2009 (6.13)*/
    double g_therm = 1.0/(Rv*temperature/dvap/pv_sat + L/kt/temperature * (L/Rv/temperature - 1.0));
    return g_therm;
};

double rain_dmean(double density, double qrain, double nrain){
    double wc = fmax(qrain * density, SMALL);
    double val = pow((wc*GSTAR_RAIN/A_RAIN/nrain), (1.0/(B_RAIN+1.0)));

    return val;
};

double snow_dmean(double density, double qsnow, double nsnow){
    double wc = fmax(qsnow * density, SMALL);
    double val = cbrt(wc*GSTAR_SNOW/A_SNOW/nsnow);

    return val;
};

double ice_dmean(double density, double qi, double ni){
    double wc = fmax(qi * density, SMALL);
    double val = pow((wc*GSTAR_ICE/A_ICE/ni), (1.0/(B_ICE+1.0)));

    return val;
};

double liquid_dmean(double density, double ql, double ccn){
    /* Martin et al. 1994 ???*/
    double varm6, ntot;
    double ccn_ = ccn*1.0e-6;

    if( ccn_ > 300.0 ){
        varm6 = 1.189633;
        ntot = (-2.1e-4*ccn_*ccn_ + 0.568*ccn_ - 27.9)*1.0e6;
    }
    else{
        varm6 = 1.055572;
        ntot = (-1.15e-3*ccn_*ccn_ + 0.963*ccn_ + 5.3)*1.0e6;
    };

    double lwc = fmax(ql * density, SMALL);
    double df = cbrt(6.0*lwc/(pi*DENSITY_LIQUID*ntot));

    return df;
};

double rain_lambda(double density, double qrain, double nrain){
    double wc = fmax(qrain * density, SMALL);
    double val = pow((A_RAIN*nrain*GB1_RAIN/wc), (1.0/(B_RAIN+1.0)));

    return val;
};

double snow_lambda(double density, double qsnow, double nsnow){
    double wc = fmax(qsnow * density, SMALL);
    double val = cbrt(A_SNOW*nsnow*GB1_SNOW/wc);
    /*Morrison et al. 2011 alternative formulation*/
    /*double val = 3.81e3*pow(wc, -0.147)*/
    return val;
};

double ice_lambda(double density, double qi, double ni){
    double wc = fmax(qi * density, SMALL);
    double val = pow((A_ICE*ni*GB1_ICE/wc), (1.0/(B_ICE+1.0)));

    return val;
};

void get_rain_n0(const struct DimStruct *dims, double* restrict density, double* restrict qrain, double* restrict nrain){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    const double b1 = 650.1466922699631;
    const double b2 = -1.222222222222222;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double rwc = fmax(qrain[ijk]*density[k], SMALL);
                double n0_rain = 1.0e7; //b1*pow(rwc, b2);
                double n0_max = n0_rain;//rwc*N_MAX_RAIN;
                double n0_min = n0_rain;//rwc*N_MIN_RAIN;

                nrain[ijk] = fmax(fmin(n0_rain,n0_max),n0_min);

            }
        }
    }

    return;

};

void get_snow_n0(const struct DimStruct *dims, double* restrict density, double* restrict qsnow, double* restrict nsnow){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    /*Morrison et al. 2011*/
    const double y1 = 5.62e7;
    const double y2 = 0.63;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double swc = fmax(qsnow[ijk]*density[k], SMALL);
                double n0_snow = 1.0e7; //y1*pow(swc*1000.0, y2);
                double n0_max = n0_snow;//swc*N_MAX_SNOW;
                double n0_min = n0_snow;//swc*N_MIN_SNOW;

                nsnow[ijk] = fmax(fmin(n0_snow,n0_max),n0_min);

            }
        }
    }

    return;

};

void autoconversion_rain(double density, double ccn, double ql, double qrain, double nrain,
                         double* qrain_tendency){
    /* Berry-Reinhardt 74 rain autoconversion model*/
    double l2, t2;
    const double ccn_ = ccn*1.0e-6;
    double varm6;

    if(ccn_ > 300.0){
        varm6 = 1.189633;
    }
    else{
        varm6 = 1.055572;
    }

    double lwc = ql*density;
    double df = liquid_dmean(density, ql, ccn);
    double db = df*varm6;

    if(db <= 15.0e-6){
        *qrain_tendency = 0.0;
    }
    else{
        l2 = 2.7e-2*(1.0e20/16.0*(db*db*db)*df - 0.4)*lwc;
        t2 = 3.72/(5.0e5*db - 7.5)/lwc;
        *qrain_tendency = fmax(l2,0.0)/fmax(t2,1.0e-20)/density;
    }

    return;
};

void autoconversion_snow(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                         double density, const double p0, double temperature, double qt,
                         double qi, double ni, double* qsnow_tendency){
    /* Harrington 1995 snow autoconversion model */
    double pv_star = lookup(LT, temperature);
    //double pv_star = pv_star_ice_c(temperature);
    double qv_star = qv_star_c(p0, qt, pv_star);
    //double satratio = qt_/qv_star;
    double satratio = (qt-qi)/qv_star;
    double db_ice = 125.0e-6;
    double gtherm, psi;
    double vapor_diff = vapor_diffusivity(temperature, p0);
    double therm_cond = thermal_conductivity(temperature);
    double ice_lam = ice_lambda(density, qi, ni);

    if( qi > 1.0e-10 && satratio > 1.0){
        //gtherm = 1.0e-7/(2.2*temperature/pv_star + 220.0/temperature);
        //gtherm = 1.0 / ( (Rv*temperature/vapor_diff/pv_star) + (8.028e12/therm_cond/Rv/(temperature*temperature)) );
        gtherm = microphysics_g_arctic(LT, lam_fp, L_fp, temperature, vapor_diff, therm_cond);
        psi = 4.0*pi*(satratio - 1.0)*gtherm;
        *qsnow_tendency = (psi*ni*exp(-ice_lam*db_ice)
                           *(db_ice*db_ice/3.0 + (1.0+ice_lam*db_ice)/(ice_lam*ice_lam))/density);
    }

    return;
};

void evaporation_rain(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                      double density, const double p0, double temperature,
                      double qt, double qrain, double nrain, double* qrain_tendency){
    double beta = 2.0;
    double pv_star = lookup(LT, temperature);
    double qv_star = qv_star_c(p0, qt, pv_star);
    double satratio = qt/qv_star;
    double vapor_diff = vapor_diffusivity(temperature, p0);
    double therm_cond = thermal_conductivity(temperature);
    double rain_diam = rain_dmean(density, qrain, nrain);
    double rain_vel = C_RAIN*pow(rain_diam, D_RAIN);
    double rain_lam = rain_lambda(density, qrain, nrain);

    double re, vent, gtherm;

    if( satratio < 1.0 && qrain > 1.0e-15){
        re = rain_diam*rain_vel/VISC_AIR;
        vent = 0.78 + 0.27*sqrt(re);
        //gtherm = 1.0e-7/(2.2*temperature/pv_star + 220.0/temperature);
        //gtherm = 1.0 / ( (Rv*temperature/vapor_diff/pv_star) + (8.028e12/therm_cond/Rv/(temperature*temperature)) );
        gtherm = microphysics_g_arctic(LT, lam_fp, L_fp, temperature, vapor_diff, therm_cond);
        *qrain_tendency = 4.0*pi/beta*(satratio - 1.0)*vent*gtherm*nrain/(rain_lam*rain_lam)/density;
    }

    return;
};

void evaporation_snow(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                        double density, double p0, double temperature, double qt,
                        double qsnow, double nsnow, double* qsnow_tendency){
    double beta = 3.0;
    double pv_star = lookup(LT, temperature);
    //double pv_star = pv_star_ice_c(temperature);
    double qv_star = qv_star_c(p0, qt, pv_star);
    double satratio = qt/qv_star;

    double vapor_diff = vapor_diffusivity(temperature, p0);
    double therm_cond = thermal_conductivity(temperature);
    double snow_diam = snow_dmean(density, qsnow, nsnow);
    double snow_vel = C_SNOW*pow(snow_diam, D_SNOW);
    double snow_lam = snow_lambda(density, qsnow, nsnow);

    double re = snow_diam*snow_vel/VISC_AIR;
    double vent = 0.65 + 0.39*sqrt(re);
    //double gtherm = 1.0e-7/(2.2*temperature/pv_star + 220.0/temperature);
    //double gtherm = 1.0 / ( (Rv*temperature/vapor_diff/pv_star) + (8.028e12/therm_cond/Rv/(temperature*temperature)) );

    if( qsnow > 1.0e-15 ){
        double gtherm = microphysics_g_arctic(LT, lam_fp, L_fp, temperature, vapor_diff, therm_cond);
        *qsnow_tendency = 4.0*pi/beta*(satratio - 1.0)*vent*gtherm*nsnow/(snow_lam*snow_lam)/density;
    }

    return;
};

void accretion_all(double density, double p0, double temperature, double ccn, double ql, double qi, double ni,
                   double qrain, double nrain, double qsnow, double nsnow,
                   double* ql_tendency, double* qi_tendency, double* qrain_tendency, double* qsnow_tendency){

    double factor_r = 0.0;
    double factor_s = 0.0;
    double piacr = 0.0;

    double e_ri = 1.0;
    double e_si = exp(9.0e-2*(temperature - 273.15));
    double e_rl = 0.85;
    double e_sl = 0.8;

    double liq_diam = liquid_dmean(density, ql, ccn);
    double snow_diam = snow_dmean(density, qsnow, nsnow);
    double rain_diam = rain_dmean(density, qrain, nrain);
    double rain_lam = rain_lambda(density, qrain, nrain);
    double snow_lam = snow_lambda(density, qsnow, nsnow);
    double ice_lam = ice_lambda(density, qi, ni);
    double rain_vel = C_RAIN*pow(rain_diam, D_RAIN);
    double snow_vel = C_SNOW*pow(snow_diam, D_SNOW);

    if( snow_diam < 150.0e-6 ){
        e_sl = 0.0;
    }
    else{
        if( liq_diam < 15.0e-6 ){
            e_sl = 0.0;
        }
        else if( liq_diam < 40.0e-6 ){
            e_sl = (liq_diam - 15.0e-6) * e_sl / 25.0e-6;
        }
    }

    if( qrain > SMALL ){
        factor_r = density*GD3_RAIN*nrain*pi*ALPHA_ACC_RAIN*C_RAIN*0.25*pow(rain_lam, (-D_RAIN - 3.0));
        if( qi > SMALL ){
            piacr = nrain*ni/ice_lam*e_ri*pi*0.25*A_RAIN*C_RAIN*GD6_RAIN*pow(rain_lam, (-D_RAIN - 6.0));
        }
    }

    if( qsnow > SMALL ){
        factor_s = density*GD3_SNOW*nsnow*pi*ALPHA_ACC_SNOW*C_SNOW*0.25*pow(snow_lam, (-D_SNOW-3.0));
    }

    double src_ri = -piacr/density;
    double src_rl = factor_r*e_rl*ql/density;
    double src_si = factor_s*e_si*qi/density;
    double rime_sl = factor_s*e_sl*ql/density;
    double src_sl = 0.0;

    if( temperature > 273.16 ){
        src_sl = -cvl/lf0*(temperature-273.16)*rime_sl;
        src_rl = src_rl + rime_sl - src_sl;
    }
    else{
        src_sl = rime_sl + (factor_r*e_ri*qi + piacr)/density;
    }

    /* Now precip-precip interactions */
    double src_r = 0.0;
    double src_s = 0.0;
    double dv, k_2s, k_2r;

    if( qrain > small && qsnow > small ){
        dv = fabs(rain_vel - snow_vel);
        k_2s = (30.0/(pow(rain_lam, 6.0)*snow_lam) + 12.0/(pow(rain_lam, 5.0)*pow(snow_lam, 2.0))
                + 3.0/(pow(rain_lam, 4.0)*pow(snow_lam, 3.0)));
        k_2r = (1.0/(pow(rain_lam, 3.0)*pow(snow_lam, 3.0)) + 3.0/(pow(rain_lam, 2.0)*pow(snow_lam, 4.0))
                + 6.0/(rain_lam*pow(snow_lam, 5.0)));
        if( temperature < 273.16 ){
            src_s = pi*dv*nsnow*nrain*A_RAIN*k_2s/density;
            src_r = -src_s;
        }
        else{
            src_r = pi*dv*nsnow*nrain*A_RAIN*k_2r/density;
            src_s = -src_r;
        }
    }

    *qrain_tendency = src_r + src_rl + src_ri;
    *qsnow_tendency = src_s + src_sl + src_si;
    *ql_tendency = -(src_rl + rime_sl);
    *qi_tendency = -(src_ri + src_si);

    return;
};

void melt_snow(double density, double temperature, double qsnow, double nsnow, double* qsnow_tendency){
    double ka = 2.43e-2;
    double snow_diam = snow_dmean(density, qsnow, nsnow);
    double snow_vel = C_SNOW*pow(snow_diam, D_SNOW);
    double snow_lam = snow_lambda(density, qsnow, nsnow);
    double fvent = 0.65 + 0.39*sqrt(snow_vel*snow_diam/VISC_AIR);

    if( temperature > 273.16 && qsnow > small ){
        *qsnow_tendency = -2.0*pi*nsnow*ka/lhf*(temperature - 273.16)*fvent/(snow_lam*snow_lam)/density;
    }
    return;
};

void microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
                             double (*L_fp)(double, double), double* restrict density, double* restrict p0,
                             double* restrict temperature,  double* restrict qt, double ccn, double n0_ice,
                             double* restrict ql, double* restrict qi, double* restrict qrain, double* restrict nrain,
                             double* restrict qsnow, double* restrict nsnow, double dt,
                             double* restrict qrain_tendency_micro, double* restrict qrain_tendency,
                             double* restrict qsnow_tendency_micro, double* restrict qsnow_tendency,
                             double* restrict precip_rate, double* restrict evap_rate){

    const double b1 = 650.1466922699631;
    const double b2 = -1.222222222222222;
    const double y1 = 5.62e7;
    const double y2 = 0.63;

    double iwc, ni;
    double qrain_tendency_aut=0.0, qrain_tendency_acc=0.0, qrain_tendency_evp=0.0;
    double qsnow_tendency_aut=0.0, qsnow_tendency_acc=0.0, qsnow_tendency_evp=0.0, qsnow_tendency_melt=0.0;
    double ql_tendency_acc=0.0, qi_tendency_acc=0.0;
    double ql_tendency_tmp=0.0, qi_tendency_tmp=0.0, qrain_tendency_tmp=0.0, qsnow_tendency_tmp=0.0;
    double qt_tmp, ql_tmp, qi_tmp, qrain_tmp, qsnow_tmp;

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;


                // First get number concentartion N_0 for micro species
                qi_tmp = fmax(qi[ijk], 0.0);
                iwc = fmax(qi_tmp * density[k], SMALL);
                ni = fmax(fmin(n0_ice, iwc*N_MAX_ICE),iwc*N_MIN_ICE);

                qrain_tmp = fmax(qrain[ijk],0.0); //clipping
                qsnow_tmp = fmax(qsnow[ijk],0.0); //clipping
                qt_tmp = qt[ijk];
                ql_tmp = fmax(ql[ijk],0.0);


                // Now do sub-timestepping
                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    iter_count += 1;
                    dt_ = dt - time_added;

                    if ((ql_tmp + qi_tmp) < SMALL && (qrain_tmp + qsnow_tmp) < SMALL)
                        break;

                    qsnow_tendency_aut = 0.0;
                    qsnow_tendency_acc = 0.0;
                    qsnow_tendency_evp = 0.0;
                    qsnow_tendency_melt = 0.0;

                    qrain_tendency_aut = 0.0;
                    qrain_tendency_acc = 0.0;
                    qrain_tendency_evp = 0.0;

                    ql_tendency_acc = 0.0;
                    qi_tendency_acc = 0.0;

                    autoconversion_rain(density[k], ccn, ql_tmp, qrain_tmp, nrain[ijk], &qrain_tendency_aut);
                    autoconversion_snow(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp,
                                        qi_tmp, ni, &qsnow_tendency_aut);
                    accretion_all(density[k], p0[k], temperature[ijk], ccn, ql_tmp, qi_tmp, ni,
                                 qrain_tmp, nrain[ijk], qsnow_tmp, nsnow[ijk],
                                  &ql_tendency_acc, &qi_tendency_acc, &qrain_tendency_acc, &qsnow_tendency_acc);

                    evaporation_rain(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qrain_tmp, nrain[ijk],
                                     &qrain_tendency_evp);
                    evaporation_snow(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qsnow_tmp,
                                     nsnow[ijk], &qsnow_tendency_evp);
                    melt_snow(density[k], temperature[ijk], qsnow_tmp, nsnow[ijk], &qsnow_tendency_melt);

                    qrain_tendency_tmp = qrain_tendency_aut + qrain_tendency_acc + qrain_tendency_evp - qsnow_tendency_melt;
                    qsnow_tendency_tmp = qsnow_tendency_aut + qsnow_tendency_acc + qsnow_tendency_evp + qsnow_tendency_melt;
                    ql_tendency_tmp = ql_tendency_acc - qrain_tendency_aut;
                    qi_tendency_tmp = qi_tendency_acc - qsnow_tendency_aut;

                    rate = 1.05 * qrain_tendency_tmp * dt_ / (-fmax(qrain_tmp, SMALL));
                    rate = fmax(1.05 * qsnow_tendency_tmp * dt_ /(-fmax(qsnow_tmp, SMALL)), rate);
                    rate = fmax(1.05 * ql_tendency_tmp * dt_ /(-fmax(ql_tmp, SMALL)), rate);
                    rate = fmax(1.05 * qi_tendency_tmp * dt_ /(-fmax(qi_tmp, SMALL)), rate);

                    if(rate > 1.0 && iter_count < MAX_ITER){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }

                    precip_rate[ijk] = -qrain_tendency_aut + ql_tendency_acc - qsnow_tendency_aut + qi_tendency_acc;
                    evap_rate[ijk] = qrain_tendency_evp + qsnow_tendency_evp;

                    //Integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    qi_tmp += qi_tendency_tmp * dt_;
                    qrain_tmp += qrain_tendency_tmp * dt_;
                    qsnow_tmp += qsnow_tendency_tmp * dt_;
                    qt_tmp += (precip_rate[ijk] - evap_rate[ijk]) * dt_;

                    qrain_tmp = fmax(qrain_tmp, 0.0);
                    qsnow_tmp = fmax(qsnow_tmp, 0.0);
                    ql_tmp = fmax(ql_tmp, 0.0);
                    qi_tmp = fmax(qi_tmp, 0.0);
                    qt_tmp = fmax(qt_tmp, 0.0);

                    time_added += dt_;
                    }while(time_added < dt && iter_count < MAX_ITER);

                qrain_tendency_micro[ijk] = (qrain_tmp - qrain[ijk])/dt;
                qrain_tendency[ijk] += qrain_tendency_micro[ijk];
                qsnow_tendency_micro[ijk] = (qsnow_tmp - qsnow[ijk])/dt;
                qsnow_tendency[ijk] += qsnow_tendency_micro[ijk];

            }
        }
    }


    return;
};

void sedimentation_velocity_rain(const struct DimStruct *dims, double* restrict density, double* restrict nrain,
                                    double* restrict qrain, double* restrict qrain_velocity){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double rain_lam = rain_lambda(density[k], qrain[ijk], nrain[ijk]);
                qrain_velocity[ijk] = -C_RAIN*GBD1_RAIN/GB1_RAIN/pow(rain_lam, D_RAIN);

            }
        }
    }


     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;

                qrain_velocity[ijk] = interp_2(qrain_velocity[ijk], qrain_velocity[ijk+1]) ;

            }
        }
    }


    return;

};

void sedimentation_velocity_snow(const struct DimStruct *dims, double* restrict density, double* restrict nsnow,
                                    double* restrict qsnow, double* restrict qsnow_velocity){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                double snow_lam = snow_lambda(density[k], qsnow[ijk], nsnow[ijk]);
                qsnow_velocity[ijk] = -C_SNOW*GBD1_SNOW/GB1_SNOW/pow(snow_lam, D_SNOW);

            }
        }
    }


     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;
                qsnow_velocity[ijk] = interp_2(qsnow_velocity[ijk], qsnow_velocity[ijk+1]) ;
            }
        }
    }


    return;

};

void qt_source_formation(const struct DimStruct *dims, double* restrict qt_tendency,
                         double* restrict precip_rate, double* restrict evap_rate){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                qt_tendency[ijk] += precip_rate[ijk] - evap_rate[ijk];
            }
        }
    }
    return;
};

double entropy_src_precipitation_c(const double p0, const double temperature, const double qt, const double qv, const double L, const double precip_rate){
    double pd = pd_c(p0, qt, qv);
    double pv = pv_c(p0, qt, qv);
    double sd = sd_c(pd, temperature);
    double sv = sv_c(pv, temperature);
    double sc = sc_c(L, temperature);

    return -(sd - sv - sc) * precip_rate;
};

double entropy_src_evaporation_c(const double p0, const double temperature, double Tw, const double qt, const double qv, const double L, const double evap_rate){
    double pd = pd_c(p0, qt, qv);
    double pv = pv_c(p0, qt, qv);
    double sd = sd_c(pd, temperature);
    double sv = sv_c(pv, Tw);
    double sc = sc_c(L, Tw);

    return -(sv + sc - sd) * evap_rate;
};

void entropy_source_formation(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
                              double (*L_fp)(double, double), double* restrict p0, double* restrict T,
                              double* restrict Twet, double* restrict qt, double* restrict qv,
                              double* restrict qrain_tendency, double* restrict qsnow_tendency,
                              double* restrict precip_rate, double* restrict evap_rate, double* restrict entropy_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    //entropy tendencies from formation or evaporation of precipitation
    //we use fact that P = d(qr)/dt > 0, E =  d(qr)/dt < 0
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;




            }
        }
    }
    return;
};

void entropy_source_heating_rain(const struct DimStruct *dims, double* restrict temperature, double* restrict Twet, double* restrict qrain,
                               double* restrict w_qrain, double* restrict w,  double* restrict entropy_tendency){


    //derivative of Twet is upwinded

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];


    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                entropy_tendency[ijk]+= qrain[ijk]*(fabs(w_qrain[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk])* dzi/temperature[ijk];
            }
        }
    }

    return;

};

void entropy_source_heating_snow(const struct DimStruct *dims, double* restrict temperature, double* restrict Twet, double* restrict qsnow,
                               double* restrict w_qsnow, double* restrict w,  double* restrict entropy_tendency){


    //derivative of Twet is upwinded

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];


    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                entropy_tendency[ijk]+= qsnow[ijk]*(fabs(w_qsnow[ijk]) - w[ijk]) * ci * (Twet[ijk+1] - Twet[ijk])* dzi/temperature[ijk];
            }
        }
    }

    return;

};


void entropy_source_drag(const struct DimStruct *dims, double* restrict temperature,  double* restrict qprec,
                            double* restrict w_qprec, double* restrict entropy_tendency){



    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;


    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                entropy_tendency[ijk]+= g * qprec[ijk]* fabs(w_qprec[ijk])/ temperature[ijk];
            }
        }
    }

    return;

};

void get_virtual_potential_temperature(const struct DimStruct *dims, double* restrict p0, double* restrict temperature, double* restrict qv,
                                       double* restrict ql, double* restrict qi, double* restrict thetav){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;
                thetav[ijk] = theta_c(p0[k], temperature[ijk]) * (1.0 + 0.608 * qv[ijk] - ql[ijk] - qi[ijk]);
            }
        }
    }

    return;


};

///==========================To facilitate output=============================

void autoconversion_snow_wrapper(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
                                 double (*L_fp)(double, double), double n0_ice, double* restrict density,
                                 double* restrict p0, double* restrict temperature,
                                 double* restrict qt, double* restrict qi, double* restrict qsnow_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double qi_tmp = fmax(qi[ijk], 0.0);
                double iwc = fmax(qi_tmp * density[k], SMALL);
                double ni = fmax(fmin(n0_ice, iwc*N_MAX_ICE),iwc*N_MIN_ICE);
                double qt_tmp = qt[ijk];

                autoconversion_snow(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp,
                                    qi_tmp, ni, &qsnow_tendency[ijk]);

            }
        }
    }
    return;
};

void evaporation_snow_wrapper(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
                              double (*L_fp)(double, double), double* restrict density, double* restrict p0,
                              double* restrict temperature, double* restrict qt, double* restrict qsnow,
                              double* restrict nsnow, double* restrict qsnow_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double qsnow_tmp = fmax(qsnow[ijk],0.0); //clipping
                double qt_tmp = qt[ijk];

                evaporation_snow(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qsnow_tmp,
                                 nsnow[ijk], &qsnow_tendency[ijk]);

            }
        }
    }
    return;

};


void autoconversion_rain_wrapper(const struct DimStruct *dims, double* restrict density, double ccn,
                                 double* restrict ql, double* restrict qrain, double* restrict nrain,
                                 double* restrict qrain_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double ql_tmp = fmax(ql[ijk], 0.0);
                double qrain_tmp = fmax(qrain[ijk], 0.0);

                autoconversion_rain(density[k], ccn, ql_tmp, qrain_tmp, nrain[ijk], &qrain_tendency[ijk]);

            }
        }
    }
    return;
};

void evaporation_rain_wrapper(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double),
                              double (*L_fp)(double, double), double* restrict density, double* restrict p0,
                              double* restrict temperature, double* restrict qt, double* restrict qrain,
                              double* restrict nrain, double* restrict qrain_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double qrain_tmp = fmax(qrain[ijk],0.0); //clipping
                double qt_tmp = qt[ijk];

                evaporation_rain(LT, lam_fp, L_fp, density[k], p0[k], temperature[ijk], qt_tmp, qrain_tmp,
                                 nrain[ijk], &qrain_tendency[ijk]);
            }
        }
    }
    return;

};

void accretion_all_wrapper(const struct DimStruct *dims, double* density, double* p0, double* temperature, double n0_ice,
                           double ccn, double* ql, double* qi, double* qrain, double* nrain, double* qsnow, double* nsnow,
                           double* ql_tendency, double* qi_tendency, double* qrain_tendency, double* qsnow_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double qi_tmp = fmax(qi[ijk], 0.0);
                double iwc = fmax(qi_tmp * density[k], SMALL);
                double ni = fmax(fmin(n0_ice, iwc*N_MAX_ICE),iwc*N_MIN_ICE);

                double qrain_tmp = fmax(qrain[ijk],0.0); //clipping
                double qsnow_tmp = fmax(qsnow[ijk],0.0); //clipping
                double ql_tmp = fmax(ql[ijk],0.0);

                accretion_all(density[k], p0[k], temperature[ijk], ccn, ql_tmp, qi_tmp, ni, qrain_tmp, nrain[ijk],
                              qsnow_tmp, nsnow[ijk], &ql_tendency[ijk], &qi_tendency[ijk], &qrain_tendency[ijk],
                              &qsnow_tendency[ijk]);

            }
        }
    }
    return;

};

void melt_snow_wrapper(const struct DimStruct *dims, double* density, double* temperature, double* qsnow, double* nsnow, double* qsnow_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double qsnow_tmp = fmax(qsnow[ijk], 0.0);

                melt_snow(density[k], temperature[ijk], qsnow_tmp, nsnow[ijk], &qsnow_tendency[ijk]);

            }
        }
    }
    return;

};
