#pragma once
#include "grid.h"
#include "lookup.h"
#include "parameters.h"
#include "micro_parameters.h"
#include <stdio.h>
#include <math.h>


inline double get_aut_rain_c(const double alpha_, const double ccn, struct hm_properties *liquid_prop){
    /* Berry-Reinhardt 74 rain autoconversion model*/
    double l2, t2;
    const double ccn_ = ccn*1.0e-6;
    double varm6, val;

    if(ccn_ > 300.0){
        varm6 = 1.189633;
    }
    else{
        varm6 = 1.055572;
    }

    double lwc = liquid_prop->mf/alpha_;
    double df = liquid_prop->diam;
    double db = df*varm6;

    if(db <= 15.0e-6){
        val = 0.0;
    }
    else{
        l2 = 2.7e-2*(1.0e20/16.0*(db*db*db)*df - 0.4)*lwc;
        t2 = 3.72/(5.0e5);
        val = fmax(l2,0.0)/fmax(t2,1.0e-20)*alpha_;
    }

    return val;
};


inline double get_aut_snow_c(struct LookupStruct *LT, const double alpha_, const double p0_, const double qt_, double temp_, struct hm_properties *ice_prop){
    /* Harrington 1995 snow autoconversion model */
    /* Saturation vapor pressure over ICE??? */
    double pv_star = lookup(LT, temp_);
    double y_sat_ice = pv_star/(p0_-pv_star)*eps_v*(1-qt_);
    double satratio = qt_/y_sat_ice;
    double db_ice = 125.0e-6;
    double val = 0.0;
    double gtherm, psi;

    if( ice_prop->mf > 1.0e-10 && satratio > 1.0){
        gtherm = 1.0e-7/(2.2*temp_/pv_star + 220.0/temp_);
        psi = 4.0*pi*(satratio - 1.0)*gtherm;
        val = (psi*ice_prop->n0*exp(-ice_prop->lam*db_ice)
               *(db_ice*db_ice/3.0 + (1.0+ice_prop->lam*db_ice)/(ice_prop->lam*ice_prop->lam))*alpha_);
    }

    return val;
};

inline double get_evp_rain_c(struct LookupStruct *LT, const double alpha_, const double p0_, const double qt_,
                             const double temp_, struct hm_properties *_prop){
    double beta = 2.0;
    double pv_star = lookup(LT, temp_);
    double y_sat = pv_star/(p0_-pv_star)*eps_v*(1-qt_);
    double satratio = qt_/y_sat;

    double re, vent, gtherm;
    double val = 0.0;

    if( satratio < 1.0 && _prop->mf > 1.0e-15){
        re = _prop->diam*_prop->vel/visc_air;
        vent = 0.78 + 0.27*(pow(re, 0.5));
        gtherm = 1.0e-7/(2.2*temp_/pv_star + 220.0/temp_);
        val = 4.0*pi/beta*(satratio - 1.0)*vent*gtherm*_prop->n0/_prop->lam/_prop->lam*alpha_;
    }

    return val;
};

inline double get_evp_snow_c(struct LookupStruct *LT, const double alpha_, const double p0_,
                           const double qt_, double const temp_, struct hm_properties *_prop){
    double beta = 3.0;
    double pv_star = lookup(LT, temp_);
    double y_sat = pv_star/(p0_-pv_star)*eps_v*(1-qt_);
    double satratio = qt_/y_sat;

    double re = _prop->diam*_prop->vel/visc_air;
    double vent = 0.65 + 0.39*(pow(re, 0.5));
    double gtherm = 1.0e-7/(2.2*temp_/pv_star + 220.0/temp_);
    double val = 0.0;

    if( _prop->mf > 1.0e-15 ){
        val = 4.0*pi/beta*(satratio - 1.0)*vent*gtherm*_prop->n0/_prop->lam/_prop->lam*alpha_;
    }

    return val;
};

void get_acc_c(const double alpha_, const double temp_, const double ccn_, struct hm_parameters *rain_param,
             struct hm_parameters *snow_param, struct hm_parameters *liquid_param, struct hm_parameters *ice_param,
             struct hm_properties *rain_prop, struct hm_properties *snow_prop, struct hm_properties *liquid_prop,
             struct hm_properties *ice_prop, struct ret_acc *_ret ){

    double factor_r = 0.0;
    double factor_s = 0.0;
    double piacr = 0.0;

    double e_ri = 1.0;
    double e_si = exp(9.0e-2*(temp_ - 273.15));
    double e_rl = 0.85;
    double e_sl = 0.8;

    double d_drop = liquid_prop->diam;
    double rho_ = 1.0/alpha_;

    if( snow_prop->diam < 150.0e-6 ){
        e_sl = 0.0;
    }
    else{
        if( d_drop < 15.0e-6 ){
            e_sl = 0.0;
        }
        else if( d_drop < 40.0e-6 ){
            e_sl = (d_drop - 15.0e-6)/25.0e-6*e_sl;
        }
    }

    if( rain_prop->mf > small ){
        factor_r = rho_*rain_param->gd3*rain_prop->n0*pi*rain_param->alpha_acc*rain_param->c*0.25*pow(rain_prop->lam, (-rain_param->d - 3.0));
        if( ice_prop->mf > 1.0e-10 ){
            piacr = rain_prop->n0*ice_prop->n0/ice_prop->lam*e_ri*pi*0.25*rain_param->a*rain_param->c*rain_param->gd6*pow(rain_prop->lam, (-rain_param->d - 6.0));
        }
    }

    if( snow_prop->mf > small ){
        factor_s = rho_*snow_param->gd3*snow_prop->n0*pi*snow_param->alpha_acc*snow_param->c*0.25*pow(snow_prop->lam, (-snow_param->d-3.0));
    }

    double src_ri = -piacr/rho_;
    double src_rl = factor_r*e_rl*liquid_prop->mf/rho_;
    double src_si = factor_s*e_si*ice_prop->mf/rho_;
    double rime_sl = factor_s*e_sl*liquid_prop->mf/rho_;
    double src_sl = 0.0;

    if( temp_ > 273.16 ){
        src_sl = -cvl/lf0*(temp_-273.16)*rime_sl;
        src_rl = src_rl + rime_sl - src_sl;
    }
    else{
        src_sl = rime_sl + (factor_r*e_ri*ice_prop->mf + piacr)/rho_;
    }

    /* Now precip-precip interactions */
    double src_r = 0.0;
    double src_s = 0.0;
    double dv, k_2s, k_2r;

    if( rain_prop->mf > small && snow_prop->mf > small ){
        dv = fabs(rain_prop->vel - snow_prop->vel);
        k_2s = (30.0/(pow(rain_prop->lam, 6.0)*snow_prop->lam) + 12.0/(pow(rain_prop->lam, 5.0)*pow(snow_prop->lam, 2.0))
                + 3.0/(pow(rain_prop->lam, 4.0)*pow(snow_prop->lam, 3.0)));
        k_2r = (1.0/(pow(rain_prop->lam, 3.0)*pow(snow_prop->lam, 3.0)) + 3.0/(pow(rain_prop->lam, 2.0)*pow(snow_prop->lam, 4.0))
                + 6.0/(rain_prop->lam*pow(snow_prop->lam, 5.0)));
        if( temp_ < 273.16 ){
            src_s = pi*dv*snow_prop->n0*rain_prop->n0*rain_param->a*k_2s*alpha_;
            src_r = -src_s;
        }
        else{
            src_r = pi*dv*snow_prop->n0*rain_prop->n0*rain_param->a*k_2r*alpha_;
            src_s = -src_r;
        }
    }

    _ret->dyr = src_r + src_rl + src_ri;
    _ret->dys = src_s + src_sl + src_si;
    _ret->dyl = -(src_rl + rime_sl);
    _ret->dyi = -(src_ri + src_si);

    return;
};

inline double get_melt_snow_c(const double alpha_, const double temp_, struct hm_properties *snow_prop){
    double snow_loss = 0.0;
    double ka = 2.43e-2;
    double fvent = 0.65 + 0.39*sqrt(snow_prop->vel*snow_prop->diam/visc_air);

    if( temp_ > 273.16 && snow_prop->mf > small ){
        snow_loss = 2.0*pi*snow_prop->n0*ka/lhf*(temp_ - 273.16)*fvent/(snow_prop->lam*snow_prop->lam)*alpha_;
    }
    return snow_loss;
};

inline double get_lambda_c(const double alpha_, struct hm_properties *_prop, struct hm_parameters *_param){
    double wc = fmax(_prop->mf/alpha_, small);
    double val = pow((_param->a*_prop->n0*_param->gb1/wc), (1.0/(_param->b+1.0)));

    return val;
};

inline double get_dmean_c(const double alpha_, struct hm_properties *_prop, struct hm_parameters *_param){
    double wc = _prop->mf/alpha_ + small;
    double val = pow((wc*_param->gamstar/_param->a/_prop->n0), (1.0/(_param->b+1.0)));

    return val;
};

inline double get_droplet_dmean_c(const double alpha_, const double liq_, const double ccn){
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

    double lwc = liq_/alpha_;
    double df = pow((6.0*lwc/(pi*rho_liq*ntot)), (1.0/3.0));

    return df;
};

inline double get_velmean_c(const double dmean, struct hm_parameters *_param){
    double val = _param->c*pow(dmean, _param->d);

    return val;
};

inline double get_n0_rain_c(const double alpha_, const double mf, struct hm_parameters *_param){
    double rwc = fmax(mf/alpha_, small);
    double b1 = 650.1466922699631;
    double b2 = -1.222222222222222;
    double n0_rain = b1*pow(rwc, b2);

    double n0_max = rwc*(_param->gamstar)/(_param->a)/pow(_param->d_min, (_param->b+1.0));
    double n0_min = rwc*(_param->gamstar)/(_param->a)/pow(_param->d_max, (_param->b+1.0));

    n0_rain = fmax(fmin(n0_rain,n0_max),n0_min);

    return n0_rain;
};

inline double get_n0_snow_c(const double alpha_, const double mf, struct hm_parameters *_param){
    /* Morrison et al. (2011a) */
    double swc = fmax(mf/alpha_, small);
    double y1 = 5.62e7;
    double y2 = 0.63;
    double n0_snow = y1*pow((swc*1000.0), y2);
    double n0_max = swc*_param->gamstar/_param->a/pow(_param->d_min, (_param->b+1.0));
    double n0_min = swc*_param->gamstar/_param->a/pow(_param->d_max, (_param->b+1.0));

    n0_snow = 1.0e6;
    n0_snow = fmax(fmin(n0_snow,n0_max),n0_min);

    return n0_snow;
};

inline double get_n0_ice_c(const double alpha_, const double mf, const double n0_input, struct hm_parameters *_param){
    double iwc = fmax(mf/alpha_, small);
    double n0_ice = n0_input;
    double n0_max = iwc*_param->gamstar/_param->a/pow(_param->d_min, (_param->b+1.0));
    double n0_min = iwc*_param->gamstar/_param->a/pow(_param->d_max, (_param->b+1.0));

    n0_ice = fmax(fmin(n0_ice,n0_max),n0_min);

    return n0_ice;
};

void micro_substep_c(struct LookupStruct *LT, const double alpha, const double p0, const double qt, const double T, const double ccn, const double n0_ice,
                     struct hm_parameters *rain_param, struct hm_parameters *snow_param, struct hm_parameters *liquid_param, struct hm_parameters *ice_param,
                     struct hm_properties *rain_prop, struct hm_properties *snow_prop, struct hm_properties *liquid_prop, struct hm_properties *ice_prop,
                     double* aut_rain, double* aut_snow, struct ret_acc *_ret, double* evp_rain,
                     double* evp_snow, double* melt_snow){

    rain_prop->n0   = get_n0_rain_c(alpha, rain_prop->mf, rain_param);
    rain_prop->lam  = get_lambda_c(alpha,rain_prop, rain_param);
    rain_prop->diam = get_dmean_c(alpha, rain_prop, rain_param);
    rain_prop->vel  = get_velmean_c(rain_prop->diam, rain_param);

    snow_prop->n0   = get_n0_snow_c(alpha, snow_prop->mf, snow_param);
    snow_prop->lam  = get_lambda_c(alpha, snow_prop, snow_param);
    snow_prop->diam = get_dmean_c(alpha, snow_prop, snow_param);
    snow_prop->vel  = get_velmean_c(snow_prop->diam, snow_param);

    liquid_prop->diam = get_droplet_dmean_c(alpha, liquid_prop->mf, ccn);

    ice_prop->n0  = get_n0_ice_c(alpha, ice_prop->mf, n0_ice, ice_param);
    ice_prop->lam = get_lambda_c(alpha, ice_prop, ice_param);
    ice_prop->diam = get_dmean_c(alpha, ice_prop, ice_param);

    *aut_rain = get_aut_rain_c(alpha, ccn, liquid_prop);
    *aut_snow = get_aut_snow_c(LT, alpha, p0, qt, T, ice_prop);

    get_acc_c(alpha, T, ccn, rain_param, snow_param, liquid_param,
              ice_param, rain_prop, snow_prop, liquid_prop, ice_prop, _ret);

    *evp_rain = get_evp_rain_c(LT, alpha, p0, qt, T, rain_prop);
    *evp_snow = get_evp_snow_c(LT, alpha, p0, qt, T, snow_prop);

    *melt_snow = get_melt_snow_c(alpha, T, snow_prop);

    return;
};

inline double get_rain_vel_c(const double alpha_, const double qrain_, struct hm_parameters *rain_param,
                             struct hm_properties *rain_prop){
    rain_prop->n0 = get_n0_rain_c(alpha_, qrain_, rain_param);
    rain_prop->lam = get_lambda_c(alpha_, rain_prop, rain_param);

    double vel_rain = rain_param->c*rain_param->gbd1/rain_param->gb1/pow(rain_prop->lam, rain_param->d);

    return vel_rain;
};

inline double get_snow_vel_c(const double alpha_, const double qsnow_, struct hm_parameters *snow_param,
                             struct hm_properties *snow_prop){
    snow_prop->n0 = get_n0_snow_c(alpha_, qsnow_, snow_param);
    snow_prop->lam = get_lambda_c(alpha_, snow_prop, snow_param);
    //snow_prop->lam = 3.81e3 * pow(fmax(qsnow_/alpha_, small), -0.147); //alternative from Morrison 2011

    double vel_snow = snow_param->c*snow_param->gbd1/snow_param->gb1/pow(snow_prop->lam, snow_param->d);

    return vel_snow;
};
