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

double rho_liq = 1000.0;
double visc_air = 2.0e-5;
double lhf = 3.34e5;
double small = 1.0e-10;
double cvl = 4190.0;
double cvi = 2106.0;
double lf0= 3.34e5;

struct hm_parameters{
    double a;
    double b;
    double c;
    double d;
    double gb1;
    double gbd1;
    double gd3;
    double gd6;
    double gamstar;
    double alpha_acc;
    double d_min;
    double d_max;
};

struct hm_properties{
    double mf;
    double diam;
    double vel;
    double lam;
    double n0;
};

struct ret_acc{
    double dyr;
    double dys;
    double dyl;
    double dyi;
};

double vapor_diffusivity(const double temp_, const double p0_){
    double val = 2.11e-5 * pow((temp_ / 273.15), 1.94) * (p0_ / 101325.0);
    return val;
};

double thermal_conductivity(const double temp_){
    double val = 2.591e-2 * pow((temp_ / 296.0), 1.5) * (416.0 / (temp_ - 120.0));
    return val;
};

double pv_star_ice_c(const double temp_){
    double t0 = 273.16;
    double a3i = 22.587;
    double a4i = -0.7;
    double es0 = 611.21;
    return es0 * exp(a3i * (temp_ - t0)/(temp_ - a4i));
}

double get_aut_rain_c(const double alpha_, const double ccn, struct hm_properties *liquid_prop){
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
        t2 = 3.72/(5.0e5*db - 7.5)/lwc;
        val = fmax(l2,0.0)/fmax(t2,1.0e-20)*alpha_;
    }

    return val;
};


double get_aut_snow_c(struct LookupStruct *LT, const double alpha_, const double p0_, const double qt_, const double qi_, const double temp_, struct hm_properties *ice_prop){
    /* Harrington 1995 snow autoconversion model */
    double pv_star = lookup(LT, temp_);
    //double pv_star = pv_star_ice_c(temp_);
    double qv_star = qv_star_c(p0_, qt_, pv_star);
    //double satratio = qt_/qv_star;
    double satratio = (qt_-qi_)/qv_star;
    double db_ice = 125.0e-6;
    double val = 0.0;
    double gtherm, psi;
    double vapor_diff = vapor_diffusivity(temp_, p0_);
    double therm_cond = thermal_conductivity(temp_);

    if( ice_prop->mf > 1.0e-10 && satratio > 1.0){
        //gtherm = 1.0e-7/(2.2*temp_/pv_star + 220.0/temp_);
        gtherm = 1.0 / ( (Rv*temp_/vapor_diff/pv_star) + (8.028e12/therm_cond/Rv/(temp_*temp_)) );
        psi = 4.0*pi*(satratio - 1.0)*gtherm;
        val = (psi*ice_prop->n0*exp(-ice_prop->lam*db_ice)
               *(db_ice*db_ice/3.0 + (1.0+ice_prop->lam*db_ice)/(ice_prop->lam*ice_prop->lam))*alpha_);
    }

    return val;
};

double get_evp_rain_c(struct LookupStruct *LT, const double alpha_, const double p0_, const double qt_,
                             const double temp_, struct hm_properties *_prop){
    double beta = 2.0;
    double pv_star = lookup(LT, temp_);
    double qv_star = qv_star_c(p0_, qt_, pv_star);
    double satratio = qt_/qv_star;
    double vapor_diff = vapor_diffusivity(temp_, p0_);
    double therm_cond = thermal_conductivity(temp_);

    double re, vent, gtherm;
    double val = 0.0;

    if( satratio < 1.0 && _prop->mf > 1.0e-15){
        re = _prop->diam*_prop->vel/visc_air;
        vent = 0.78 + 0.27*sqrt(re);
        //gtherm = 1.0e-7/(2.2*temp_/pv_star + 220.0/temp_);
        gtherm = 1.0 / ( (Rv*temp_/vapor_diff/pv_star) + (8.028e12/therm_cond/Rv/(temp_*temp_)) );
        val = 4.0*pi/beta*(satratio - 1.0)*vent*gtherm*_prop->n0/_prop->lam/_prop->lam*alpha_;
    }

    return val;
};

double get_evp_snow_c(struct LookupStruct *LT, const double alpha_, const double p0_,
                           const double qt_, double const temp_, struct hm_properties *_prop){
    double beta = 3.0;
    double pv_star = lookup(LT, temp_);
    //double pv_star = pv_star_ice_c(temp_);
    double qv_star = qv_star_c(p0_, qt_, pv_star);
    double satratio = qt_/qv_star;

    double vapor_diff = vapor_diffusivity(temp_, p0_);
    double therm_cond = thermal_conductivity(temp_);
    double re = _prop->diam*_prop->vel/visc_air;
    double vent = 0.65 + 0.39*sqrt(re);
    //double gtherm = 1.0e-7/(2.2*temp_/pv_star + 220.0/temp_);
    double gtherm = 1.0 / ( (Rv*temp_/vapor_diff/pv_star) + (8.028e12/therm_cond/Rv/(temp_*temp_)) );
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

double get_melt_snow_c(const double alpha_, const double temp_, struct hm_properties *snow_prop){
    double snow_loss = 0.0;
    double ka = 2.43e-2;
    double fvent = 0.65 + 0.39*sqrt(snow_prop->vel*snow_prop->diam/visc_air);

    if( temp_ > 273.16 && snow_prop->mf > small ){
        snow_loss = 2.0*pi*snow_prop->n0*ka/lhf*(temp_ - 273.16)*fvent/(snow_prop->lam*snow_prop->lam)*alpha_;
    }
    return snow_loss;
};

double get_lambda_c(const double alpha_, struct hm_properties *_prop, struct hm_parameters *_param){
    double wc = fmax(_prop->mf/alpha_, small);
    double val = pow((_param->a*_prop->n0*_param->gb1/wc), (1.0/(_param->b+1.0)));

    return val;
};

double get_dmean_c(const double alpha_, struct hm_properties *_prop, struct hm_parameters *_param){
    double wc = _prop->mf/alpha_ + small;
    double val = pow((wc*_param->gamstar/_param->a/_prop->n0), (1.0/(_param->b+1.0)));

    return val;
};

double get_droplet_dmean_c(const double alpha_, const double liq_, const double ccn){
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

double get_velmean_c(const double dmean, struct hm_parameters *_param){
    double val = _param->c*pow(dmean, _param->d);

    return val;
};

double get_n0_rain_c(const double alpha_, const double mf, struct hm_parameters *_param){
    double rwc = fmax(mf/alpha_, small);
    double b1 = 650.1466922699631;
    double b2 = -1.222222222222222;
    double n0_rain = b1*pow(rwc, b2);

    double n0_max = rwc*(_param->gamstar)/(_param->a)/pow(_param->d_min, (_param->b+1.0));
    double n0_min = rwc*(_param->gamstar)/(_param->a)/pow(_param->d_max, (_param->b+1.0));

    n0_rain = fmax(fmin(n0_rain,n0_max),n0_min);

    return n0_rain;
};

double get_n0_snow_c(const double alpha_, const double mf, struct hm_parameters *_param){
    /* Morrison et al. (2011a) */
    double swc = fmax(mf/alpha_, small);
    double y1 = 5.62e7;
    double y2 = 0.63;
    double n0_snow = y1*pow((swc*1000.0), y2);
    double n0_max = swc*_param->gamstar/_param->a/pow(_param->d_min, (_param->b+1.0));
    double n0_min = swc*_param->gamstar/_param->a/pow(_param->d_max, (_param->b+1.0));

    //n0_snow = 1.0e6;
    n0_snow = fmax(fmin(n0_snow,n0_max),n0_min);

    return n0_snow;
};

double get_n0_ice_c(const double alpha_, const double mf, const double n0_input, struct hm_parameters *_param){
    double iwc = fmax(mf/alpha_, small);
    double n0_ice = n0_input;
    double n0_max = iwc*_param->gamstar/_param->a/pow(_param->d_min, (_param->b+1.0));
    double n0_min = iwc*_param->gamstar/_param->a/pow(_param->d_max, (_param->b+1.0));

    n0_ice = fmax(fmin(n0_ice,n0_max),n0_min);

    return n0_ice;
};

void micro_substep_c(struct LookupStruct *LT, const double alpha, const double p0, const double qt, const double qi, const double T, const double ccn, const double n0_ice,
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
    *aut_snow = get_aut_snow_c(LT, alpha, p0, qt, qi, T, ice_prop);

    get_acc_c(alpha, T, ccn, rain_param, snow_param, liquid_param,
              ice_param, rain_prop, snow_prop, liquid_prop, ice_prop, _ret);

    *evp_rain = get_evp_rain_c(LT, alpha, p0, qt, T, rain_prop);
    *evp_snow = get_evp_snow_c(LT, alpha, p0, qt, T, snow_prop);

    *melt_snow = get_melt_snow_c(alpha, T, snow_prop);

    return;
};

double get_rain_vel_c(const double alpha_, const double qrain_, struct hm_parameters *rain_param,
                             struct hm_properties *rain_prop){
    rain_prop->n0 = get_n0_rain_c(alpha_, qrain_, rain_param);
    rain_prop->lam = get_lambda_c(alpha_, rain_prop, rain_param);

    double vel_rain = rain_param->c*rain_param->gbd1/rain_param->gb1/pow(rain_prop->lam, rain_param->d);

    return vel_rain;
};

double get_snow_vel_c(const double alpha_, const double qsnow_, struct hm_parameters *snow_param,
                             struct hm_properties *snow_prop){
    snow_prop->n0 = get_n0_snow_c(alpha_, qsnow_, snow_param);
    snow_prop->lam = get_lambda_c(alpha_, snow_prop, snow_param);
    //snow_prop->lam = 3.81e3 * pow(fmax(qsnow_/alpha_, small), -0.147); //alternative from Morrison 2011

    double vel_snow = snow_param->c*snow_param->gbd1/snow_param->gb1/pow(snow_prop->lam, snow_param->d);

    return vel_snow;
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

    const double b1 = 650.1466922699631;
    const double b2 = -1.222222222222222;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin-1; k<kmax+1; k++){
                const ssize_t ijk = ishift + jshift + k;

                double rwc = fmax(qrain[ijk]*density[k], small);
                double n0_rain = b1*pow(rwc, b2);
                double n0_max = rwc*N_MAX_RAIN;
                double n0_min = rwc*N_MIN_RAIN;

                nrain[ijk] = fmax(fmin(n0_rain,n0_max),n0_min);

                double lam = pow((A_RAIN*n0_rain*GB1_RAIN/rwc), (1.0/(B_RAIN+1.0)));
                qrain_velocity[ijk] = C_RAIN*GBD1_RAIN/GB1_RAIN/pow(lam, D_RAIN);

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

}

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

    const double y1 = 5.62e7;
    const double y2 = 0.63;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin-1; k<kmax+1; k++){
                const ssize_t ijk = ishift + jshift + k;

                double swc = fmax(qsnow[ijk]*density[k], SMALL);
                double n0_snow = y1*pow(swc*1000.0, y2);
                double n0_max = swc*N_MAX_SNOW;
                double n0_min = swc*N_MIN_SNOW;

                nsnow[ijk] = fmax(fmin(n0_snow,n0_max),n0_min);

                double lam = pow((A_SNOW*n0_snow*GB1_SNOW/swc), (1.0/(B_SNOW+1.0)));
                qsnow_velocity[ijk] = C_SNOW*GBD1_SNOW/GB1_SNOW/pow(lam, D_SNOW);

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

}


double entropy_src_precipitation_c(const double p0, const double T, const double qt, const double qv, const double L, const double precip_rate){
    double pd = pd_c(p0, qt, qv);
    double pv = pv_c(p0, qt, qv);
    double sd = sd_c(pd, T);
    double sv = sv_c(pv, T);
    double sc = sc_c(L, T);

    return (sd - sv + sc) * precip_rate;
};

double entropy_src_evaporation_c(const double p0, const double T, double Tw, const double qt, const double qv, const double L, const double evap_rate){
    double pd = pd_c(p0, qt, qv);
    double pv = pv_c(p0, qt, qv);
    double sd = sd_c(pd, T);
    double sv = sv_c(pv, Tw);
    double sc = sc_c(L, Tw);

    return (sv - sc - sd) * evap_rate;
};

void entropy_source_heating_rain(const struct DimStruct *dims, double* restrict T, double* restrict Twet, double* restrict qrain,
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
                entropy_tendency[ijk]+= qrain[ijk]*(fabs(w_qrain[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk])* dzi/T[ijk];
            }
        }
    }

    return;

}

void entropy_source_heating_snow(const struct DimStruct *dims, double* restrict T, double* restrict Twet, double* restrict qsnow,
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
                entropy_tendency[ijk]+= qsnow[ijk]*(fabs(w_qsnow[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk])* dzi/T[ijk];
            }
        }
    }

    return;

}


void entropy_source_drag(const struct DimStruct *dims, double* restrict T,  double* restrict qprec,
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
                entropy_tendency[ijk]+= g * qprec[ijk]* fabs(w_qprec[ijk])/ T[ijk];
            }
        }
    }

    return;

}
