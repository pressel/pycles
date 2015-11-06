#pragma once
#include "parameters.h"
#include "microphysics.h"
#include "advection_interpolation.h"
#include "entropies.h"
#include "thermodynamic_functions.h"
#include <math.h>

#define MAX_ITER  15 //maximum substep loops in source term computation
#define RAIN_MAX_MASS  5.2e-7 //kg; DALES: 5.0e-6 kg
#define RAIN_MIN_MASS  2.6e-10 //kg
#define DROPLET_MIN_MASS 4.20e-15 // kg
#define DROPLET_MAX_MASS  2.6e-10 //1.0e-11  // kg
#define DENSITY_SB  1.225 // kg/m^3; a reference density used in Seifert & Beheng 2006, DALES
#define XSTAR  2.6e-10
#define KCC  10.58e9 // Constant in cloud-cloud kernel, m^3 kg^{-2} s^{-1}: Using Value in DALES; also, 9.44e9 (SB01, SS08), 4.44e9 (SB06)
#define KCR  5.25   // Constant in cloud-rain kernel, m^3 kg^{-1} s^{-1}: Using Value in DALES and SB06;  KCR = kr = 5.78 (SB01, SS08)
#define KRR  7.12   // Constant in rain-rain kernel,  m^3 kg^{-1} s^{-1}: Using Value in DALES and SB06; KRR = kr = 5.78 (SB01, SS08); KRR = 4.33 (S08)
#define KAPRR  60.7  // Raindrop typical mass (4.471*10^{-6} kg to the -1/3 power), kg^{-1/3}; = 0.0 (SB01, SS08)
#define KAPBR  2.3e3 // m^{-1} - Only used in SB06 break-up
#define D_EQ  0.9e-3 // equilibrium raindrop diameter, m, used for SB-breakup
#define D_EQ_MU  1.1e-3 // equilibrium raindrop diameter, m, used for SB-mu, opt=4
#define A_RAIN_SED  9.65    // m s^{-1}
#define B_RAIN_SED  9.796 // 10.3    # m s^{-1}
#define C_RAIN_SED  600.0   // m^{-1}
#define C_LIQUID_SED 702780.63036 //1.19e8 *(3.0/(4.0*pi*rho_liq))**(2.0/3.0)*np.exp(5.0*np.log(1.34)**2.0)
#define A_VENT_RAIN  0.78
#define B_VENT_RAIN  0.308
#define NSC_3  0.892112 //cbrt(0.71) // Schmidt number to the 1/3 power
#define KIN_VISC_AIR  1.4086e-5 //m^2/s kinematic viscosity of air
#define A_NU_SQ sqrt(A_RAIN_SED/KIN_VISC_AIR)
#define SB_EPS  1.0e-13 //small value

//Unless specified otherwise, Diameter = Dm not Dp

//Note: All sb_shape_parameter_X functions must have same signature
double sb_rain_shape_parameter_0(double density, double qr, double Dm){
    //Seifert & Beheng 2001 and Seifert & Beheng 2006
    double shape_parameter = 0.0;
    return shape_parameter;
}

double sb_rain_shape_parameter_1(double density, double qr, double Dm ){
    //qr: rain specific humidity kg/kg
    //Dm is mass-weighted mean diameter
    //Seifert and Stevens 2008 and DALES v 3.1.1
    double shape_parameter = 10.0 * (1.0 + tanh( 1200.0 * (Dm - 1.4e-3) ));   // Note: UCLA-LES uses 1.5e-3
    return shape_parameter;
}

double sb_rain_shape_parameter_2(double density, double qr, double Dm){
    //qr: rain specific humidity kg/kg
    //DALES v3.2, v4.0
    double shape_parameter = fmin(30.0, -1.0+0.008*pow(qr*density, -0.6));
    return shape_parameter;
}

double sb_rain_shape_parameter_4(double density, double qr, double Dm ){
    //qr: rain specific humidity kg/kg
    //Dm: mass-weighted mean diameter
    //Seifert 2008
    double shape_parameter;
    if(Dm <= D_EQ_MU){
        shape_parameter = 6.0 * tanh((4000.0*(Dm - D_EQ_MU))*(4000.0*(Dm - D_EQ_MU))) + 1.0;
    }
    else{
        shape_parameter = 30.0 * tanh((1000.0*(Dm - D_EQ_MU))*(1000.0*(Dm - D_EQ_MU))) + 1.0;
    }
 return shape_parameter;
}

double sb_droplet_nu_0(double density, double ql){
    // ql: cloud liquid droplet specific humidity kg/kg
    // density: kg/m^3
    // Seifert & Beheng 2001, Seifert and Stevens 2008
    double nu = 0.0;
    return nu;
}

double sb_droplet_nu_1(double density, double ql){
    // ql: cloud liquid droplet specific humidity kg/kg
    // density: kg/m^3
    // Seifert & Beheng 2006
    double nu=1.0;
    return nu;
}

double sb_droplet_nu_2(double density, double ql){
    // ql: cloud liquid specific humidity kg/kg
    // density: kg/m^3
    // DALES
    double nu = 1.58 * (1000.0 * density * ql) - 0.28;
    return nu;

}


void sb_autoconversion_rain(double (*droplet_nu)(double,double), double density, double nl, double ql, double qr, double* nr_tendency, double* qr_tendency){
    // Computation of rain specific humidity and number source terms from autoconversion of cloud liquid to rain
    double nu, phi, tau, tau_pow, droplet_mass;

    if(ql < SB_EPS){
        // if liquid specific humidity is negligibly small, set source terms to zero
        *qr_tendency = 0.0;
        *nr_tendency = 0.0;
    }
    else{
        nu = droplet_nu(density, ql);
        tau = fmin(fmax(1.0 - ql/(ql + qr), 0.0), 0.99);

        // Formulation used by DALES and Seifert & Beheng 2006
        tau_pow = pow(tau,0.7);
        phi = 400.0 * tau_pow * (1.0 - tau_pow) * (1.0 - tau_pow) * (1.0 - tau_pow);
        // Formulation used by Seifert & Beheng 2001, Seifert & Stevens 2008
        // tau_pow = pow(tau, 0.68);
        // phi = 600.0 * tau_pow * (1.0 - tau_pow)* (1.0 - tau_pow)* (1.0 - tau_pow)
        droplet_mass = microphysics_mean_mass(nl, ql, DROPLET_MIN_MASS, DROPLET_MAX_MASS);
        *qr_tendency = (KCC / (20.0 * XSTAR) * (nu + 2.0) * (nu + 4.0)/(nu * nu + 2.0 * nu + 1.0)
                        * ql * ql * droplet_mass * droplet_mass * (1.0 + phi/(1.0 - 2.0 * tau + tau * tau)) * DENSITY_SB);
        *nr_tendency = (*qr_tendency)/XSTAR;
    }
    // if prognostic cloud liquid is used, the following tendencies would also need to be computed
    //ql_tendency = -qr_tendency
    //nl_tendency = -2.0 * nr_tendency
    return;
}

void sb_accretion_rain(double density, double ql, double qr, double* qr_tendency){
    //Computation of tendency of rain specific humidity due to accretion of cloud liquid droplets
    double tau, phi;
    if(ql < SB_EPS || qr < SB_EPS){
        *qr_tendency = 0.0;
    }
    else{
        tau = fmin(fmax(1.0 - ql/(ql + qr), 0.0), 0.99);
        phi = pow((tau / (tau + 5.0e-5)), 4.0);      // - DALES, and SB06
        // phi = pow((tau/(tau+5.0e-4)),4.0);   // - SB01, SS08
        *qr_tendency = KCR * ql * qr * phi * sqrt(DENSITY_SB * density); //SB06, DALES formulation of effective density
    }
    // if prognostic cloud liquid is used, the following tendencies would also need to be computed
    //ql_tendency = -qr_tendency
    //droplet_mass = microphysics_mean_mass(nl, ql, DROPLET_MIN_MASS, DROPLET_MAX_MASS);
    //nl_tendency = ql_tendency/droplet_mass;
    return;
}

void sb_selfcollection_breakup_rain(double density, double nr, double qr, double mu, double rain_mass, double Dm, double* nr_tendency){
    //this function gives the net tendency breakup + selfcollection: nr_tendency = -phi*nr_tendency_sc
    double lambda_rain, phi_sc, phi_bk = 0.0;
    double nr_tendency_sc;

    if(qr < SB_EPS || nr < SB_EPS){
        *nr_tendency = 0.0;
    }
    else{
        lambda_rain = 1.0/cbrt(rain_mass * tgamma(mu + 1.0)/ tgamma(mu + 4.0));
        phi_sc = pow((1.0 + KAPRR/lambda_rain), -9.0); //Seifert & Beheng 2006, DALES
        // phi_sc = 1.0; //Seifert & Beheng 2001, Seifert & Stevens 2008, Seifert 2008
        nr_tendency_sc = -KRR * nr * qr * phi_sc * sqrt(DENSITY_SB*density);
        // Seifert & Stevens 2008, Seifert 2008, DALES
        if(Dm > 0.3e-3){
            phi_bk = 1000.0 * Dm - 0.1;
        }
        *nr_tendency = -phi_bk * nr_tendency_sc;

    }
    return;
}

void sb_evaporation_rain( double g_therm, double sat_ratio, double nr, double qr, double mu, double rain_mass, double Dp,
double Dm, double* nr_tendency, double* qr_tendency){
    double gamma, dpfv, phi_v;
    const double bova = B_RAIN_SED/A_RAIN_SED;
    const double cdp  = C_RAIN_SED * Dp;
    const double mupow = mu + 2.5;
    double qr_tendency_tmp = 0.0;

    if(qr < SB_EPS || nr < SB_EPS){
        *nr_tendency = 0.0;
        *qr_tendency = 0.0;
    }
    else if(sat_ratio >= 0.0){
        *nr_tendency = 0.0;
        *qr_tendency = 0.0;
    }
    else{
        gamma = 0.7; // gamma = 0.7 is used by DALES ; alternative expression gamma= d_eq/Dm * exp(-0.2*mu) is used by S08;
        phi_v = 1.0 - (0.5  * bova * pow(1.0 +  cdp, -mupow) + 0.125 * bova * bova * pow(1.0 + 2.0*cdp, -mupow)
                      + 0.0625 * bova * bova * bova * pow(1.0 +3.0*cdp, -mupow) + 0.0390625 * bova * bova * bova * bova * pow(1.0 + 4.0*cdp, -mupow));


        dpfv  = (A_VENT_RAIN * tgamma(mu + 2.0) * Dp + B_VENT_RAIN * NSC_3 * A_NU_SQ * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);

        qr_tendency_tmp = 2.0 * pi * g_therm * sat_ratio* nr * dpfv;
        *nr_tendency = gamma /rain_mass * qr_tendency_tmp;
        *qr_tendency = qr_tendency_tmp;
    }
    return;
}


void sb_sedimentation_velocity_rain(const struct DimStruct *dims, double (*rain_mu)(double,double,double),
                                        double* restrict density, double* restrict nr, double* restrict qr,
                                        double* restrict nr_velocity, double* restrict qr_velocity){


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
            for(ssize_t k=kmin-1; k<kmax+1; k++){
                const ssize_t ijk = ishift + jshift + k;
                double qr_tmp = fmax(qr[ijk],0.0);
                double density_factor = sqrt(DENSITY_SB/density[k]);
                double rain_mass = microphysics_mean_mass(nr[ijk], qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                double Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                double mu = rain_mu(density[k], qr_tmp, Dm);
                double Dp = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));

                nr_velocity[ijk] = -fmin(fmax( density_factor * (A_RAIN_SED - B_RAIN_SED * pow(1.0 + C_RAIN_SED * Dp, -mu - 1.0)) , 0.0),10.0);
                qr_velocity[ijk] = -fmin(fmax( density_factor * (A_RAIN_SED - B_RAIN_SED * pow(1.0 + C_RAIN_SED * Dp, -mu - 4.0)) , 0.0),10.0);

            }
        }
    }


     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;

                nr_velocity[ijk] = interp_2(nr_velocity[ijk], nr_velocity[ijk+1]) ;
                qr_velocity[ijk] = interp_2(qr_velocity[ijk], qr_velocity[ijk+1]) ;

            }
        }
    }


    return;

}




void sb_sedimentation_velocity_liquid(const struct DimStruct *dims, double* restrict density, double ccn,
                                      double* restrict ql, double* restrict qt_velocity){

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
            for(ssize_t k=kmin-1; k<kmax+1; k++){
                const ssize_t ijk = ishift + jshift + k;
                double nl = ccn/density[k];
                double ql_tmp = fmax(ql[ijk],0.0);
                double liquid_mass = microphysics_mean_mass(nl, ql_tmp, DROPLET_MIN_MASS, DROPLET_MAX_MASS);
                qt_velocity[ijk] = -C_LIQUID_SED * cbrt(liquid_mass * liquid_mass);

            }
        }
    }


     for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;

                qt_velocity[ijk] = interp_2(qt_velocity[ijk], qt_velocity[ijk+1]) ;
            }
        }
    }


    return;

}





void sb_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
                             double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt, double ccn,
                             double* restrict ql, double* restrict nr, double* restrict qr, double dt,
                             double* restrict nr_tendency_micro, double* restrict qr_tendency_micro, double* restrict nr_tendency, double* restrict qr_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evp;
    double qr_tendency_au, qr_tendency_ac,  qr_tendency_evp;
    double sat_ratio;


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
                qr[ijk] = fmax(qr[ijk],0.0);
                nr[ijk] = fmax(fmin(nr[ijk], qr[ijk]/RAIN_MIN_MASS),qr[ijk]/RAIN_MAX_MASS);
                double qv_tmp = qt[ijk] - fmax(ql[ijk],0.0);
                double qt_tmp = qt[ijk];
                double nl = ccn/density[k];
                double ql_tmp = fmax(ql[ijk],0.0);
                double qr_tmp = fmax(qr[ijk],0.0);
                double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);


                //holding nl fixed since it doesn't change between timesteps

                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    iter_count += 1;
                    sat_ratio = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt_tmp);
                    nr_tendency_au = 0.0;
                    nr_tendency_scbk = 0.0;
                    nr_tendency_evp = 0.0;
                    qr_tendency_au = 0.0;
                    qr_tendency_ac = 0.0;
                    qr_tendency_evp = 0.0;
                    //obtain some parameters
                    rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                    Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                    mu = rain_mu(density[k], qr_tmp, Dm);
                    Dp = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));
                    //compute the source terms
                    sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, &nr_tendency_au, &qr_tendency_au);
                    sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency_ac);
                    sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm, &nr_tendency_scbk);
                    sb_evaporation_rain( g_therm, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm, &nr_tendency_evp, &qr_tendency_evp);
                    //find the maximum substep time
                    dt_ = dt - time_added;
                    //check the source term magnitudes
                    nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp;
                    qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp;
                    ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac;

                    //Factor of 1.05 is ad-hoc
                    rate = 1.05 * ql_tendency_tmp * dt_ /(- fmax(ql_tmp,SB_EPS));
                    rate = fmax(1.05 * nr_tendency_tmp * dt_ /(-fmax(nr_tmp,SB_EPS)), rate);
                    rate = fmax(1.05 * qr_tendency_tmp * dt_ /(-fmax(qr_tmp,SB_EPS)), rate);
                    if(rate > 1.0 && iter_count < MAX_ITER){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }
                    //Integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    nr_tmp += nr_tendency_tmp * dt_;
                    qr_tmp += qr_tendency_tmp * dt_;
                    qv_tmp += -qr_tendency_evp * dt_;
                    qr_tmp = fmax(qr_tmp,0.0);
                    nr_tmp = fmax(fmin(nr_tmp, qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                    ql_tmp = fmax(ql_tmp,0.0);
                    qt_tmp = ql_tmp + qv_tmp;
                    time_added += dt_ ;


                }while(time_added < dt);
                nr_tendency_micro[ijk] = (nr_tmp - nr[ijk] )/dt;
                qr_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                nr_tendency[ijk] += nr_tendency_micro[ijk];
                qr_tendency[ijk] += qr_tendency_micro[ijk];

            }
        }
    }


    return;
}


void sb_qt_source_formation(const struct DimStruct *dims,double* restrict qr_tendency, double* restrict qt_tendency ){

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
                qt_tendency[ijk] += -qr_tendency[ijk];
            }
        }
    }
    return;
}


void sb_entropy_source_formation(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                              double* restrict p0, double* restrict T, double* restrict Twet, double* restrict qt, double* restrict qv,
                              double* restrict qr_tendency,  double* restrict entropy_tendency){


    //Here we compute the source terms of total water and entropy related to microphysics. See Pressel et al. 2015, Eq. 49-54
    //
    //Some simplifications are possible because there is only a single hydrometeor species (rain), so d(qr)/dt
    //can only represent a transfer to/from the equilibrium mixture. Furthermore, formation and evaporation of
    //rain cannot occur simultaneously at the same physical location so keeping track of each sub-tendency is not required


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

                const double lam_T = lam_fp(T[ijk]);
                const double L_fp_T = L_fp(T[ijk],lam_T);
                const double lam_Tw = lam_fp(Twet[ijk]);
                const double L_fp_Tw = L_fp(Twet[ijk],lam_Tw);
                const double pv_star_T = lookup(LT, T[ijk]);
                const double pv_star_Tw = lookup(LT,Twet[ijk]);
                const double pv = pv_c(p0[k], qt[ijk], qv[ijk]);
                const double pd = p0[k] - pv;
                const double sd_T = sd_c(pd, T[ijk]);
                const double sv_star_T = sv_c(pv_star_T,T[ijk] );
                const double sv_star_Tw = sv_c(pv_star_Tw, Twet[ijk]);
                const double S_P = sd_T - sv_star_T + L_fp_T/T[ijk];
                const double S_E = sv_star_Tw - L_fp_Tw/Twet[ijk] - sd_T;
                const double S_D = -Rv * log(pv/pv_star_T) + cpv * log(T[ijk]/Twet[ijk]);
                entropy_tendency[ijk] += S_P * 0.5 * (qr_tendency[ijk] + fabs(qr_tendency[ijk])) - (S_E + S_D) * 0.5 *(qr_tendency[ijk] - fabs(qr_tendency[ijk])) ;

            }
        }
    }

    return;

}





void sb_entropy_source_heating(const struct DimStruct *dims, double* restrict T, double* restrict Twet, double* restrict qr,
                               double* restrict w_qr, double* restrict w,  double* restrict entropy_tendency){


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
                entropy_tendency[ijk]+= qr[ijk]*(fabs(w_qr[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk])* dzi/T[ijk];
            }
        }
    }

    return;

}

void sb_entropy_source_drag(const struct DimStruct *dims, double* restrict T,  double* restrict qr,
                            double* restrict w_qr, double* restrict entropy_tendency){



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
                entropy_tendency[ijk]+= g * qr[ijk]* fabs(w_qr[ijk])/ T[ijk];
            }
        }
    }

    return;

}

///==========================To facilitate output=============================

void sb_autoconversion_rain_wrapper(const struct DimStruct *dims,  double (*droplet_nu)(double,double),
                                    double* restrict density,  double ccn, double* restrict ql,  double* restrict qr,
                                    double* restrict nr_tendency, double* restrict qr_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments

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
                const double nl = ccn/density[k];
                //compute the source terms
                double ql_tmp = fmax(ql[ijk], 0.0);
                double qr_tmp = fmax(qr[ijk], 0.0);
                sb_autoconversion_rain(droplet_nu, density[k], nl, ql_tmp, qr_tmp, &nr_tendency[ijk], &qr_tendency[ijk]);


            }
        }
    }
    return;
}

void sb_accretion_rain_wrapper(const struct DimStruct *dims, double* restrict density,  double* restrict ql,
                               double* restrict qr, double* restrict qr_tendency){

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
                const double ql_tmp = fmax(ql[ijk], 0.0);
                const double qr_tmp = fmax(qr[ijk], 0.0);
                sb_accretion_rain(density[k], ql_tmp, qr_tmp, &qr_tendency[ijk]);

            }
        }
    }
    return;
}

void sb_selfcollection_breakup_rain_wrapper(const struct DimStruct *dims, double (*rain_mu)(double,double,double),
                                            double* restrict density, double* restrict nr, double* restrict qr, double* restrict nr_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu;
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
               //obtain some parameters
                const double qr_tmp = fmax(qr[ijk],0.0);
                const double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                const double rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                const double Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                const double mu = rain_mu(density[k], qr_tmp, Dm);

                //compute the source terms
                sb_selfcollection_breakup_rain(density[k], nr_tmp, qr_tmp, mu, rain_mass, Dm, &nr_tendency[ijk]);

            }
        }
    }
    return;
}

void sb_evaporation_rain_wrapper(const struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double (*rain_mu)(double,double,double),  double* restrict density, double* restrict p0,  double* restrict temperature,  double* restrict qt,
                             double* restrict ql, double* restrict nr, double* restrict qr, double* restrict nr_tendency, double* restrict qr_tendency){

    //Here we compute the source terms for nr and qr (number and mass of rain)
    //Temporal substepping is used to help ensure boundedness of moments
    double rain_mass, Dm, mu, Dp;
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
                const double qr_tmp = fmax(qr[ijk],0.0);
                const double nr_tmp = fmax(fmin(nr[ijk], qr_tmp/RAIN_MIN_MASS),qr_tmp/RAIN_MAX_MASS);
                const double qv = qt[ijk] - ql[ijk];
                const double sat_ratio = microphysics_saturation_ratio(LT, temperature[ijk], p0[k], qt[ijk]);
                const double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature[ijk]);
                //obtain some parameters
                const double rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, RAIN_MIN_MASS, RAIN_MAX_MASS);
                const double Dm = cbrt(rain_mass * 6.0/DENSITY_LIQUID/pi);
                const double mu = rain_mu(density[k], qr_tmp, Dm);
                const double Dp = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));
                //compute the source terms
                sb_evaporation_rain( g_therm, sat_ratio, nr_tmp, qr_tmp, mu, rain_mass, Dp, Dm, &nr_tendency[ijk], &qr_tendency[ijk]);

            }
        }
    }
    return;
}

