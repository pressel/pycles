#pragma once
#include "parameters.h"
#include "microphysics.h"
#include <math.h>

#define max_iter  5 //maximum substep loops in source term computation
#define rain_max_mass  5.2e-7 //kg; DALES: 5.0e-6 kg
#define rain_min_mass  2.6e-10 //kg
#define droplet_min_mass 4.20e-15 // kg
#define droplet_max_mass  2.6e-10 //1.0e-11  // kg
#define density_sb  1.225 // kg/m^3; a reference density used in Seifert & Beheng 2006, DALES
#define xstar  rain_min_mass
#define kcc  10.58e9 // Constant in cloud-cloud kernel, m^3 kg^{-2} s^{-1}: Using Value in DALES; also, 9.44e9 (SB01, SS08), 4.44e9 (SB06)
#define kcr  5.25   // Constant in cloud-rain kernel, m^3 kg^{-1} s^{-1}: Using Value in DALES and SB06;  kcr = kr = 5.78 (SB01, SS08)
#define krr  7.12   // Constant in rain-rain kernel,  m^3 kg^{-1} s^{-1}: Using Value in DALES and SB06; krr = kr = 5.78 (SB01, SS08); krr = 4.33 (S08)
#define kaprr  60.7  // Raindrop typical mass (4.471*10^{-6} kg to the -1/3 power), kg^{-1/3}; = 0.0 (SB01, SS08)
#define kapbr  2.3e3 // m^{-1} - Only used in SB06 break-up
#define d_eq  0.9e-3 // equilibrium raindrop diameter, m, used for SB-breakup
#define d_eq_mu  1.1e-3 // equilibrium raindrop diameter, m, used for SB-mu, opt=4
#define a_rain_sed  9.65    // m s^{-1}
#define b_rain_sed  9.796 // 10.3    # m s^{-1}
#define c_rain_sed  600.0   // m^{-1}
#define a_vent_rain  0.78
#define b_vent_rain  0.308

#define nsc_3  cbrt(0.71) // Schmidt number to the 1/3 power
#define kin_visc_air  1.4086e-5 //m^2/s kinematic viscosity of air
#define a_nu_sq sqrt(a_rain_sed/kin_visc_air)

#define sb_eps  1.0e-13 //small value
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
    if(Dm <= d_eq_mu){
        shape_parameter = 6.0 * tanh((4000.0*(Dm - d_eq_mu))*(4000.0*(Dm - d_eq_mu))) + 1.0;
    }
    else{
        shape_parameter = 30.0 * tanh((1000.0*(Dm - d_eq_mu))*(1000.0*(Dm - d_eq_mu))) + 1.0;
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

    if(ql < sb_eps){
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
        droplet_mass = microphysics_mean_mass(nl, ql, droplet_min_mass, droplet_max_mass);
        *qr_tendency = (kcc / (20.0 * xstar) * (nu + 2.0) * (nu + 4.0)/(nu * nu + 2.0 * nu + 1.0)
                        * ql * ql * droplet_mass * droplet_mass * (1.0 + phi/(1.0 - 2.0 * tau + tau * tau)) * density_sb);
        *nr_tendency = (*qr_tendency)/xstar;
    }
    // if prognostic cloud liquid is used, the following tendencies would also need to be computed
    //ql_tendency = -qr_tendency
    //nl_tendency = -2.0 * nr_tendency
    return;
}

void sb_accretion_rain(double density, double ql, double qr, double* qr_tendency){
    //Computation of tendency of rain specific humidity due to accretion of cloud liquid droplets
    double tau, phi;
    if(ql < sb_eps || qr < sb_eps){
        *qr_tendency = 0.0;
    }
    else{
        tau = fmin(fmax(1.0 - ql/(ql + qr), 0.0), 0.99);
        phi = pow((tau / (tau + 5.0e-5)), 4.0);      // - DALES, and SB06
        // phi = pow((tau/(tau+5.0e-4)),4.0);   // - SB01, SS08
        *qr_tendency = kcr * ql * qr * phi * sqrt(density_sb * density); //SB06, DALES formulation of effective density
    }
    // if prognostic cloud liquid is used, the following tendencies would also need to be computed
    //ql_tendency = -qr_tendency
    //droplet_mass = microphysics_mean_mass(nl, ql, droplet_min_mass, droplet_max_mass);
    //nl_tendency = ql_tendency/droplet_mass;
    return;
}

void sb_selfcollection_breakup_rain(double density, double nr, double qr, double mu, double rain_mass, double Dm, double* nr_tendency){
    //this function gives the net tendency breakup + selfcollection: nr_tendency = -phi*nr_tendency_sc
    double lambda_rain, phi_sc, phi_bk = 0.0;
    double nr_tendency_sc;

    if(qr < sb_eps){
        *nr_tendency = 0.0;
    }
    else{
        lambda_rain = 1.0/cbrt(rain_mass * tgamma(mu + 1.0)/ tgamma(mu + 4.0));
        phi_sc = pow((1.0 + kaprr/lambda_rain), -9.0); //Seifert & Beheng 2006, DALES
        // phi_sc = 1.0; //Seifert & Beheng 2001, Seifert & Stevens 2008, Seifert 2008
        nr_tendency_sc = -krr * nr * qr * phi_sc * sqrt(density_sb/density);
        // Seifert & Stevens 2008, Seifert 2008, DALES
        if(Dm > 0.3e-3){
            phi_bk = 1000.0 * Dm - 0.1;
        }
        *nr_tendency = -phi_bk * nr_tendency_sc;

    }
    return;
}

void sb_evaporation_rain( double g_therm, double sat_ratio, double nr, double qr, double mu, double Dp, double Dm, double* nr_tendency, double* qr_tendency){
    double gamma, dpfv, phi_v;
    const double bova = b_rain_sed/a_rain_sed;
    const double cdp  = c_rain_sed*Dp;
    const double mupow = mu + 2.5;
    double qr_tendency_tmp;

    if(qr < sb_eps || sat_ratio >= 0.0){
        *nr_tendency = 0.0;
        *qr_tendency = 0.0;
    }
    else{
        gamma = 0.7; // gamma = 0.7 is used by DALES ; alternative expression gamma= d_eq/Dm * exp(-0.2*mu) is used by S08;
        phi_v = 1.0 - (0.5  * bova * pow(1.0 +  cdp, -mupow) + 0.125 * bova * bova * pow(1.0 + 2.0*cdp, -mupow)
                      + 0.0625 * bova * bova * bova * pow(1.0 +3.0*cdp, -mupow) + 0.0390625 * bova * bova * bova * bova * pow(1.0 + 4.0*cdp, -mupow));


        dpfv  = (a_vent_rain * tgamma(mu + 2.0) * Dp + b_vent_rain * nsc_3 * a_nu_sq * tgamma(mupow) * pow(Dp, 1.5) * phi_v)/tgamma(mu + 1.0);
        qr_tendency_tmp = 2.0 * pi * g_therm * sat_ratio* nr * dpfv;
        *nr_tendency = gamma * nr/qr * qr_tendency_tmp;
        *qr_tendency = qr_tendency_tmp;
    }
    return;
}


void sb_sedimentation_velocity_rain(double (*rain_mu)(double,double,double),double density, double nr, double qr, double* nr_velocity, double* qr_velocity){

    double density_factor = sqrt(density_sb/density);
    double rain_mass = microphysics_mean_mass(nr, qr, rain_min_mass, rain_max_mass);
    double Dm = cbrt(rain_mass * 6.0/density_liquid/pi);
    double mu = rain_mu(density, qr, Dm);
    double Dp = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));

    *nr_velocity = -fmax( density_factor * (a_rain_sed - b_rain_sed * pow(1.0 + c_rain_sed * Dp, -mu - 1.0)) , 0.0);
    *qr_velocity = -fmax( density_factor * (a_rain_sed - b_rain_sed * pow(1.0 + c_rain_sed * Dp, -mu - 4.0)) , 0.0);
    return;
}

void sb_microphysics_sources(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                             double (*rain_mu)(double,double,double), double (*droplet_nu)(double,double),
                             double density, double p0,  double temperature,  double qt, double ccn,
                             double ql, double nr, double qr, double dt, double* nr_tendency, double* qr_tendency){

    const double qv = qt - ql;
    const double sat_ratio = microphysical_saturation_ratio(LT, lam_fp, L_fp, temperature, p0, qt, qv);
    const double g_therm = microphysics_g(LT, lam_fp, L_fp, temperature);
    const double nl = ccn/density;
    double ql_tmp = ql;
    double nr_tmp = nr;
    double qr_tmp = qr;
    //holding nl fixed since it doesn't change between timesteps
    double rain_mass, Dm, mu, Dp, nr_tendency_tmp, qr_tendency_tmp, ql_tendency_tmp;
    double nr_tendency_au, nr_tendency_scbk, nr_tendency_evp;
    double qr_tendency_au, qr_tendency_ac,  qr_tendency_evp;
    double time_added = 0.0, dt_, rate;
    ssize_t iter_count = 0;


    do{
        nr_tendency_tmp = 0.0;
        qr_tendency_tmp = 0.0;
        iter_count += 1;
        //obtain some parameters
        rain_mass = microphysics_mean_mass(nr_tmp, qr_tmp, rain_min_mass, rain_max_mass);
        Dm = cbrt(rain_mass * 6.0/density_liquid/pi);
        mu = rain_mu(density, qr, Dm);
        Dp = Dm * cbrt(tgamma(mu + 1.0) / tgamma(mu + 4.0));
        //compute the source terms
        sb_autoconversion_rain(droplet_nu, density, nl, ql_tmp, qr_tmp, &nr_tendency_au, &qr_tendency_au);
        sb_accretion_rain(density, ql_tmp, qr_tmp, &qr_tendency_ac);
        sb_selfcollection_breakup_rain(density, nr_tmp, qr_tmp, mu, rain_mass, Dm, &nr_tendency_scbk);
        sb_evaporation_rain( g_therm, sat_ratio, nr_tmp, qr_tmp, mu, Dp, Dm, &nr_tendency_evp, &qr_tendency_evp);
        //find the maximum substep time
        dt_ = dt - time_added;
        //check the source term magnitudes
        nr_tendency_tmp = nr_tendency_au + nr_tendency_scbk + nr_tendency_evp;
        qr_tendency_tmp = qr_tendency_au + qr_tendency_ac + qr_tendency_evp;
        ql_tendency_tmp = -qr_tendency_au - qr_tendency_ac;

        //Factor of 1.05 is ad-hoc
        rate = 1.05 * ql_tendency_tmp * dt_ /(-ql_tmp - sb_eps);
        rate = fmax(1.05 * nr_tendency_tmp * dt_ /(-nr_tmp - sb_eps), rate);
        rate = fmax(1.05 * qr_tendency_tmp * dt_ /(-qr_tmp - sb_eps), rate);
        if(rate > 1.0 && iter_count < max_iter){
            //Limit the timestep, but don't allow it to become vanishingly small
            //Don't adjust if we have reached the maximum iteration number
            dt_ = fmax(dt_/rate, 1.0e-3);
        }
        //Integrate forward in time
        ql_tmp += ql_tendency_tmp * dt_;
        nr_tmp += nr_tendency_tmp * dt_;
        qr_tmp += qr_tendency_tmp * dt_;
        time_added += dt_ ;


    }while(time_added < dt);

    *nr_tendency += (nr_tmp - nr )/dt;
    *qr_tendency += (qr_tmp - qr)/dt;


    return;
}

