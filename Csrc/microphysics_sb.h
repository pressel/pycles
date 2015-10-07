#pragma once
#include "parameters.h"
#include <math.h>
//Note: All sb_shape_parameter_X functions must have same signature
inline double sb_shape_parameter_0(double q, double diameter, double density){
 double shape_parameter = 0.0;
 return shape_parameter;
}

inline double sb_shape_parameter_1(double q, double diameter, double density){
 double shape_parameter = 10.0 * (1.0 + tanh( 1200.0 * (diameter - 1.4e-3) ));   // Note: UCLA-LES uses 1.5e-3
 return shape_parameter;
}

inline double sb_shape_parameter_2(double q, double diameter, double density){
 double shape_parameter = fmin(30.0, -1.0+0.008*pow(q*density, -0.6));
 return shape_parameter;
}

inline double sb_shape_parameter_4(double q, double diameter, double density){
    const double d_eq_mu = 1.1e-3; // equilibrium raindrop diameter, m, used for SB-mu, opt=4
    double shape_parameter;
    if(diameter <= d_eq_mu){
        shape_parameter = 6.0 * tanh((4000.0*(dm_-d_eq_mu))*(4000.0*(dm_-d_eq_mu))) + 1.0;
    }
    else{
        shape_parameter = 30.0 * tanh((1000.0*(dm_-d_eq_mu))*(1000.0*(dm_-d_eq_mu))) + 1.0;
    }

 return shape_parameter;
}

