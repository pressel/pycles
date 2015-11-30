import numpy as np
cimport numpy as np
include 'parameters.pxi'

#Below are microphysical parameters corresponding to Grabowski, JAS, 1998 microphysics scheme
#following Kaul et al. 2015

'''some parameters held in common by microphysics schemes'''
cdef double rho_liq = 1000.0 #[kg/m^3]
cdef double visc_air = 2.0e-5
cdef double lhf = 3.34e5 # latent heat of fusion
cdef double small = 1.0e-10
cdef double cvl = 4190.0
cdef double cvi = 2106.0
cdef double lf0 = 3.34e5 # latent heat of fusion

# cdef hm_parameters rain_param
# cdef hm_parameters snow_param
# cdef hm_parameters ice_param
# cdef hm_parameters liquid_param
#
#
# rain_param.a = pi/6.0*rho_liq
# rain_param.b = 3.0
# rain_param.c = 130.0
# rain_param.d = 0.5
# rain_param.gb1 = gamma(1.0+rain_param.b)
# rain_param.gbd1 = gamma(1.0+rain_param.b+rain_param.d)
# rain_param.gd3 = gamma(3.0+rain_param.d)
# rain_param.gd6 = gamma(6.0 + rain_param.d)
# rain_param.gamstar = rain_param.gb1**(1.0/rain_param.b)
# rain_param.alpha_acc = 1.0
# rain_param.d_min = 50.0e-6
# rain_param.d_max = 2000.0e-6
#
#
# snow_param.a = 2.5e-2
# snow_param.b = 2.0
# snow_param.c = 4.0
# snow_param.d = 0.25
# snow_param.gb1 = gamma(1.0+snow_param.b)
# snow_param.gbd1 = gamma(1.0+snow_param.b+snow_param.d)
# snow_param.gd3 = gamma(3.0+snow_param.d)
# snow_param.gamstar = snow_param.gb1**(1.0/snow_param.b)
# snow_param.alpha_acc = 0.3
# snow_param.d_min = 30.0e-6
# snow_param.d_max = 2000.0e-6
#
#
#
# liquid_param.a = pi/6.0*rho_liq
# liquid_param.b = 3.0
# liquid_param.c = 0.0
# liquid_param.d = 0.0
# liquid_param.gb1 = gamma(1.0+liquid_param.b)
# liquid_param.gbd1 = gamma(1.0+liquid_param.b+liquid_param.d)
# liquid_param.gd3 = gamma(3.0+liquid_param.d)
# liquid_param.gamstar = liquid_param.gb1**(1.0/liquid_param.b)
# liquid_param.alpha_acc = 1.0
# liquid_param.d_min = 2e-6
# liquid_param.d_max = 30.0e-6
#
#
# cdef double rho_ice = 900.0
# ice_param.a = pi/6.0*rho_ice
# ice_param.b = 3.0
# ice_param.c = 0.0
# ice_param.d = 0.0
# ice_param.gb1 = gamma(1.0+ice_param.b)
# ice_param.gbd1 = gamma(1.0+ice_param.b+ice_param.d)
# ice_param.gd3 = gamma(3.0+ice_param.d)
# ice_param.gamstar = ice_param.gb1**(1.0/ice_param.b)
# ice_param.alpha_acc = 1.0
# ice_param.d_min = 12.5e-6
# ice_param.d_max = 650.0e-6
#
#
#
#
# cdef inline double gamma(double xx):
#     cdef double val
#
#     cdef double stp = 2.5066282746310005
#     cdef double cof[6]
#     cof[0] = 76.18009172947146
#     cof[1] = -86.50532032941677
#     cof[2] = 24.01409824083091
#     cof[3] =  -1.231739572450155
#     cof[4] =  0.1208650973866179e-2
#     cof[5] =  -0.5395239384953e-5
#     cdef double x = xx
#     cdef double y = x
#     cdef double tmp = x + 5.5
#     tmp = (x+0.5)*np.log(tmp) - tmp
#     cdef double ser = 1.000000000190015
#     cdef int i
#     for i in np.arange(6):
#         y = y + 1.0
#         ser = ser + cof[i]/y
#
#     cdef double gammaln = tmp + np.log(stp*ser/x)
#     val = np.exp(gammaln)
#     return val
#
#
#
#
