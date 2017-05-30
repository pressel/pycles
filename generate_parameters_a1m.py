import numpy as np
from collections import OrderedDict
from scipy.special import gamma


def main():

    #########################
    # Users should modify here
    #########################

    parameters = OrderedDict()

    # Microphysical constants
    parameters['VISC_AIR'] = 2.0e-5
    parameters['LHF'] = 3.34e5  # constant latent heat of fusion
    parameters['SMALL'] = 1.0e-10
    parameters['DENSITY_ICE'] = 900.0  # density of ice, kg/m^3
    parameters['DENSITY_LIQUID'] = 1000.0  # density of liquid, km/m^3
    parameters['MAX_ITER'] = 15 # max interation of micro source terms

    # Liquid fraction power parameter
    parameters['POW_N'] = 0.5

    # Rain parameters
    parameters['A_RAIN'] = np.pi/6.0*parameters['DENSITY_LIQUID']
    parameters['B_RAIN'] = 3.0
    parameters['C_RAIN'] = 130.0
    parameters['D_RAIN'] = 0.5
    parameters['GB1_RAIN'] = gamma(1.0 + parameters['B_RAIN'])
    parameters['GBD1_RAIN'] = gamma(1.0 + parameters['B_RAIN'] + parameters['D_RAIN'])
    parameters['GD3_RAIN'] = gamma(3.0 + parameters['D_RAIN'])
    parameters['GD6_RAIN'] = gamma(6.0 + parameters['D_RAIN'])
    parameters['GSTAR_RAIN'] = parameters['GB1_RAIN']**(1.0 / parameters['B_RAIN'])
    parameters['DIA_MIN_RAIN'] = 50.0e-6  # min rain diameter
    parameters['DIA_MAX_RAIN'] = 2000.0e-6  # max rain diameter
    parameters['ALPHA_ACC_RAIN'] = 1.0  # geometrical factor for rain
    parameters['N_MAX_RAIN'] = parameters['GSTAR_RAIN'] / (parameters['A_RAIN'] * parameters['DIA_MIN_RAIN'] **
                                                           (1.0 + parameters['B_RAIN']))
    parameters['N_MIN_RAIN'] = parameters['GSTAR_RAIN'] / (parameters['A_RAIN'] * parameters['DIA_MAX_RAIN'] **
                                                           (1.0 + parameters['B_RAIN']))

    # Snow parameters
    parameters['A_SNOW'] = 2.5e-2
    parameters['B_SNOW'] = 2.0
    parameters['C_SNOW'] = 4.0
    parameters['D_SNOW'] = 0.25
    parameters['GB1_SNOW'] = gamma(1.0 + parameters['B_SNOW'])
    parameters['GBD1_SNOW'] = gamma(1.0 + parameters['B_SNOW'] + parameters['D_SNOW'])
    parameters['GD3_SNOW'] = gamma(3.0 + parameters['D_SNOW'])
    parameters['GSTAR_SNOW'] = parameters['GB1_SNOW']**(1.0 / parameters['B_SNOW'])
    parameters['DIA_MIN_SNOW'] = 30.0e-6  # min rain diameter
    parameters['DIA_MAX_SNOW'] = 2000.0e-6  # max rain diameter
    parameters['ALPHA_ACC_SNOW'] = 0.3  # geometrical factor for rain
    parameters['N_MAX_SNOW'] = parameters['GSTAR_SNOW'] / (parameters['A_SNOW'] * parameters['DIA_MIN_SNOW'] **
                                                           (1.0 + parameters['B_SNOW']))
    parameters['N_MIN_SNOW'] = parameters['GSTAR_SNOW'] / (parameters['A_SNOW'] * parameters['DIA_MAX_SNOW'] **
                                                           (1.0 + parameters['B_SNOW']))

    # Liquid parameters
    parameters['A_LIQ'] = np.pi/6.0*parameters['DENSITY_LIQUID']
    parameters['B_LIQ'] = 3.0
    parameters['C_LIQ'] = 0.0
    parameters['D_LIQ'] = 0.
    parameters['GB1_LIQ'] = gamma(1.0 + parameters['B_LIQ'])
    parameters['GBD1_LIQ'] = gamma(1.0 + parameters['B_LIQ'] + parameters['D_LIQ'])
    parameters['GD3_LIQ'] = gamma(3.0 + parameters['D_LIQ'])
    parameters['GSTAR_LIQ'] = parameters['GB1_LIQ']**(1.0 / parameters['B_LIQ'])
    parameters['DIA_MIN_LIQ'] = 2.0e-6  # min rain diameter
    parameters['DIA_MAX_LIQ'] = 30.0e-6  # max rain diameter
    parameters['ALPHA_ACC_LIQ'] = 1.0  # geometrical factor for rain

    # Ice parameters
    parameters['A_ICE'] = np.pi/6.0*parameters['DENSITY_ICE']
    parameters['B_ICE'] = 3.0
    parameters['C_ICE'] = 0.0
    parameters['D_ICE'] = 0.0
    parameters['GB1_ICE'] = gamma(1.0 + parameters['B_ICE'])
    parameters['GBD1_ICE'] = gamma(1.0 + parameters['B_ICE'] + parameters['D_ICE'])
    parameters['GD3_ICE'] = gamma(3.0 + parameters['D_ICE'])
    parameters['GSTAR_ICE'] = parameters['GB1_ICE']**(1.0 / parameters['B_ICE'])
    parameters['DIA_MIN_ICE'] = 12.5e-6  # min rain diameter
    parameters['DIA_MAX_ICE'] = 650.0e-6  # max rain diameter
    parameters['ALPHA_ACC_ICE'] = 1.0  # geometrical factor for rain
    parameters['N_MAX_ICE'] = parameters['GSTAR_ICE'] / (parameters['A_ICE'] * parameters['DIA_MIN_ICE'] **
                                                           (1.0 + parameters['B_ICE']))
    parameters['N_MIN_ICE'] = parameters['GSTAR_ICE'] / (parameters['A_ICE'] * parameters['DIA_MAX_ICE'] **
                                                           (1.0 + parameters['B_ICE']))



    #############################
    # Users shouldn't modify below
    #############################

    # Some warning to put in the generated code
    message1 = 'Generated code! Absolutely DO NOT modify this file, ' \
               'microphysical parameters should be modified in generate_parameters_a1m.py \n'
    message2 = 'End generated code'

    # First write the pxi file
    f = './parameters_micro.pxi'
    fh = open(f, 'w')
    fh.write('#' + message1)
    fh.write('\n')
    for param in parameters:
        fh.write(
            'cdef double ' + param + ' = ' + str(parameters[param]) + '\n')
    fh.write('#' + 'End Generated Code')
    fh.close()

    # Now write the C include file
    f = './Csrc/parameters_micro.h'
    fh = open(f, 'w')
    fh.write('//' + message1)
    for param in parameters:
        fh.write('#define ' + param + ' ' + str(parameters[param]) + '\n')
    fh.write('//' + message2)
    fh.close()

    print('Generated ./parameters_micro.pxi and '
          './Csrc/parameters_micro.h with the following values:')
    for param in parameters:
        print('\t' + param + ' = ' + str(parameters[param]))

    return


if __name__ == "__main__":
    main()
