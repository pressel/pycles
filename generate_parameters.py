import numpy as np
from collections import OrderedDict


def main():

    #########################
    # Users should modify here
    #########################

    parameters = OrderedDict()

    parameters['pi'] = np.pi
    parameters['g'] = 9.80665
    parameters['Rd'] = 287.1
    parameters['Rv'] = 461.5
    parameters['eps_v'] = parameters['Rd'] / parameters['Rv']
    parameters['eps_vi'] = 1.0 / parameters['eps_v']
    parameters['cpd'] = 1004.0
    parameters['cpv'] = 1859.0
    parameters['cl'] = 4218.0
    parameters['ci'] = 2106.0
    parameters['kappa'] = parameters['Rd'] / parameters['cpd']
    parameters['Tf'] = 273.15
    parameters['Tt'] = 273.16
    parameters['T_tilde'] = 298.15
    parameters['p_tilde'] = 10.0**5
    parameters['pv_star_t'] = 611.7
    parameters['sd_tilde'] = 6864.8
    parameters['sv_tilde'] = 10513.6
    parameters['omega'] = 7.2921151467064e-5  # Earth's rotational rate (http://hpiers.obspm.fr/eop-pc/models/constants.html)
    parameters['ql_threshold'] = 1.0e-8



    # Surface Monin-Obukhov related parameters
    parameters['vkb'] = 0.35     # Von Karman constant from Businger 1971 used by Byun surface formulation
    parameters['Pr0'] = 0.74 
    parameters['beta_m'] = 4.7 
    parameters['beta_h'] = parameters['beta_m']/parameters['Pr0'] 
    parameters['gamma_m'] = 15.0
    parameters['gamma_h'] = 9.0


    # Surface Monin-Obukhov related parameters
    parameters['vkb'] = 0.35     # Von Karman constant from Businger 1971 used by Byun surface formulation
    parameters['Pr0'] = 0.74
    parameters['beta_m'] = 4.7
    parameters['beta_h'] = parameters['beta_m']/parameters['Pr0']
    parameters['gamma_m'] = 15.0
    parameters['gamma_h'] = 9.0

    # if GABLS use these values:
    # parameters['vkb'] = 0.4
    # parameters['Pr0'] = 1.0
    # parameters['beta_m'] = 4.8
    # parameters['beta_h'] = 7.8


    #############################
    # Users shouldn't modify below
    #############################

    # Some warning to put in the generated code
    message1 = 'Generated code! Absolutely DO NOT modify this file, ' \
               'parameters should be modified in generate_parameters.py \n'
    message2 = 'End generated code'

    # First write the pxi file
    f = './parameters.pxi'
    fh = open(f, 'w')
    fh.write('#' + message1)
    fh.write('\n')
    for param in parameters:
        fh.write(
            'cdef double ' + param + ' = ' + str(parameters[param]) + '\n')
    fh.write('#' + 'End Generated Code')
    fh.close()

    # Now write the C include file
    f = './Csrc/parameters.h'
    fh = open(f, 'w')
    fh.write('//' + message1)
    for param in parameters:
        fh.write('#define ' + param + ' ' + str(parameters[param]) + '\n')
    fh.write('//' + message2)
    fh.close()

    print('Generated ./parameters.pxi and '
          './Csrc/parameters.h with the following values:')
    for param in parameters:
        print('\t' + param + ' = ' + str(parameters[param]))

    return


if __name__ == "__main__":
    main()
