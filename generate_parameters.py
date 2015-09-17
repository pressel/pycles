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
    parameters['kappa'] = parameters['Rd'] / parameters['cpd']
    parameters['Tf'] = 273.15
    parameters['Tt'] = 273.16
    parameters['T_tilde'] = 298.15
    parameters['p_tilde'] = 10.0**5
    parameters['pv_star_t'] = 611.7
    parameters['sd_tilde'] = 6864.8
    parameters['sv_tilde'] = 10513.6
    parameters['vkb'] = 0.35     # Von Karman constant from Businger 1971 used by Byun surface formulation

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
