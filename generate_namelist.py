import argparse
import json
import pprint
from sys import exit
import uuid
import ast


def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')
    # Optional Arguments for CGILS
    parser.add_argument('--perturbed_temperature', default='False',
                        help='Specify if perturbed temperature case is to be run (CGILS) as True/False')
    parser.add_argument('--control_subsidence', default='False',
                        help='Specify if control subsidence is to be used in perturbed runs (CGILS) as True/False')
    parser.add_argument('--zgils_location', default='False',
                        help='specify location (6/11/12)')

    args = parser.parse_args()

    case_name = args.case_name

    #Optional Arguments for CGILS
    is_p2 = ast.literal_eval(args.perturbed_temperature)
    is_ctl_omega = ast.literal_eval(args.control_subsidence)
    zgils_loc = ast.literal_eval(args.zgils_location)
    print(zgils_loc)

    if case_name == 'StableBubble':
        namelist = StableBubble()
    elif case_name == 'SaturatedBubble':
        namelist = SaturatedBubble()
    elif case_name == 'SullivanPatton':
        namelist = SullivanPatton()
    elif case_name == 'Bomex':
        namelist = Bomex()
    elif case_name == 'Gabls':
        namelist = Gabls()
    elif case_name == 'DYCOMS_RF01':
        namelist = DYCOMS_RF01()
    elif case_name == 'DYCOMS_RF02':
        namelist = DYCOMS_RF02()
    elif case_name == 'SMOKE':
        namelist = SMOKE()
    elif case_name == 'Rico':
        namelist = Rico()
    elif case_name == 'Isdac':
        namelist = Isdac()
    elif case_name == 'IsdacCC':
        namelist = IsdacCC()
    elif case_name == 'Mpace':
        namelist = Mpace()
    elif case_name == 'Sheba':
        namelist = Sheba()
    elif case_name == 'CGILS_S6':
        namelist = CGILS_S6(is_p2, is_ctl_omega)
    elif case_name == 'CGILS_S11':
        namelist = CGILS_S11(is_p2, is_ctl_omega)
    elif case_name == 'CGILS_S12':
        namelist = CGILS_S12(is_p2, is_ctl_omega)
    elif case_name == 'ZGILS':
        namelist = ZGILS(zgils_loc)
    else:
        print('Not a vaild case name')
        exit()

    write_file(namelist)


def SullivanPatton():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 32
    namelist['grid']['ny'] = 32
    namelist['grid']['nz'] = 32
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 160.0
    namelist['grid']['dy'] = 160.0
    namelist['grid']['dz'] = 64.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 7200.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['TKE']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'SullivanPatton'
    namelist['meta']['casename'] = 'SullivanPatton'

    return namelist


def SaturatedBubble():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 100
    namelist['grid']['ny'] = 5
    namelist['grid']['nz'] = 50
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 200.0
    namelist['grid']['dy'] = 200.0
    namelist['grid']['dz'] = 200.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.3
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 1000.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'UniformViscosity'
    namelist['sgs']['UniformViscosity'] = {}
    namelist['sgs']['UniformViscosity']['viscosity'] = 0.0
    namelist['sgs']['UniformViscosity']['diffusivity'] = 0.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'None'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] = {}

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 100.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['meta'] = {}
    namelist['meta']['casename'] = 'SaturatedBubble'
    namelist['meta']['simname'] = 'SaturatedBubble'

    return namelist


def StableBubble():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 512
    namelist['grid']['ny'] = 7
    namelist['grid']['nz'] = 64
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 100.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 1000.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'UniformViscosity'
    namelist['sgs']['UniformViscosity'] = {}
    namelist['sgs']['UniformViscosity']['viscosity'] = 75.0
    namelist['sgs']['UniformViscosity']['diffusivity'] = 75.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'None'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] = {} 

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 100.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','buoyancy_frequency']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'StableBubble'
    namelist['meta']['casename'] = 'StableBubble'

    return namelist


def Bomex():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 100 / 2.5

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 21600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 600

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus','TKE']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex'
    namelist['meta']['casename'] = 'Bomex'

    namelist['initialization'] = {}
    namelist['initialization']['random_seed_factor'] = 1

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'PurityTracers'

    return namelist


def Gabls():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 64
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 6.25
    namelist['grid']['dy'] = 6.25
    namelist['grid']['dz'] = 6.25

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] =1.0
    namelist['time_stepping']['dt_max'] = 2.0
    namelist['time_stepping']['t_max'] = 43200.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['cs'] = 0.17
    namelist['sgs']['Smagorinsky']['prt'] = 1.0/3.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 100.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['StableBL']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Gabls'
    namelist['meta']['casename'] = 'Gabls'

    return namelist

def DYCOMS_RF01():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 96
    namelist['grid']['ny'] = 96
    namelist['grid']['nz'] = 300
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 35.0
    namelist['grid']['dy'] = 35.0
    namelist['grid']['dz'] = 5.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 4.0
    namelist['time_stepping']['t_max'] = 4.0 * 3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = False
    namelist['microphysics']['ccn'] = 100.0e6

    namelist['radiation'] = {}
    namelist['radiation']['use_RRTM'] = True
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 60.0

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {}
    namelist['sgs']['Smagorinsky']['iles'] = True
    
    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.002
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['DYCOMS', 'Flux']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 3600.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 1e6

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'DYCOMS_RF01'
    namelist['meta']['casename'] = 'DYCOMS_RF01'

    return namelist

def DYCOMS_RF02():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 128
    namelist['grid']['ny'] = 128
    namelist['grid']['nz'] = 300
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 5.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 6.0 * 3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 55.0e6

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {} 
    namelist['sgs']['Smagorinsky']['iles'] = True

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.002
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['DYCOMS', 'Flux']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 3600.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency']

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 1e6

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    namelist['meta'] = {}
    namelist['meta']['simname'] = 'DYCOMS_RF02'
    namelist['meta']['casename'] = 'DYCOMS_RF02'

    return namelist

def SMOKE():

    '''
    Namelist generator for the smoke cloud case:
    Bretherton, C. S., and coauthors, 1999:
    An intercomparison of radiatively- driven entrainment and turbulence in a smoke cloud,
    as simulated by different numerical models. Quart. J. Roy. Meteor. Soc., 125, 391-423. Full text copy.
    :return:
    '''


    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 50
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 25.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 4.0 * 3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5


    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.002
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['SMOKE']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 3600.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'SMOKE'
    namelist['meta']['casename'] = 'SMOKE'

    return namelist

def Rico():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 128
    namelist['grid']['ny'] = 128
    namelist['grid']['nz'] = 150
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 40.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = False
    namelist['microphysics']['ccn'] = 70.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}

    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 800

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Rico'
    namelist['meta']['casename'] = 'Rico'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist


def Isdac():

    namelist = {}

    namelist["grid"] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 250
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 10.0

    namelist["mpi"] = {}
    namelist["mpi"]["nprocx"] = 1
    namelist["mpi"]["nprocy"] = 1
    namelist["mpi"]["nprocz"] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.5
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 8.0


    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'Arctic_1M'
    namelist['microphysics']['phase_partitioning'] = 'Arctic'
    namelist['microphysics']['n0_ice'] = 1.0e7

    namelist["sgs"] = {}
    namelist["sgs"]['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {}
    namelist['sgs']['Smagorinsky']['iles'] = True

    namelist['radiation'] = {}
    namelist['radiation']['use_RRTM'] = False
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 60.0
    namelist['radiation']['RRTM']['buffer_points'] = 15
    namelist['radiation']['RRTM']['patch_pressure'] = 600.0*100.0
    namelist['radiation']['RRTM']['adjes'] = 0.0

    namelist["diffusion"] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 600

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = "stats"
    namelist['stats_io']['auxiliary'] = 'None'
    namelist['stats_io']['frequency'] = 30.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = "fields"
    namelist['fields_io']['frequency'] = 36000.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Isdac'
    namelist['meta']['casename'] = 'Isdac'

    return namelist


def IsdacCC():

    namelist = {}

    namelist["grid"] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 250
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 10.0

    namelist["mpi"] = {}
    namelist["mpi"]["nprocx"] = 1
    namelist["mpi"]["nprocy"] = 1
    namelist["mpi"]["nprocz"] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.5
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 8.0

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'Arctic_1M'
    namelist['microphysics']['phase_partitioning'] = 'Arctic'
    namelist['microphysics']['n0_ice'] = 1.0e7

    namelist['sgs'] = {}
    namelist["sgs"]['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {}
    namelist['sgs']['Smagorinsky']['iles'] = True

    namelist["diffusion"] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 600

    namelist['initial'] = {}
    namelist['initial']['SST'] = 265.0 #initial surface temperature
    namelist['initial']['dTi'] = 7.0 #temperature jump at the inversion
    namelist['initial']['rh0'] = 0.8 #Surface relative humidity
    namelist['initial']['gamma'] = 5.0/1000. #free tropospheric lapse rate
    namelist['initial']['rh'] = 0.6 #free tropospheric relative humidity
    namelist['initial']['z_top'] = 820.0 #top of mixed layer
    namelist['initial']['dzi'] = 30.0 #inversion height
    namelist['initial']['dSST'] = 8.0 #SST change (climate change)
    namelist['initial']['divergence'] = 5.0e-6 # LS divergence
    namelist['initial']['fix_dqt'] = True

    namelist['surface'] = {}
    namelist['surface']['sensible'] = 0.0 #surface sensible heat flux Wm-2

    namelist['radiation'] = {}
    namelist['radiation']['use_RRTM'] = True
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 60.0
    namelist['radiation']['RRTM']['buffer_points'] = 15
    namelist['radiation']['RRTM']['patch_pressure'] = 600.0*100.0
    namelist['radiation']['RRTM']['adjes'] = 0.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = "stats"
    namelist['stats_io']['auxiliary'] = 'None'
    namelist['stats_io']['frequency'] = 30.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = "fields"
    namelist['fields_io']['frequency'] = 36000.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'IsdacCC'
    namelist['meta']['casename'] = 'IsdacCC'

    return namelist


def Mpace():

    namelist = {}

    namelist["grid"] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 250
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 10.0

    namelist["mpi"] = {}
    namelist["mpi"]["nprocx"] = 1
    namelist["mpi"]["nprocy"] = 1
    namelist["mpi"]["nprocz"] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.5
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 12.0

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'Arctic_1M'
    namelist['microphysics']['phase_partitioning'] = 'Arctic'
    namelist['microphysics']['n0_ice'] = 1.0e7

    namelist["sgs"] = {}
    namelist["sgs"]['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {}
    namelist['sgs']['Smagorinsky']['iles'] = True

    namelist['radiation'] = {}
    namelist['radiation']['use_RRTM'] = True
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 60.0
    namelist['radiation']['RRTM']['buffer_points'] = 15
    namelist['radiation']['RRTM']['patch_pressure'] = 600.0*100.0
    namelist['radiation']['RRTM']['dyofyr'] = 283
    namelist['radiation']['RRTM']['daily_mean_sw'] = False
    namelist['radiation']['RRTM']['hourz'] = 17.0
    namelist['radiation']['RRTM']['latitude'] = 71.75
    namelist['radiation']['RRTM']['longitude'] = 151.0

    namelist["diffusion"] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 600

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = "stats"
    namelist['stats_io']['auxiliary'] = 'None'
    namelist['stats_io']['frequency'] = 30.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = "fields"
    namelist['fields_io']['frequency'] = 36000.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Mpace'
    namelist['meta']['casename'] = 'Mpace'

    return namelist


def Sheba():

    namelist = {}

    namelist["grid"] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 250
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 10.0

    namelist["mpi"] = {}
    namelist["mpi"]["nprocx"] = 1
    namelist["mpi"]["nprocy"] = 1
    namelist["mpi"]["nprocz"] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.5
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 12.0


    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'Arctic_1M'
    namelist['microphysics']['phase_partitioning'] = 'Arctic'
    namelist['microphysics']['n0_ice'] = 1.0e7

    namelist["sgs"] = {}
    namelist["sgs"]['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {}
    namelist['sgs']['Smagorinsky']['iles'] = True

    namelist['radiation'] = {}
    namelist['radiation']['use_RRTM'] = True
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 60.0
    namelist['radiation']['RRTM']['buffer_points'] = 15
    namelist['radiation']['RRTM']['stretch_factor'] = 1.2
    namelist['radiation']['RRTM']['patch_pressure'] = 500.0*100.0
    namelist['radiation']['RRTM']['dyofyr'] = 127
    namelist['radiation']['RRTM']['daily_mean_sw'] = False
    namelist['radiation']['RRTM']['hourz'] = 12.0
    namelist['radiation']['RRTM']['latitude'] = 76.0
    namelist['radiation']['RRTM']['longitude'] = 195.0
    namelist['radiation']['RRTM']['adir'] = 0.827

    namelist["diffusion"] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 5

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 600

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = "stats"
    namelist['stats_io']['auxiliary'] = 'None'
    namelist['stats_io']['frequency'] = 30.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = "fields"
    namelist['fields_io']['frequency'] = 36000.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Sheba'
    namelist['meta']['casename'] = 'Sheba'

    return namelist



def CGILS_S6(is_p2,is_ctl_omega):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 96
    namelist['grid']['ny'] = 96
    namelist['grid']['nz'] = 180
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 30.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0*10.0 # 10 days

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'variable'


    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 600.0


    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 100.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}

    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['iles'] = False

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus']
    namelist['stats_io']['frequency'] = 5 * 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 86400.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy']

    namelist['meta'] = {}
    namelist['meta']['CGILS'] = {}
    namelist['meta']['casename'] = 'CGILS'
    namelist['meta']['CGILS']['location'] = 6
    namelist['meta']['CGILS']['P2'] = is_p2
    namelist['meta']['CGILS']['CTL_omega'] = is_ctl_omega

    simname = 'CGILS_S' + str(namelist['meta']['CGILS']['location'] )
    if namelist['meta']['CGILS']['P2']:
        if namelist['meta']['CGILS']['CTL_omega']:
            simname += '_P2'
        else:
            simname += '_P2S'
    else:
        simname += '_CTL'
    namelist['meta']['simname'] = simname



    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['delete_old'] = True
    namelist['restart']['times_retained'] = range(86400, 86400*11, 86400)

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 43200.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist







def CGILS_S11(is_p2,is_ctl_omega):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 96
    namelist['grid']['ny'] = 96
    namelist['grid']['nz'] = 180
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 20.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0*10.0 # 10 days

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'variable'

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 600.0


    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 100.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}

    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['iles'] = False

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Flux']
    namelist['stats_io']['frequency'] = 5 * 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 86400.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy']

    namelist['meta'] = {}
    namelist['meta']['CGILS'] = {}
    namelist['meta']['casename'] = 'CGILS'
    namelist['meta']['CGILS']['location'] = 11
    namelist['meta']['CGILS']['P2'] = is_p2
    namelist['meta']['CGILS']['CTL_omega'] = is_ctl_omega

    simname = 'CGILS_S' + str(namelist['meta']['CGILS']['location'] )
    if namelist['meta']['CGILS']['P2']:
        if namelist['meta']['CGILS']['CTL_omega']:
            simname += '_P2'
        else:
            simname += '_P2S'
    else:
        simname += '_CTL'
    namelist['meta']['simname'] = simname



    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['delete_old'] = True
    namelist['restart']['times_retained'] = range(86400, 86400*11, 86400)

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 43200.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist





def CGILS_S12(is_p2,is_ctl_omega):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 96
    namelist['grid']['ny'] = 96
    namelist['grid']['nz'] = 200
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 25.0
    namelist['grid']['dy'] = 25.0
    namelist['grid']['dz'] = 10.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0*10.0 # 10 days

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'variable'


    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 100.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}

    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['iles'] = False

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Flux']
    namelist['stats_io']['frequency'] = 5 * 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 86400.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy']

    namelist['meta'] = {}
    namelist['meta']['CGILS'] = {}
    namelist['meta']['casename'] = 'CGILS'
    namelist['meta']['CGILS']['location'] = 12
    namelist['meta']['CGILS']['P2'] = is_p2
    namelist['meta']['CGILS']['CTL_omega'] = is_ctl_omega

    simname = 'CGILS_S' + str(namelist['meta']['CGILS']['location'] )
    if namelist['meta']['CGILS']['P2']:
        if namelist['meta']['CGILS']['CTL_omega']:
            simname += '_P2'
        else:
            simname += '_P2S'
    else:
        simname += '_CTL'
    namelist['meta']['simname'] = simname



    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['delete_old'] = True
    namelist['restart']['times_retained'] = range(86400, 86400*11, 86400)

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 43200.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist




def ZGILS(zgils_loc):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 86
    namelist['grid']['ny'] = 86
    namelist['grid']['nz'] = 216
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 75.0
    namelist['grid']['dy'] = 75.0
    namelist['grid']['dz'] = 20.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0*20.0 # 20 days

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'variable'


    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 100.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}
    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['iles'] = False


    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['surface_budget'] = {}
    if zgils_loc == 12:
        namelist['surface_budget']['ocean_heat_flux'] = 70.0
    elif zgils_loc == 11:
        namelist['surface_budget']['ocean_heat_flux'] = 90.0
    elif zgils_loc == 6:
        namelist['surface_budget']['ocean_heat_flux'] = 60.0

    # To run a fixed_sst case set fixed_sst_time > t_max of simulation
    namelist['surface_budget']['fixed_sst_time'] = 24.0 * 3600.0 * 30.0 # 3 days spinup

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Flux']
    namelist['stats_io']['frequency'] = 5 * 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 86400.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy']

    namelist['meta'] = {}
    namelist['meta']['ZGILS'] = {}
    namelist['meta']['casename'] = 'ZGILS'
    namelist['meta']['ZGILS']['location'] = zgils_loc


    simname = 'ZGILS_S' + str(namelist['meta']['ZGILS']['location'] )
    namelist['meta']['simname'] = simname



    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['delete_old'] = True
    namelist['restart']['times_retained'] = range(86400, 86400*21, 86400)

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 43200.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist


def write_file(namelist):

    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())

    fh = open(namelist['meta']['simname'] + '.in', 'w')
    pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return


if __name__ == '__main__':
    main()
