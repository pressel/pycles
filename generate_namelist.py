import argparse
import json
import pprint
from sys import exit
import uuid


def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')
    args = parser.parse_args()

    case_name = args.case_name

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
    elif case_name == 'Soares':
        namelist = Soares()
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
    namelist['grid']['gw'] = 7
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
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

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
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','buoyancy_frequency','viscosity']

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
    namelist['grid']['gw'] = 5
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
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'None'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 100.0

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
    namelist['grid']['gw'] = 7
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
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'None'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 100.0

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
    namelist['grid']['gw'] = 7
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
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

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
    namelist['stats_io']['auxiliary'] = ['Cumulus']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex'
    namelist['meta']['casename'] = 'Bomex'

    return namelist


def Gabls():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 64
    namelist['grid']['gw'] = 7
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
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

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
    namelist['grid']['gw'] = 5
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
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 4.0 * 3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = False
    namelist['microphysics']['ccn'] = 100.0e6

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

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
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 10.0

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
    namelist['grid']['gw'] = 5
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

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

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
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 10.0

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
    namelist['grid']['gw'] = 5
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
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7


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
    namelist['grid']['gw'] = 7
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
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7
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
    namelist['stats_io']['frequency'] = 100.0

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

    return namelist



def Soares():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    # Soares (2004): domain size = 6400 x 6400 m, domain height = 3000 (?) m; dx = ?, dy = ?, dz = 20 m
    # Nieuwstadt: domain size = ?, domain height = 2400m; dx = dy = 60 m, dz = 50-60 m
    # IOP Paper, old code: domain size = 6400 x 6400 m, domain height = 3750 m
    namelist['grid']['nx'] = 256    # IOP
    namelist['grid']['ny'] = 256    # IOP
    namelist['grid']['nz'] = 150    # IOP
    namelist['grid']['gw'] = 3      # for 2nd order
    namelist['grid']['dx'] = 25.0   # IOP
    namelist['grid']['dy'] = 25.0   # IOP
    namelist['grid']['dz'] = 25.0   # IOP

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3    # seems to be 3 in all cases???
    namelist['time_stepping']['cfl_limit'] = 0.3    # default: 0.7; IOP: 0.3
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 6*3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'       # seems to be constant in all cases and NOWHERE called???

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'     # ???
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'  # seems to be this in all cases???

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False      # seems to be set to False for all cases???

    # 2 = second_order_m
    # 32 = second_order_ml_m
    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 2
    # 2 = second_order_a
    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 2

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'  # no more 'DampingToDomainMean' ???
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 800.0  # ??? depth of damping layer?

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 900.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','viscosity']   # defines diagnostic variable output fields (progn. variables output in restart files?!)

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Soares'
    namelist['meta']['casename'] = 'Soares'

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
