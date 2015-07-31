import argparse
import json
import pprint 
from sys import exit

def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')
    args = parser.parse_args()

    case_name = args.case_name

    if case_name == 'StableBubble':
        namelist = StableBubble()
    else:
        print 'Not a vaild case name'
        sys.exit()


    write_file(namelist)


def StableBubble():

    namelist = {} 

    namelist["grid"] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 256
    namelist['grid']['ny'] = 7
    namelist['grid']['nz'] = 32
    namelist['grid']['gw'] = 7
    namelist['grid']['dx'] = 200.0
    namelist['grid']['dy'] = 200.0
    namelist['grid']['dz'] = 200.0

    namelist["mpi"] = {}
    namelist["mpi"]["nprocx"] = 1  
    namelist["mpi"]["nprocy"] = 1
    namelist["mpi"]["nprocz"] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3  

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist["sgs"] = {}

    namelist['meta'] = {}
    namelist['meta']['casename'] = 'StableBubble'



    return namelist

def write_file(namelist):

    try:
        type(namelist['meta']['casename'])
    except:
        print("Casename not specified in namelist dictionary!")
        print("FatalError")
        exit()


    fh = open(namelist['meta']['casename']+".in","w")

    pprint.pprint(namelist) 
    json.dump(namelist,fh,sort_keys = True, indent = 4)
    fh.close()

    return


if __name__ == "__main__":
    main()
