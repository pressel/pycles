import json 
import pprint 
from sys import exit
def main():

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
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist["sgs"] = {}

    namelist['meta'] = {}
    namelist['meta']['casename'] = 'StableBubble'

    write_file(namelist)

    return

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
