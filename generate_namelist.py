import pickle
from collections import OrderedDict
from sys import exit
def main():

    namelist = OrderedDict()

    namelist["grid"] = OrderedDict()
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 256
    namelist['grid']['ny'] = 7
    namelist['grid']['nz'] = 32
    namelist['grid']['gw'] = 7
    namelist['grid']['dx'] = 200.0
    namelist['grid']['dy'] = 200.0
    namelist['grid']['dz'] = 200.0

    namelist["mpi"] = OrderedDict()
    namelist["mpi"]["nprocx"] = 1  
    namelist["mpi"]["nprocy"] = 1
    namelist["mpi"]["nprocz"] = 1

    namelist['time_stepping'] = OrderedDict()
    namelist['time_stepping']['ts_type'] = 3  

    namelist['thermodynamics'] = OrderedDict()
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = OrderedDict()
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist["sgs"] = OrderedDict()

    namelist['meta'] = OrderedDict()
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


    fh = open(namelist['meta']['casename']+".in","wb")

    #Loop over keys in name list nested dictionary and print output
    print("PyCLES Namelist Parameters")
    for key in namelist:
        print(key+":")
        for key1 in namelist[key]:
            print('\t'+key1 + ' = ' +  str(namelist[key][key1]))


    pickle.dump(namelist,fh)
    fh.close()

    return


if __name__ == "__main__":
    main()
