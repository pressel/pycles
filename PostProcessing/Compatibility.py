import netCDF4 as nc
import argparse
import os
import numpy as np
import json as  simplejson

def main():
    parser = argparse.ArgumentParser(prog='PyCLES')
    #parser.add_argument("fields_dir")
    parser.add_argument("dir")
    parser.add_argument("time1")
    parser.add_argument("time2")
    args = parser.parse_args()
    
    t = 0
    for time in [args.time1, args.time2]:
        print(time)
        path = time + '.nc'
        print('????', path)
        fullpath_in = os.path.join(args.dir,'fields',path)
        if t==0:
            var1 = read_in_netcdf('phi',fullpath_in)
        elif t==1:
            var2 = read_in_netcdf('phi',fullpath_in)
        t+=1

    print(var1.shape)
    print(var2.shape)
    
    case_name = 'DCBLSoares'
    nml = simplejson.loads(open(args.dir + case_name + '.in').read())
    dx = nml['grid']['dx']
    dy = nml['grid']['dy']
    dz = nml['grid']['dz']
    global nx, ny, nz
    nx = nml['grid']['nx']
    ny = nml['grid']['ny']
    nz = nml['grid']['nz']
    print('nx,ny,nz:', nx,ny,nz)
    global dt
    dt = np.int(args.time2) - np.int(args.time1)
    print('dt', dt)


    test_compatibility(var1, var2)
    return

#----------------------------------------------------------------------
def test_compatibility(data1,data2):
    #    global nx
    phimax = -9999.99
    phimin = 9999.99
    for i in xrange(1,nx-1):
        print(i)
        for j in xrange(1,ny-1):
            for k in xrange(1,nz-1):
                for di in xrange(-1,2):
                    for dj in xrange(-1,2):
                        for dk in xrange(-1,2):
                            a = data1[i+di,j+dj,k+dk]
                            if a<phimin:
                                phimin = a
                            elif a>max:
                                phimax = a
                if data2[i,j,k] < phimin:
                    print('problem min')
                elif data2[i,j,k] > phimax:
                    print('problem max')

    #phimin = min(data1[i+di,j+dj,k+dk])
    return

#----------------------------------------------------------------------
def read_in_netcdf(variable_name, fullpath_in):
    rootgrp = nc.Dataset(fullpath_in, 'r')
    var = rootgrp.groups['fields'].variables[variable_name]
    
    shape = var.shape
    print('shape:',var.shape)
    data = np.ndarray(shape = var.shape)
    
    rootgrp.close()
    return data

#----------------------------------------------------------------------
def read_in(variable_name, group_name, fullpath_in):
    f = File(fullpath_in)
    
    #Get access to the profiles group
    profiles_group = f[group_name]
    #Get access to the variable dataset
    variable_dataset = profiles_group[variable_name]
    #Get the current shape of the dataset
    variable_dataset_shape = variable_dataset.shape
    
    variable = np.ndarray(shape = variable_dataset_shape)
    for t in range(variable_dataset_shape[0]):
        if group_name == "timeseries":
            variable[t] = variable_dataset[t]
        elif group_name == "profiles":
            variable[t,:] = variable_dataset[t, :]
        elif group_name == "correlations":
            variable[t,:] = variable_dataset[t, :]
        elif group_name == "fields":
            variable[t] = variable_dataset[t]

    f.close()
    return variable




if __name__ == "__main__":
    main()