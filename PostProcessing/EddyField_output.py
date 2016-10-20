# (0) chose time
# (1) import Fields
#     --> load field data for a advected variable phi and all velocities u,v,w
# (2) import Statistical File (nc-file)
#     (mean Profiles --> horizontal domain mean)
#     --> load mean-profile for advected variable phi and all velocities u,v,w
#     --> chose array[var,z] at time t
# (3) phi' = phi - mean[phi], u' = ... etc.
import netCDF4 as nc
import argparse
import os
import numpy as np
import json as  simplejson

def main():
    global case
    case = 'DCBLSoares'
    
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("dir")
    parser.add_argument("time")
    args = parser.parse_args()
    print(args.dir, args.time)
    global time
    time = np.int(args.time)
    
    # (0) import Namelist --> to chose right mean profile, fitting with time
    nml_name = case + '.in'
    path_nml = os.path.join(args.dir,nml_name)
    nml = simplejson.loads(open(args.dir + case + '.in').read())
    dt = nml['stats_io']['frequency']
    dz = nml['grid']['dz']
    print('dt:', dt, 'dz:', dz)
    
    
    
    # (1) define time index of profile
    # (2) import fields & mean profiles
    # time: array with all output times of profile statistics
    # nt: index of profile at time=args.time (same time as fields)
    # u_profile, v_profile, w_profile, phi_profile: mean profiles of resp. variable at time t
    
    file_name = 'Stats.' + case + '.nc'
    path_profiles = os.path.join(args.dir,file_name)
    field_name = args.time + '.nc'
    path_fields = os.path.join(args.dir, 'fields', field_name)
    
    var_name = 't'
    global nt
    time_series = read_in_netcdf_profile_all(var_name,"timeseries",path_profiles)
    print('time[0]', time_series[0], np.int(args.time))
    if time_series[0] == 0:
        nt = np.int(args.time) / dt + 1
        if np.mod(nt,1) > 0:
            print(nt)
            sys.exit()
        else:
            nt = np.int(nt)
        print('nt', nt)
    else:
        print('profiles do not start at zero')
        sys.exit()

    var_name = 'u'
    u_profile = read_in_netcdf_profile(var_name+'_mean',"profiles",path_profiles)
    u_field = read_in_netcdf_fields(var_name,path_fields)
    var_name = 'v'
    v_profile = read_in_netcdf_profile(var_name+'_mean',"profiles",path_profiles)
    v_field = read_in_netcdf_fields(var_name,path_fields)
    var_name = 'w'
    w_profile = read_in_netcdf_profile(var_name+'_mean',"profiles",path_profiles)
    w_field = read_in_netcdf_fields(var_name,path_fields)
    var_name = 'phi'
    phi_profile = read_in_netcdf_profile(var_name+'_mean',"profiles",path_profiles)
    phi_field = read_in_netcdf_fields(var_name,path_fields)
    
    print('profile: ', u_profile.shape)
    print('field: ', u_field.shape)

    # (3) read in grid dimensions
    ni_ = np.zeros((3,))
    global ni, ntot
    ni = np.zeros((3,))
    ni = ni.astype(int)
    ni_[0] = nml['grid']['nx']
    ni_[1] = nml['grid']['ny']
    ni_[2] = nml['grid']['nz']
    for i in range(3):
        ni[i] = u_field.shape[i]
        if ni[i] != ni_[i]:
            print('Dimensions do not fit!')
            sys.exit()
    if ni[2] != u_profile.size:
        print('Dimensions profile vs. field do not fit!')
        print('nz:',ni[2],'field:',u_field.shape[2],'profile:',u_profile.size)
        sys.exit()
    ntot = ni[0]*ni[1]*ni[2]


    # (4) compute eddy fields
    sh = u_field.shape
    u_eddy = np.zeros(shape=sh)
    v_eddy = np.zeros(shape=sh)
    w_eddy = np.zeros(shape=sh)
    phi_eddy = np.zeros(shape=sh)

    for i in range(ni[0]):
        if np.mod(i,50) == 0:
            print(i)
        for j in range(ni[1]):
            for k in range(ni[2]):
                u_eddy[i,j,k] = u_field[i,j,k] - u_profile[k]

    # (5) IO
    # (a) create file for eddy fields
    out_path = args.dir
    nc_file_name = 'eddy_field_' + args.time
    print(out_path)
    create_fields_file(out_path,nc_file_name)

    # (b) dump eddy fields
    #    add_field(os.path.join(out_path,nc_file_name+'.nc'), var_name)
    dump_variables(os.path.join(out_path,nc_file_name+'.nc'), 'u_eddy', u_eddy)
    dump_variables(os.path.join(out_path,nc_file_name+'.nc'), 'v_eddy', v_eddy)
    dump_variables(os.path.join(out_path,nc_file_name+'.nc'), 'w_eddy', w_eddy)
    dump_variables(os.path.join(out_path,nc_file_name+'.nc'), 'phi_eddy', phi_eddy)

    return




# ____________________

def create_fields_file(path,file_name):
    print('create field:', path)
    rootgrp = nc.Dataset(path+file_name+'.nc', 'w', format='NETCDF4')
    dimgrp = rootgrp.createGroup('dims')
    fieldgrp = rootgrp.createGroup('fields')
    fieldgrp.createDimension('n', ni[0]*ni[1]*ni[2])
    
    dimgrp.createDimension('d1', 1)
    
    n_0 = dimgrp.createVariable('n_0', 'i4', ('d1'))
    n_1 = dimgrp.createVariable('n_1', 'i4', ('d1'))
    n_2 = dimgrp.createVariable('n_2', 'i4', ('d1'))
    ng = dimgrp.createVariable('ng', 'i4', ('d1'))

    n_0[:] = ni[0]
    n_1[:] = ni[1]
    n_2[:] = ni[2]

    rootgrp.close()
    print('create field end')
    return

def dump_variables(path, var_name, var):
    print('dump variables', path, var_name, var.shape)
    istride = ni[1] * ni[2]
    jstride = ni[2]
    data = np.empty((ntot,),dtype=np.double,order='c')
    #        double[:] data = np.empty((Gr.dims.npl,), dtype=np.double, order='c')
    add_field(path, var_name)
    count = 0
    for i in range(0, ni[0]):
        ishift = i * istride
        for j in range(0, ni[1]):
            jshift = j * jstride
            for k in range(0, ni[2]):
                ijk = ishift + jshift + k
                data[count] = var[i,j,k]
                count += 1
    write_field(path,var_name, data)
    return

def add_field(path, var_name):
    print('add field: ', var_name)
    rootgrp = nc.Dataset(path, 'r+', format='NETCDF4')
    fieldgrp = rootgrp.groups['fields']
    fieldgrp.createVariable(var_name, 'f8', ('n'))
    rootgrp.close()
    return

def write_field(path, var_name, data):
    print('write field:', path, var_name)
    rootgrp = nc.Dataset(path, 'r+', format='NETCDF4')
    fieldgrp = rootgrp.groups['fields']
    var = fieldgrp.variables[var_name]
    var[:] = np.array(data)
    rootgrp.close()
    return


# ----------------------------------
def read_in_netcdf_fields(variable_name, fullpath_in):
    rootgrp = nc.Dataset(fullpath_in, 'r')
    var = rootgrp.groups['fields'].variables[variable_name]
    shape = var.shape
    data = np.ndarray(shape = var.shape)
    rootgrp.close()
    return data


def read_in_netcdf_profile_all(variable_name, group_name, fullpath_in):
#    print(fullpath_in)
    rootgrp = nc.Dataset(fullpath_in, 'r')
    var = rootgrp.groups[group_name].variables[variable_name]
    shape = var.shape
    data = np.ndarray(shape = var.shape)
    if group_name != 'profiles':
        var = rootgrp.groups[group_name].variables[variable_name]
    for t in range(shape[0]):
        if group_name == "profiles":
            data[t,:] = var[t, :]
            nkr = rootgrp.groups['profiles'].variables['z'].shape[0]
        if group_name == "correlations":
            data[t,:] = var[t, :]
        if group_name == "timeseries":
            data[t] = var[t]
    rootgrp.close()
    return data


def read_in_netcdf_profile(variable_name, group_name, fullpath_in):
    rootgrp = nc.Dataset(fullpath_in, 'r')
    var = rootgrp.groups[group_name].variables[variable_name]
    shape = var.shape
    #print('read_in_profile: ', time, var.shape, nt, type(nt))
    
    if group_name == "profiles":
        data = np.ndarray((shape[1],))
        data[:] = var[nt, :]
    if group_name == "correlations":
        data = np.ndarray((shape[1],))
        data[:] = var[nt, :]
    if group_name == "timeseries":
        data = var[nt]
    rootgrp.close()
    return data

# ----------------------------------



if __name__ == "__main__":
    main()