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

# 1. Eddy Fields should be computed from 3D output fields with: EddyField_output.py
# 2. this field computes the correlations

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
    field_name = 'eddy_field_' + args.time + '.nc'
    path_fields = os.path.join(args.dir, field_name)
    
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

    print('reading in eddy fields: ', path_fields)
    var_name = 'u'
#    u_profile = read_in_netcdf_profile(var_name+'_mean',"profiles",path_profiles)
    u_field = read_in_netcdf_fields(var_name+'_eddy',path_fields)
    var_name = 'v'
#    v_profile = read_in_netcdf_profile(var_name+'_mean',"profiles",path_profiles)
    v_field = read_in_netcdf_fields(var_name+'_eddy',path_fields)
    var_name = 'w'
#    w_profile = read_in_netcdf_profile(var_name+'_mean',"profiles",path_profiles)
    w_field = read_in_netcdf_fields(var_name+'_eddy',path_fields)
    var_name = 'phi'
#    phi_profile = read_in_netcdf_profile(var_name+'_mean',"profiles",path_profiles)
    phi_field = read_in_netcdf_fields(var_name+'_eddy',path_fields)
    
#    print('profile: ', u_profile.shape)
    print('field: ', u_field.shape)

    # (3) read in grid dimensions
    ni_ = np.zeros((3,))
    global n, ntot
    n = np.zeros((3,))
    n = n.astype(int)
    ni_[0] = nml['grid']['nx']
    ni_[1] = nml['grid']['ny']
    ni_[2] = nml['grid']['nz']
    for i in range(3):
        n[i] = u_field.shape[i]
        if n[i] != ni_[i]:
            print('Dimensions do not fit!')
            sys.exit()
#    if n[2] != u_profile.size:
#        print('Dimensions profile vs. field do not fit!')
#        print('nz:',n[2],'field:',u_field.shape[2],'profile:',u_profile.size)
#        sys.exit()
    ntot = n[0]*n[1]*n[2]


    # (4) compute Correlations
    sh = u_field.shape
    uphi = np.zeros(shape=sh)
    vphi = np.zeros(shape=sh)
    wphi = np.zeros(shape=sh)

    for i in range(n[0]):
        if np.mod(i,50) == 0:
            print(i)
        for j in range(n[1]):
            for k in range(n[2]):
                uphi[i,j,k] = u_field[i,j,k]*phi_field[i,j,k]
                vphi[i,j,k] = v_field[i,j,k]*phi_field[i,j,k]
                wphi[i,j,k] = w_field[i,j,k]*phi_field[i,j,k]
    print('uphi:', np.amax(np.abs(u_field)), np.amax(np.abs(phi_field)), np.amax(np.abs(uphi)))
    print('vphi:', np.amax(np.abs(v_field)), np.amax(np.abs(phi_field)), np.amax(np.abs(vphi)))
    print('wphi:', np.amax(np.abs(w_field)), np.amax(np.abs(phi_field)), np.amax(np.abs(wphi)))

    # (5) IO
    # (a) create file for eddy fields
    out_path = args.dir
    nc_file_name = 'correlations_' + args.time
    print(out_path)
    create_fields_file(out_path,nc_file_name)

    # (b) dump correlation fields
    #    add_field(os.path.join(out_path,nc_file_name+'.nc'), var_name)
    dump_variables(os.path.join(out_path,nc_file_name+'.nc'), 'uphi', uphi)
    dump_variables(os.path.join(out_path,nc_file_name+'.nc'), 'vphi', vphi)
    dump_variables(os.path.join(out_path,nc_file_name+'.nc'), 'wphi', wphi)


    return




# ____________________

def create_fields_file(path,file_name):
    print('create field:', path)
    rootgrp = nc.Dataset(path+file_name+'.nc', 'w', format='NETCDF4')
    dimgrp = rootgrp.createGroup('dims')
    fieldgrp = rootgrp.createGroup('fields')
    fieldgrp.createDimension('n', n[0]*n[1]*n[2])
    fieldgrp.createDimension('nx', n[0])
    fieldgrp.createDimension('ny', n[1])
    fieldgrp.createDimension('nz', n[2])
    rootgrp.close()
    print('create field end')
    return

def dump_variables(path, var_name, var):
    print('dump variables', path, var_name, var.shape)
    data = np.empty((n[0],n[1],n[2]),dtype=np.double,order='c')
    #        double[:] data = np.empty((Gr.dims.npl,), dtype=np.double, order='c')
    add_field(path, var_name)
    for i in range(0, n[0]):
        for j in range(0, n[1]):
            for k in range(0, n[2]):
                data[i,j,k] = var[i,j,k]
    write_field(path,var_name, data)
    return

def add_field(path, var_name):
    print('add field: ', var_name)
    rootgrp = nc.Dataset(path, 'r+', format='NETCDF4')
    fieldgrp = rootgrp.groups['fields']
    #    fieldgrp.createVariable(var_name, 'f8', ('n'))
    var = fieldgrp.createVariable(var_name, 'f8', ('nx', 'ny', 'nz'))
    rootgrp.close()
    return

def write_field(path, var_name, data):
    print('write field:', path, var_name, data.shape)
    rootgrp = nc.Dataset(path, 'r+', format='NETCDF4')
    fieldgrp = rootgrp.groups['fields']
    var = fieldgrp.variables[var_name]
    #    var[:] = np.array(data)
    var[:, :, :] = data
    rootgrp.close()
    return


# ----------------------------------
def read_in_netcdf_fields(variable_name, fullpath_in):
#    print(fullpath_in)
    rootgrp = nc.Dataset(fullpath_in, 'r')
    var = rootgrp.groups['fields'].variables[variable_name]
    shape = var.shape
    data = np.ndarray(shape = var.shape)
    data = var[:]
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


def setup_stats_file(path):
    #    path = 'test_field/'
    root_grp = nc.Dataset(path, 'w', format='NETCDF4')
    
    # Set profile dimensions
    profile_grp = root_grp.createGroup('profiles')
    profile_grp.createDimension('z', Gr.dims.n[2])
    profile_grp.createDimension('t', None)
    z = profile_grp.createVariable('z', 'f8', ('z'))
    z[:] = np.array(Gr.z[Gr.dims.gw:-Gr.dims.gw])
    z_half = profile_grp.createVariable('z_half', 'f8', ('z'))
    z_half[:] = np.array(Gr.z_half[Gr.dims.gw:-Gr.dims.gw])
    profile_grp.createVariable('t', 'f8', ('t'))
    del z
    del z_half
    
    reference_grp = root_grp.createGroup('reference')
    reference_grp.createDimension('z', Gr.dims.n[2])
    z = reference_grp.createVariable('z', 'f8', ('z'))
    z[:] = np.array(Gr.z[Gr.dims.gw:-Gr.dims.gw])
    z_half = reference_grp.createVariable('z_half', 'f8', ('z'))
    z_half[:] = np.array(Gr.z_half[Gr.dims.gw:-Gr.dims.gw])
    del z
    del z_half
    
    ts_grp = root_grp.createGroup('timeseries')
    ts_grp.createDimension('t', None)
    ts_grp.createVariable('t', 'f8', ('t'))
    
    root_grp.close()
    return



if __name__ == "__main__":
    main()