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
import matplotlib.cm as cm
import pylab as plt


def main():
    global case
    global nx0, ny0, nz0
    global fullpath_out
    global time
    # -----------
    case = 'DCBLSoares'
    path = 'test_field/'
    fullpath_out = path
    T = [1800]

    path_to_fields = os.path.join(path,'fields/')
    var_list_corr = ['wphi','uphi', 'vphi', 'uphi_div', 'vphi_div', 'wphi_div']
    var_list = ['phi']
    # -----------


    # (0) import Namelist --> to chose right mean profile, fitting with time
    nml = simplejson.loads(open(os.path.join(path,case + '.in')).read())
    dt = nml['stats_io']['frequency']
    dz = nml['grid']['dz']
    #print('dt:', dt, 'dz:', dz)
    
    # (1) Read in Test field
    field = read_in_netcdf_fields('wphi',os.path.join(path,'correlations_1800.nc'))
    #print('field: ', field.shape)
#    for var_name in var_list:
#        field = read_in_netcdf_fields(var_name,'test_field/eddy_fields_1800.nc')
#        print('field: ', field.shape)

    # (2) read in grid dimensions
    ni_ = np.zeros((3,))
    global n, ntot
    n = np.zeros((3,))
    n = n.astype(int)
    ni_[0] = nml['grid']['nx']
    ni_[1] = nml['grid']['ny']
    ni_[2] = nml['grid']['nz']
    for i in range(3):
        n[i] = field.shape[i]
        if n[i] != ni_[i]:
            print('Dimensions do not fit!')
            sys.exit()
    nx0 = np.floor(n[0]/2)
    ny0 = np.floor(n[1]/2)
    nz0 = np.floor(n[2]/2)
    ntot = n[0]*n[1]*n[2]


    # (4)  Visualize
    for time in T:
        path_to_correlations = os.path.join(path,'correlations_'+np.str(time)+'.nc')
        print(path_to_correlations)
        path_to_fields = os.path.join(path,'fields',np.str(time)+'.nc')
        print(path_to_fields)
        for corr_name in var_list_corr:
            #field = read_in_netcdf_fields(corr_name,'test_field/correlations_1800.nc')
            field = read_in_netcdf_fields(corr_name,path_to_correlations)
            print(corr_name, ': max = ', np.amax(np.abs(field)))
            file_name = corr_name + '_' + np.str(time)
            plot_data_vertical(field[:,ny0,:], corr_name, file_name)
        for var_name in var_list:
            field = read_in_netcdf_fields(var_name,path_to_fields)
            #field = read_in_netcdf_fields(var_name,'test_field/fields/1800.nc')
            print(var_name, ': max = ', np.amax(np.abs(field)))
            file_name = var_name + '_' + np.str(time)
            plot_data_vertical(field[:,ny0,:], var_name, file_name)

    return


# ----------------------------------
def plot_data_vertical(data, var_name, file_name):
    print('plot vertical: ', var_name, data.shape)
    plt.figure()
    plt.contourf(data.T)
    if var_name == 'phi':
        cont = np.linspace(1.0,1.1,11)
        plt.contour(data.T, levels = cont)
    # plt.show()
    plt.colorbar()
    max = np.amax(data)
    plt.title(var_name + ', max:' + np.str(np.amax(data)), fontsize=12)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(fullpath_out + file_name + '.png')
    plt.close()


def plot_data_vertical_levels(data, var_name, level):
    print(data.shape)
    plt.figure()
    plt.contourf(data.T, levels = level)
    plt.colorbar()
    plt.title(var_name + ', max:' + np.str(np.amax(data)), fontsize=12)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(fullpath_out + file_name + '.png')
    plt.close()


def plot_data_horizontal(data, var_name):
    print(data.shape)
    plt.figure()
    plt.contourf(data.T)
    # plt.show()
    plt.colorbar()
    max = np.amax(data)
    plt.title(var_name + ', max:' + np.str(np.amax(data)))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(fullpath_out + file_name + '.png')
    plt.close()


def plot_data_horizontal_levels(data, var_name, level):
    print(data.shape)
    plt.figure()
    plt.contourf(data.T, levels = level)
    # plt.show()
    plt.colorbar()
    max = np.amax(data)
    plt.title(var_name + ', max:' + np.str(np.amax(data)))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(fullpath_out + file_name + '.png')
    plt.close()


# ----------------------------------
def read_in_netcdf_fields(variable_name, fullpath_in):
    #print('.....', fullpath_in, variable_name)
    rootgrp = nc.Dataset(fullpath_in, 'r')
    var = rootgrp.groups['fields'].variables[variable_name]
    shape = var.shape
    data = np.ndarray(shape = var.shape)
    data = var[:,:,:]
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