#!/bin/sh

#  Vis.py
#
#
#  Created by Meyer  Bettina on 01/04/16.
#

import argparse
import os
from h5py import File
import h5py
import numpy as np
import matplotlib.cm as cm
import pylab as plt

import glob
import json as  simplejson


import pickle
from matplotlib.backends.backend_pdf import PdfPages


#from Namelist import Namelist
#nml = NamelistDCBL()
#nml = Namelist()



#----------------------------------------------------------------------
#----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("path")
    parser.add_argument("casename")
    parser.add_argument("--var_name")
    parser.add_argument("--time_min")
    args = parser.parse_args()
    path = args.path
    case_name = args.casename
    if args.var_name:
        var_list = [args.var_name]
    else:
        var_list = ['w', 's', 'potential_temperature', 'temperature', 'ql', 'qt', 'u', 'v']
        var_list = ['w', 's', 'theta', 'temperature', 'ql', 'qt', 'u', 'v']

    global fullpath_out, file_name
    global t, dt, dx, dz, nx
    if args.time_min:
        t_min = np.int(args.time_min)
    else:
        t_min = 0


    # case_name = 'Bomex'

    # path_list = ['../bomex/161130_test/n24/2_full_old_EV12/']
    # path = '../bomex/161130_test/n24/2_QL_old_EV12/'
    path_list = [path]
    for path in path_list:
        fullpath_out = os.path.join(path,'Visualization/')
        print('fullpath_out', fullpath_out)
        #     scheme = 'QL, 2nd, TKE, CFL = 0.1'

        nml = simplejson.loads(open(os.path.join(path,case_name + '.in')).read())
        # namelist_files = glob.glob(path +'*.in')
        # print(namelist_files)
        # for namelist in namelist_files:
        #     nml = simplejson.loads(open(namelist).read())
        dt = nml['visualization']['frequency']
        dx = nml['grid']['dx']
        dz = nml['grid']['dz']
        print('vis dt:' + str(dt) +', dz: '+ str(dz))

        # files = os.listdir(os.path.join(path, 'vis/*.pkl'))       # type = list
        files = os.listdir(os.path.join(path, 'vis'))  # type = list
        nx = nml['grid']['nx']
        print('vis dt:' + str(dt) +', dz: '+ str(dz))

        # files = os.listdir(os.path.join(path, 'vis/*.pkl'))       # type = list
        files = os.listdir(os.path.join(path, 'Visualization'))  # type = list
        print('visualisation files: ', str(files))
        print('')


        # T = np.linspace(0,2400,5)
        # print('T',T)
        # for t in T:
        #     if t < 10:
        #         file_name = np.str(1000000) + np.str(np.int(t))
        #     elif t < 100:
        #         file_name = np.str(100000) + np.str(np.int(t))
        #     elif t < 1000:
        #         file_name = np.str(10000) + np.str(np.int(t))
        #     elif t < 10000:
        #         file_name = np.str(1000) + np.str(np.int(t))
        #     else:
        #         file_name = np.str(100) + np.str(np.int(t))
        #     print('name:', file_name)
        #     fullpath_in = fullpath_out + file_name  + '.pkl'
        #     print('fullpath_in: ' + fullpath_in)

        for file_name in files:
            if file_name[-4:] == '.pkl':
                t = np.int(file_name[0:-4])
                fullpath_in = fullpath_out + file_name

                f = open(fullpath_in)
                data = pickle.load(f)
                for var_name in var_list:
                    print(var_name+ ', fullpath_in: ' + fullpath_in)
                    try:
                        var = data[var_name]
                    except:
                        print("No variable " + var_name)
                        print " "
                        continue

                    levels = np.linspace(np.amin(var), np.amax(var), 100)

                    plot_data_levels(var, var_name, t, levels)
                    plot_data(var, var_name, t)
        ''' (A) setting levels '''
        file_name = files[0]
        i = 0
        while file_name[-4:] != '.pkl':
            i += 1
            file_name = files[i]
        f = open(fullpath_out + file_name)
        data0 = pickle.load(f)


        ''' (B) plotting '''
        for file_name in files:
            if file_name[-4:] == '.pkl':
                t = np.int(file_name[0:-4]) - 1e7
                if t >= t_min:
                    fullpath_in = fullpath_out + file_name
                    print('fullpath_in: ' + fullpath_in)

                    f = open(fullpath_in)
                    data = pickle.load(f)
                    for var_name in var_list:
                        try:
                            var = data[var_name]
                            print(var_name)
                        except:
                            print("No variable " + var_name)
                            continue

                        if np.amin(var) != np.amax(var):
                            if var_name == 'w':
                                levels = np.linspace(-5., 5.0, 100)
                            else:
                                var0 = data0[var_name]
                                levels = np.linspace(np.amin(var0), np.amax(var0), 100)
                            plot_data_levels(var, var_name, t, levels)
                            # plot_data(var, var_name, t)
                    print " "


    print('Ende')


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def read_in(variable_name, group_name, fullpath_in):
    f = File(fullpath_in)

    # Get access to the profiles group
    profiles_group = f[group_name]
    # Get access to the variable dataset
    variable_dataset = profiles_group[variable_name]
    # Get the current shape of the dataset
    variable_dataset_shape = variable_dataset.shape

    variable = np.ndarray(shape=variable_dataset_shape)
    for t in range(variable_dataset_shape[0]):
        if group_name == "timeseries":
            variable[t] = variable_dataset[t]
        elif group_name == "profiles":
            variable[t, :] = variable_dataset[t, :]
        elif group_name == "correlations":
            variable[t, :] = variable_dataset[t, :]
        elif group_name == "fields":
            variable[t] = variable_dataset[t]

    f.close()
    return variable


# ----------------------------------------------------------------------
def plot_data(data, var_name, t):
    # print(data.shape)
    plt.figure()
    if var_name == 'w':
        plt.contourf(data.T,cmap = cm.bwr)
    else:
        plt.contourf(data.T)
    # plt.show()
    plt.colorbar()
    plt.title(var_name + ', (t=' + np.str(t) + 's)')
    plt.xlabel('x (dx=' + np.str(dx) + 'm, nx='+np.str(nx)+')')
    plt.ylabel('height z (dz=' + np.str(dz) + 'm)')
    plt.savefig(fullpath_out + var_name + '_' + str(np.int(t + 1e7)) + '.png')
    plt.savefig(os.path.join(fullpath_out, 'pdf', var_name + '_' + str(t) + '.pdf'))
    plt.close()
    return


def plot_data_levels(data, var_name, t, levels_):
    # print(data.shape)
    plt.figure()
    if var_name == 'w':
        # print('bwr')
        # plt.contourf(data.T, levels=levels_, cmap=plt.cm.bwr)
        plt.contourf(data.T, cmap=cm.bwr, levels=levels_)
    else:
        plt.contourf(data.T, levels=levels_)
    # plt.contourf(data.T)
    # plt.show()
    plt.title(var_name + ', (t=' + np.str(t) + 's)')
    plt.xlabel('x (dx=' + np.str(dx) + 'm)')
    plt.ylabel('height z (dz=' + np.str(dz) + 'm)')
    plt.colorbar()
    plt.savefig(os.path.join(fullpath_out, 'pdf', 'levels_' + var_name + '_' + str(t) + '.pdf'))
    plt.savefig(fullpath_out + 'levels_' + var_name + '_' + str(t) + '_levels.png')
    plt.figure(figsize=(12,8))
    if var_name == 'w':
        # print('bwr')
        # plt.contourf(data.T, levels=levels_, cmap=plt.cm.bwr)
        '''set out - of - bounds colors'''
        cs = plt.contourf(data.T, cmap=cm.bwr, levels=levels_, extend="both")
        cs.cmap.set_under('navy')
        cs.cmap.set_over('darkred') #crimson
    else:
        cs = plt.contourf(data.T, levels=levels_, extend="both")
        cs.cmap.set_under('k')
        cs.cmap.set_over('0.5')

    plt.title(var_name + ', (t=' + np.str(t) + 's)')
    plt.xlabel('x (dx=' + np.str(dx) + 'm, nx=' + np.str(nx) + ')')
    plt.ylabel('height z (dz=' + np.str(dz) + 'm)')
    plt.colorbar()
    path_out = os.path.join(fullpath_out, 'pdf')
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    # plt.savefig(os.path.join(path_out, 'levels_' + var_name + '_' + str(t) + '.pdf'))
    plt.savefig(fullpath_out + 'levels_' + var_name + '_' + str(np.int(t + 1e7)) + '.png')
    plt.close()
    return



def plot_data_levels_cont(var_field,var_name_field,levels_field,var_cont,var_name_cont,levels_cont):
    print('cont print')

    # print('cont print')
    plt.figure()
    if var_name_field == 'w':
        ax1 = plt.contourf(var_field.T, cmap=cm.bwr, levels=levels_field)
    else:
        ax1 = plt.contourf(var_field.T, levels=levels_field)
    ax2 = plt.contour(var_cont.T, levels=levels_cont,colors='k')
    ax3 = plt.contour(var_cont.T, levels=[0.018,0.02], colors='b',linewidths=1)
    plt.clabel(ax2,inline=1)
    plt.colorbar(ax1,shrink=0.8)
    plt.colorbar(ax3,shrink=0.8)
    plt.title(var_name_field + ', (t=' + np.str(t) + 's)')
    plt.xlabel('x (dx=' + np.str(dx) + 'm)')
    plt.ylabel('height z (dz=' + np.str(dz) + 'm)')
    # plt.savefig(fullpath_out + 'phi/' + var_name + '_' + file_name + '.png')
    print('hoi')
    path = fullpath_out + var_name_field + '_cont_' + file_name + '.png'
    print('hohioi')
    print('out: ', fullpath_out + var_name_field + '_cont_' + file_name + '.png')
    plt.savefig(fullpath_out + var_name_field + '_cont_' + file_name + '.png')
    plt.close()


def output(fullpath_in, fullpath_out, file_name, var_name):
    print('out')



# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()