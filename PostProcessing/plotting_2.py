import netCDF4 as nc
import numpy as np
import pylab as plt
import glob
import os
import json as  simplejson
import argparse



'''
Use glob to find all directories whose names match a certain format, with wildcards allowed.
Namelist is found and read in to provide information about the file.
'''



my_colors=['blue','red','green','magenta','cyan','yellow','brown','Orange','DarkViolet','SkyBlue','Chartreuse', 'DarkKhaki','DarkSeaGreen','DeepPink','IndianRed','Lavender', 'MediumOrchid']

def main():
    start_hour=1.5
    stop_hour=2
    interval=2

    profile_vars = ['u_translational_mean', 'v_translational_mean', 's_mean', 'qt_mean','qv_mean','temperature_mean',
                    'thetali_mean','cloud_fraction', 'ql_mean',
                    's_sgs_variance','qt_sgs_variance','qt_sgs_variance_clip', 'sgs_covariance', 'sgs_correlation']#, 'qr_mean', 'qt_sedimentation_flux', 'qr_sedimentation_flux', ]


    variances = ['horizontal_velocity_variance','vertical_velocity_variance','resolved_tke', 'resolved_skewness','resolved_theta_variance']
    timeseries_vars = ['cloud_base', 'cloud_top', 'cloud_fraction', 'lwp']


    dir = '/Users/ckaul/Desktop/LES_results/DYCOMS_RF02/'
    namelist_files = glob.glob(dir +'*.in')
    print(namelist_files)

    count = -1
    #--------MEANS
    for namelist in namelist_files:
        simname = namelist[len(dir):-3]
        print(simname)

        # print(namelist)
        nml = simplejson.loads(open(namelist).read())
        flag, name  = case_specific_parse(nml)
        #overwrite hue

        if flag == 1:
            print(name)
            count += 1
            ncfile = dir + 'Stats.' + simname + '.nc'
            print(ncfile)
            data = nc.Dataset(ncfile,'r')
            nkr = data.groups['profiles'].variables['z'].shape[0]

            perhour =int(3600.0/nml['stats_io']['frequency'])

            my_i = np.arange(int(start_hour*perhour), int(stop_hour*perhour), interval)

            line_color = my_colors[count]

            n_i = np.shape(my_i)[0]
            # plot profile vars
            for var in profile_vars:
                print(var)

                title = 'profile_'+ var
                plt.figure(title)
                reg1 = np.zeros(nkr)
                for t in my_i:
                    for k in np.arange(nkr):
                        reg1[k] += data.groups['profiles'].variables[var][t,k]/float(n_i)
                plt.plot(reg1, data.groups['profiles'].variables['z'][:], linewidth = 2,label = name, color=line_color, marker='o')

            # plot profiles of variances/tke, additional computation required here
            reg1 = np.zeros(nkr)
            reg1w = np.zeros(nkr)
            regth = np.zeros(nkr)
            regsk = np.zeros(nkr)
            try:
                for t in my_i:
                    for k in np.arange(nkr):
                        reg1[k] += data.groups['profiles'].variables['u_mean2'][t,k]/float(n_i)-data.groups['profiles'].variables['u_mean'][t,k]**2/float(n_i)
                        reg1[k] += data.groups['profiles'].variables['v_mean2'][t,k]/float(n_i)-data.groups['profiles'].variables['v_mean'][t,k]**2/float(n_i)
                        reg1w[k] += data.groups['profiles'].variables['w_mean2'][t,k]/float(n_i)
                        regsk[k]+= (data.groups['profiles'].variables['w_mean3'][t,k]) /float(n_i)
                        # regsk[k]+= (data.groups['profiles'].variables['w_mean3'][t,k]/(data.groups['profiles'].variables['w_mean2'][t,k]+1e-10)**1.5) /float(n_i)
                        regth[k] += data.groups['profiles'].variables['theta_mean2'][t,k]/float(n_i)-data.groups['profiles'].variables['theta_mean'][t,k]**2/float(n_i)

                plt.figure('horizontal_velocity_variance')
                plt.plot(reg1, data.groups['profiles'].variables['z'][:], linewidth = 2,label = name, color=line_color)

                plt.figure('vertical_velocity_variance')
                plt.plot(reg1w, data.groups['profiles'].variables['z'][:], linewidth = 2,label =name, color=line_color)


                plt.figure('resolved_tke')
                plt.plot((reg1[:]+reg1w[:])*0.5, data.groups['profiles'].variables['z'][:], linewidth = 2,label = name, color=line_color)


                plt.figure('resolved_skewness')
                plt.plot(regsk, data.groups['profiles'].variables['z'][:], linewidth = 2,label = name, color=line_color)


                plt.figure('resolved_theta_variance')
                plt.plot(regth, data.groups['profiles'].variables['z'][:], linewidth = 2,label = name, color=line_color)
            except:
                pass

            # plot timeseries
            for var in timeseries_vars:
                print(var)
                plt.figure(var)
                plt.title(var)
               plt.plot(data.groups['timeseries'].variables['t'][:]/3600.0,data.groups['timeseries'].variables[var][:],linewidth = 2,label = name, color=line_color)


            data.close()

    for var in profile_vars:
        plt.figure('profile_'+var)
        plt.legend(loc=0)


    for var in variances:
        plt.figure(var)
        plt.legend(loc=0)

    for var in timeseries_vars:
        plt.figure(var)
        plt.legend(loc=0)




    plt.show()




def case_specific_parse(nml):
    '''
    Use the namelist dictionary to set up specific plotting procedures
    :param nml: JSON namelist file converted to dictionary
    :return:
        flag: 1=use this data file, 0=skip
        label: string to be used for plot labeling
        hue: color to be used for plot line
    '''
    flag=1
    label = nml['meta']['simname']
    # try:
    #    sgs_flag = nml['sgs']['sgs_condensation']
    # except:
    #     sgs_flag = False
    # if sgs_flag :
    #     flag = 0
    #     if nml['sgs']['condensation']['quadrature_order'] == 20:      #nml['sgs']['condensation']['c_variance']< 0.15 and nml['sgs']['condensation']['c_variance']> 0.05:
    #         flag = 1

    return flag, label

if __name__=='__main__':


    main()
