#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import netCDF4 as nc
import os
import shutil
cimport ParallelMPI
cimport TimeStepping
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Grid
import numpy as np
cimport numpy as np
import cython
import shutil
import subprocess

import combine3d
cdef class NetCDFIO_Stats:
    def __init__(self):
        self.root_grp = None
        self.profiles_grp = None
        self.ts_grp = None
        return

    @cython.wraparound(True)
    cpdef initialize(self, dict namelist, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])
        self.frequency = namelist['stats_io']['frequency']

        # Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + self.uuid[-5:]))

        if Pa.rank == 0:
            try:
                os.mkdir(outpath)
            except:
                pass

        self.stats_path = str( os.path.join(outpath, namelist['stats_io']['stats_dir']))
        if Pa.rank == 0:
            try:
                os.mkdir(self.stats_path)
            except:
                pass


        self.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname'] + '.nc')
        if os.path.exists(self.path_plus_file):
            for i in range(100):
                res_name = 'Restart_'+str(i)
                print "Here " + res_name
                if os.path.exists(self.path_plus_file):
                    self.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname']
                           + '.' + res_name + '.nc')
                else:
                    break

        Pa.barrier()



        if Pa.rank == 0:
            shutil.copyfile(
                os.path.join( './', namelist['meta']['simname'] + '.in'),
                os.path.join( outpath, namelist['meta']['simname'] + '.in'))
            self.setup_stats_file(Gr, Pa)
        return

    cpdef open_files(self, ParallelMPI.ParallelMPI Pa):
        if Pa.rank == 0:
            self.root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            self.profiles_grp = self.root_grp.groups['profiles']
            self.ts_grp = self.root_grp.groups['timeseries']
        return

    cpdef close_files(self, ParallelMPI.ParallelMPI Pa):
        if Pa.rank == 0:
            self.root_grp.close()
        return

    cpdef setup_stats_file(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')

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

    cpdef add_profile(self, var_name, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        if Pa.rank == 0:
            root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            profile_grp = root_grp.groups['profiles']
            new_var = profile_grp.createVariable(var_name, 'f8', ('t', 'z'))

            root_grp.close()

        return

    cpdef add_reference_profile(self, var_name, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
        '''
        Adds a profile to the reference group NetCDF Stats file.
        :param var_name: name of variable
        :param Gr: Grid class
        :param Pa: ParallelMPI class
        :return:
        '''
        if Pa.rank == 0:
            root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            reference_grp = root_grp.groups['reference']
            new_var = reference_grp.createVariable(var_name, 'f8', ('z',))

            root_grp.close()

        return

    cpdef add_ts(self, var_name, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
        if Pa.rank == 0:
            root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            ts_grp = root_grp.groups['timeseries']
            new_var = ts_grp.createVariable(var_name, 'f8', ('t',))

            root_grp.close()
        return

    cpdef write_profile(self, var_name, double[:] data, ParallelMPI.ParallelMPI Pa):
        if Pa.rank == 0:
            #root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            #profile_grp = root_grp.groups['profiles']
            var = self.profiles_grp.variables[var_name]
            var[-1, :] = np.array(data)
            #root_grp.close()
        return

    cpdef write_reference_profile(self, var_name, double[:] data, ParallelMPI.ParallelMPI Pa):
        '''
        Writes a profile to the reference group NetCDF Stats file. The variable must have already been
        added to the NetCDF file using add_reference_profile
        :param var_name: name of variables
        :param data: data to be written to file
        :param Pa: ParallelMPI class
        :return:
        '''
        if Pa.rank == 0:
            root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            reference_grp = root_grp.groups['reference']
            var = reference_grp.variables[var_name]
            var[:] = np.array(data)
            root_grp.close()
        return

    @cython.wraparound(True)
    cpdef write_ts(self, var_name, double data, ParallelMPI.ParallelMPI Pa):
        if Pa.rank == 0:
            #root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            #ts_grp = root_grp.groups['timeseries']
            var = self.ts_grp.variables[var_name]
            var[-1] = data
            #root_grp.close()
        return

    cpdef write_simulation_time(self, double t, ParallelMPI.ParallelMPI Pa):
        if Pa.rank == 0:
            #root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            #profile_grp = root_grp.groups['profiles']
            #ts_grp = root_grp.groups['timeseries']

            # Write to profiles group
            profile_t = self.profiles_grp.variables['t']
            profile_t[profile_t.shape[0]] = t

            # Write to timeseries group
            ts_t = self.ts_grp.variables['t']
            ts_t[ts_t.shape[0]] = t

            #root_grp.close()
        return

cdef class NetCDFIO_Fields:
    def __init__(self):
        return

    @cython.wraparound(True)
    cpdef initialize(self, dict namelist, ParallelMPI.ParallelMPI Pa):

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])
        self.frequency = namelist['fields_io']['frequency']
        self.glue_count = 1000000
        self.diagnostic_fields = namelist['fields_io']['diagnostic_fields']

        # Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + self.uuid[-5:]))
        self.fields_path = str(os.path.join(outpath, namelist['fields_io']['fields_dir']))
        if Pa.rank == 0:
            try:
                os.mkdir(outpath)
            except:
                pass
            try:
                os.mkdir(self.fields_path)
            except:
                pass

            shutil.copyfile( os.path.join('./', namelist['meta']['simname'] + '.in'),
                             os.path.join( outpath, namelist['meta']['simname'] + '.in'))
        return

    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        new_dir = os.path.join(
            self.fields_path, str(int(self.last_output_time)))

        if Pa.rank == 0 and not os.path.exists(new_dir):
            try:
                os.mkdir(new_dir)
            except:
                print('Problem creating fields output dir')

        Pa.barrier()
        self.output_path = str(new_dir)
        self.path_plus_file = str(
            os.path.join(
                self.output_path, str(
                    Pa.rank) + '.nc'))
        self.create_fields_file(Gr, Pa)
        self.do_output = True


        return

    cpdef glue_fields(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        Pa.barrier() #Make sure tha all output has finished

        if Pa.rank == 0:
            #Rename current output directory
            dir = None
            dir_files = os.listdir(self.fields_path)
            for df in dir_files:
                if '_glue' not in df:
                    dir = df

            tmp_path = self.fields_path + '/tmp_files'
            out_path = self.fields_path + '/' + dir

            os.rename(out_path, tmp_path)


            #Create a new directory with the correct name
            os.mkdir(out_path + '_glue')

            #Get list of files
            file_list = os.listdir(tmp_path)
            #open first file in list
            print file_list, tmp_path, out_path
            rt_grp = nc.Dataset(tmp_path + '/' + file_list[0],'r')

            fields = rt_grp['fields']
            dims = rt_grp.groups['dims'].variables
            vars = fields.variables.keys()

            #Get the input dimensions
            n_0_in = dims['n_0'][0]
            n_1_in = dims['n_1'][0]
            n_2_in = dims['n_2'][0]
            rt_grp.close()

            #Setup output file
            out_rt_grp = nc.Dataset(out_path + '_glue' + '/0.nc','w',format='NETCDF3_64BIT' )

            #Set up output in paraview in compatible untisl
            xd = out_rt_grp.createDimension('x', n_0_in)
            yd = out_rt_grp.createDimension('y', n_1_in)
            zd = out_rt_grp.createDimension('z', n_2_in)
            td = out_rt_grp.createDimension('Times', 1)

            xv = out_rt_grp.createVariable('x','f8',('x',))
            xv[:] = np.arange(1, n_0_in + 1,dtype=np.double) * Gr.dims.dx[0] -  Gr.dims.dx[0]/2.0
            xv.units = 'm'
            yv = out_rt_grp.createVariable('y','f8',('y',))
            yv[:] = np.arange(1, n_1_in + 1,dtype=np.double) * Gr.dims.dx[1] -  Gr.dims.dx[1]/2.0
            yv.units = 'm'
            zv = out_rt_grp.createVariable('z','f8',('z',))
            zv[:] = np.arange(1, n_2_in + 1,dtype=np.double) * Gr.dims.dx[2] -  Gr.dims.dx[2]/2.0
            zv.units = 'm'
            tv = out_rt_grp.createVariable('Times','f8',('Times',))
            tv.units = "<time length> since <date>"
            tv[:] = 0.0
            out_rt_grp.sync()

            #Setup output file in COADS format


            for v in vars:
                v_data = out_rt_grp.createVariable(v, 'f8', ('Times', 'z', 'y', 'x',))
                v_data.units = ""
                out_rt_grp.sync()

            for tf in file_list:
                print tmp_path + '/' + tf
                print os.listdir(self.fields_path)
                rt_grp = nc.Dataset(tmp_path + '/' + tf,'r')
                fields = rt_grp['fields'].variables
                dims = rt_grp.groups['dims'].variables
                nl_0_in = dims['nl_0'][0]
                nl_1_in = dims['nl_1'][0]
                nl_2_in = dims['nl_2'][0]


                indx_lo_0 = dims['indx_lo_0'][0]
                indx_lo_1 = dims['indx_lo_1'][0]
                indx_lo_2 = dims['indx_lo_2'][0]
                for v in vars:
                    d_3d = np.empty((nl_2_in, nl_1_in, nl_0_in), dtype=np.double)
                    f_data = fields[v][:]

                    combine3d.to_3d(
                        f_data, nl_0_in, nl_1_in, nl_2_in, indx_lo_0, indx_lo_1, indx_lo_2, d_3d)

                    vd = out_rt_grp.variables[v][0,
                         indx_lo_2:indx_lo_2+nl_2_in,
                         indx_lo_1:indx_lo_1+nl_1_in,
                         indx_lo_0:indx_lo_0+nl_0_in] = d_3d[:,:,:]


                out_rt_grp.sync()

                rt_grp.close()



            out_rt_grp.close()

            #Danger Danger Danger
            shutil.rmtree(tmp_path)

            #Now setup commands to call visualization scripts
            run_script = 'pvbatch dcbl_plot.py ' + out_path + '_glue' + '/0.nc' + ' ./paraview_figs/' + str(self.glue_count) + ".png"
            subprocess.call([run_script], shell=True)

            shutil.rmtree(out_path + '_glue')

        self.glue_count += 1




        return

    cpdef create_fields_file(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        rootgrp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')
        dimgrp = rootgrp.createGroup('dims')
        fieldgrp = rootgrp.createGroup('fields')

        fieldgrp.createDimension('nl', np.int(Gr.dims.npl))
        dimgrp.createDimension('d1', 1)

        nl_0 = dimgrp.createVariable('nl_0', 'i4', ('d1'))
        nl_1 = dimgrp.createVariable('nl_1', 'i4', ('d1'))
        nl_2 = dimgrp.createVariable('nl_2', 'i4', ('d1'))
        n_0 = dimgrp.createVariable('n_0', 'i4', ('d1'))
        n_1 = dimgrp.createVariable('n_1', 'i4', ('d1'))
        n_2 = dimgrp.createVariable('n_2', 'i4', ('d1'))
        indx_lo_0 = dimgrp.createVariable('indx_lo_0', 'i4', ('d1'))
        indx_lo_1 = dimgrp.createVariable('indx_lo_1', 'i4', ('d1'))
        indx_lo_2 = dimgrp.createVariable('indx_lo_2', 'i4', ('d1'))
        ng = dimgrp.createVariable('ng', 'i4', ('d1'))

        # Dimension of equivalent 3d array owned by this rank
        nl_0[:] = Gr.dims.nl[0]
        nl_1[:] = Gr.dims.nl[1]
        nl_2[:] = Gr.dims.nl[2]

        n_0[:] = Gr.dims.n[0]
        n_1[:] = Gr.dims.n[1]
        n_2[:] = Gr.dims.n[2]

        # Lower Left has point in global 3d array of the equivalent 3d array
        # owned by this processor
        indx_lo_0[:] = Gr.dims.indx_lo[0]
        indx_lo_1[:] = Gr.dims.indx_lo[1]
        indx_lo_2[:] = Gr.dims.indx_lo[2]

        ng[:] = Gr.dims.npd

        rootgrp.close()
        return

    cpdef dump_prognostic_variables(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):

        cdef:
            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t var_shift
            double[:] data = np.empty((Gr.dims.npl,), dtype=np.double, order='c')
            Py_ssize_t count
        for name in PV.name_index.keys():
            self.add_field(name)
            var_shift = PV.get_varshift(Gr, name)
            count = 0
            with nogil:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            data[count] = PV.values[var_shift + ijk]
                            count += 1
            self.write_field(name, data)
        return


    cpdef dump_diagnostic_variables(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t var_shift
            double[:] data = np.empty((Gr.dims.npl,), dtype=np.double, order='c')
            Py_ssize_t count
        for name in self.diagnostic_fields:
            try:
                self.add_field(name)
                var_shift = DV.get_varshift(Gr, str(name))
                count = 0
                with nogil:
                    for i in range(imin, imax):
                        ishift = i * istride
                        for j in range(jmin, jmax):
                            jshift = j * jstride
                            for k in range(kmin, kmax):
                                ijk = ishift + jshift + k
                                data[count] = DV.values[var_shift + ijk]
                                count += 1
                self.write_field(str(name), data)
            except:
                Pa.root_print('Could not output DiagnosticVariable Field: ' + name )
        return

    cpdef add_field(self, name):
        rootgrp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        fieldgrp = rootgrp.groups['fields']
        fieldgrp.createVariable(name, 'f8', ('nl'))
        rootgrp.close()
        return

    cpdef write_field(self, name, double[:] data):
        rootgrp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        fieldgrp = rootgrp.groups['fields']
        var = fieldgrp.variables[name]
        var[:] = np.array(data)
        rootgrp.close()
        return



cdef class NetCDFIO_CondStats:
    def __init__(self):

        return

    @cython.wraparound(True)
    cpdef initialize(self, dict namelist, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])
        # if a frequency is not defined for the conditional statistics, set frequency to the maximum simulation time
        try:
            self.frequency = namelist['conditional_stats']['frequency']
        except:
            self.frequency = namelist['time_stepping']['t_max']


        # Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + self.uuid[-5:]))

        if Pa.rank == 0:
            try:
                os.mkdir(outpath)
            except:
                pass

        # Set a default name for the output directory if it is not defined in the namelist
        try:
            self.stats_path = str( os.path.join(outpath, namelist['conditional_stats']['stats_dir']))
        except:
            self.stats_path = str( os.path.join(outpath, 'cond_stats'))

        if Pa.rank == 0:
            try:
                os.mkdir(self.stats_path)
            except:
                pass


        self.path_plus_file = str( self.stats_path + '/' + 'CondStats.' + namelist['meta']['simname'] + '.nc')
        if os.path.exists(self.path_plus_file):
            for i in range(100):
                res_name = 'Restart_'+str(i)
                if os.path.exists(self.path_plus_file):
                    self.path_plus_file = str( self.stats_path + '/' + 'CondStats.' + namelist['meta']['simname']
                           + '.' + res_name + '.nc')
                else:
                    break

        Pa.barrier()



        if Pa.rank == 0:
            shutil.copyfile(
                os.path.join( './', namelist['meta']['simname'] + '.in'),
                os.path.join( outpath, namelist['meta']['simname'] + '.in'))
        return

    cpdef create_condstats_group(self, str groupname, str dimname, double [:] dimval, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        if Pa.rank == 0:
            root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')
            sub_grp = root_grp.createGroup(groupname)
            sub_grp.createDimension('z', Gr.dims.n[2])
            sub_grp.createDimension(dimname, len(dimval))
            sub_grp.createDimension('t', None)
            z = sub_grp.createVariable('z', 'f8', ('z'))
            z[:] = np.array(Gr.z[Gr.dims.gw:-Gr.dims.gw])
            dim = sub_grp.createVariable(dimname, 'f8', (dimname))
            dim[:] = np.array(dimval[:])
            sub_grp.createVariable('t', 'f8', ('t'))
            del z
            del dim
            root_grp.close()
        return

    cpdef add_condstat(self, str varname, str groupname, str dimname, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        if Pa.rank == 0:
            root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            sub_grp = root_grp.groups[groupname]
            new_var = sub_grp.createVariable(varname, 'f8', ('t', 'z', dimname))

            root_grp.close()

        return


    cpdef write_condstat(self, varname, groupname, double [:,:] data, ParallelMPI.ParallelMPI Pa):
        if Pa.rank == 0:
            root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            sub_grp = root_grp.groups[groupname]
            var = sub_grp.variables[varname]

            var[-1, :,:] = np.array(data)[:,:]

            root_grp.close()
        return


    cpdef write_condstat_time(self, double t, ParallelMPI.ParallelMPI Pa):
        if Pa.rank == 0:
            try:
                root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
                for groupname in root_grp.groups:
                    sub_grp = root_grp.groups[groupname]

                    # Write to sub_grp
                    group_t = sub_grp.variables['t']
                    group_t[group_t.shape[0]] = t

                root_grp.close()
            except:
                pass
        return