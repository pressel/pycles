import netCDF4 as nc
import os
import shutil
cimport ParallelMPI
cimport TimeStepping
cimport PrognosticVariables
cimport Grid
import numpy as np
cimport numpy as np
import cython

cdef class NetCDFIO_Stats:
    def __init__(self):
        return

    cpdef initialize(self,dict namelist, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])
        self.frequency=namelist['stats_io']['frequency']

        #Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root']+'Output.'+namelist['meta']['simname'] + '.'+self.uuid[-5:]))
        self.stats_path = str(os.path.join(outpath, namelist['stats_io']['stats_dir'] ))
        self.path_plus_file = str(self.stats_path + '/' + 'Stats.'+namelist['meta']['simname']+'.nc')
        if Pa.rank == 0:
            try:
                os.mkdir(outpath)
            except:
                pass
            try:
                os.mkdir(self.stats_path)
            except:
                pass

            shutil.copyfile(os.path.join('./',namelist['meta']['simname']+'.in'),os.path.join(outpath,namelist['meta']['simname']+'.in'))



            self.setup_stats_file(Gr, Pa)

        return

    cpdef setup_stats_file(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        root_grp = nc.Dataset(self.path_plus_file,'w',format='NETCDF4')

        #Set profile dimensions
        profile_grp = root_grp.createGroup('profiles')
        profile_grp.createDimension('z',Gr.dims.n[2])
        z = profile_grp.createVariable('z','f8',('z'))
        z[:] = np.array(Gr.z[Gr.dims.gw:-Gr.dims.gw])
        del z

        reference_grp = root_grp.createGroup('reference')
        reference_grp.createDimension('z',Gr.dims.n[2])
        z = reference_grp.createVariable('z','f8',('z'))
        z[:] = np.array(Gr.z[Gr.dims.gw:-Gr.dims.gw])
        del z

        ts_grp = root_grp.createGroup('timeseries')


        root_grp.close()

        return


cdef class NetCDFIO_Fields:
    def __init__(self):
        return
    cpdef initialize(self, dict namelist, ParallelMPI.ParallelMPI Pa):

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])

        self.frequency=namelist['fields_io']['frequency']

        #Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root']+'Output.'+namelist['meta']['simname'] + '.'+self.uuid[-5:]))
        self.fields_path = str(os.path.join(outpath, namelist['fields_io']['fields_dir'] ))
        if Pa.rank == 0:
            try:
                os.mkdir(outpath)
            except:
                pass
            try:
                os.mkdir(self.fields_path)
            except:
                pass

            shutil.copyfile(os.path.join('./',namelist['meta']['simname']+'.in'),os.path.join(outpath,namelist['meta']['simname']+'.in'))

        return


    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        #Only do this at the last RK time step
        if TS.rk_step == TS.n_rk_steps - 1:
            if self.last_output_time + self.frequency <=  TS.t + TS.dt:
                TS.dt = self.last_output_time + self.frequency - TS.t
                self.last_output_time += self.frequency

            if self.last_output_time == TS.t:
                try:
                    new_dir = os.path.join(self.fields_path,str(int(self.last_output_time)))
                    if not os.path.exists(new_dir):
                        os.mkdir(new_dir)
                except:
                    print('Problem creating fields output dir')
                self.output_path = str(new_dir)
                self.path_plus_file = str(os.path.join(self.output_path,str(Pa.rank)+'.nc'))
                self.create_fields_file(Gr,Pa)
                Pa.root_print('Now doing 3D IO')
                self.dump_prognostic_variables(Gr, PV)

        elif TS.t == 0.0 and TS.rk_step == 0:
            try:
                new_dir = os.path.join(self.fields_path,str(int(self.last_output_time)))
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
            except:
                print('Problem creating fields output dir')
            self.output_path = str(new_dir)
            self.path_plus_file = str(os.path.join(self.output_path,str(Pa.rank)+'.nc'))
            self.create_fields_file(Gr,Pa)
            Pa.root_print('Now doing 3D IO')
            self.dump_prognostic_variables(Gr, PV)


        return


    cpdef create_fields_file(self,Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        rootgrp = nc.Dataset(self.path_plus_file,'w',format='NETCDF4')
        dimgrp = rootgrp.createGroup('dims')
        fieldgrp = rootgrp.createGroup('fields')

        fieldgrp.createDimension('nl',np.int(Gr.dims.npl))
        dimgrp.createDimension('d1',1)

        nl_0 = dimgrp.createVariable('nl_0','i4',('d1'))
        nl_1 = dimgrp.createVariable('nl_1','i4',('d1'))
        nl_2 = dimgrp.createVariable('nl_2','i4',('d1'))
        n_0 = dimgrp.createVariable('n_0','i4',('d1'))
        n_1 = dimgrp.createVariable('n_1','i4',('d1'))
        n_2 = dimgrp.createVariable('n_2','i4',('d1'))
        indx_lo_0 = dimgrp.createVariable('indx_lo_0','i4',('d1'))
        indx_lo_1 = dimgrp.createVariable('indx_lo_1','i4',('d1'))
        indx_lo_2 = dimgrp.createVariable('indx_lo_2','i4',('d1'))
        ng = dimgrp.createVariable('ng','i4',('d1'))

        #Dimension of equivalent 3d array owned by this rank
        nl_0[:] = Gr.dims.nl[0]
        nl_1[:] = Gr.dims.nl[1]
        nl_2[:] = Gr.dims.nl[2]

        n_0[:] = Gr.dims.n[0]
        n_1[:] = Gr.dims.n[1]
        n_2[:] = Gr.dims.n[2]

        #Lower Left has point in global 3d array of the equivalent 3d array owned by this processor
        indx_lo_0[:] = Gr.dims.indx_lo[0]
        indx_lo_1[:] = Gr.dims.indx_lo[1]
        indx_lo_2[:] = Gr.dims.indx_lo[2]

        ng[:] = Gr.dims.npd

        rootgrp.close()

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef dump_prognostic_variables(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):

        cdef:
            long i,j,k, ijk, ishift, jshift
            long istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            long jstride = Gr.dims.nlg[2]
            long imin = Gr.dims.gw
            long jmin = Gr.dims.gw
            long kmin = Gr.dims.gw
            long imax = Gr.dims.nlg[0] - Gr.dims.gw
            long jmax = Gr.dims.nlg[1] - Gr.dims.gw
            long kmax = Gr.dims.nlg[2] - Gr.dims.gw
            long var_shift
            double [:] data = np.empty((Gr.dims.npl,),dtype=np.double,order='c')
            long count
        for name in PV.name_index.keys():
            self.add_field(name)
            var_shift = PV.get_varshift(Gr,name)
            count = 0
            with nogil:
                for i in range(imin,imax):
                    ishift = i * istride
                    for j in range(jmin,jmax):
                        jshift = j * jstride
                        for k in range(kmin,kmax):
                            ijk = ishift + jshift + k
                            data[count] = PV.values[var_shift+ijk]
                            count += 1
            self.write_field(name,data)

        return


    cpdef add_field(self,name):
        rootgrp = nc.Dataset(self.path_plus_file,'r+',format='NETCDF4')
        fieldgrp = rootgrp.groups['fields']
        fieldgrp.createVariable(name,'f8',('nl'))
        rootgrp.close()
        return

    cpdef write_field(self,name,double [:] data):
        rootgrp = nc.Dataset(self.path_plus_file,'r+',format='NETCDF4')
        fieldgrp = rootgrp.groups['fields']
        var = fieldgrp.variables[name]
        var[:] = np.array(data)
        rootgrp.close()
        return
