#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import cython
import netCDF4 as nc
import xarray as xr
import os
import shutil
import numpy as np
cimport numpy as np
cimport ParallelMPI
cimport ReferenceState

cdef class PostProcessing:

    def __init__(self,namelist) :
        pass

    cpdef initialize(self, namelist):
        uuid = str(namelist['meta']['uuid'])
        out_dir = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + uuid[-5:])) 
        self.out_dir = out_dir
        self.fields_dir = str(os.path.join(out_dir, namelist['fields_io']['fields_dir']))
        stats_dir = str(os.path.join(out_dir, namelist['stats_io']['stats_dir']))
        self.stats_file = os.path.join(stats_dir,'Stats.'+namelist['meta']['simname']+'.nc')
        self.gridsize = [namelist["grid"]["nx"], namelist["grid"]["ny"], namelist["grid"]["nz"]]
        self.gridspacing = [namelist["grid"]["dx"], namelist["grid"]["dy"], namelist["grid"]["dz"]]
        if namelist['postprocessing']['collapse_y']:
            self.collapse_y = True
        else:
            self.collapse_y = False
        if namelist['postprocessing']['half_x']:
            self.half_x = True
        else:
            self.half_x = False
        if namelist['postprocessing']['only_T_anomaly']:
            self.only_T_anomaly = True
        else:
            self.only_T_anomaly = False
        return

    
    cpdef combine3d(self, ParallelMPI.ParallelMPI Pa, ReferenceState.ReferenceState Ref):
        '''
        Before: 
            every time step is a directory with .nc files for each rank (i.e. processor)
            and data is stored as 1D arrays
        After: 
            every time step is one .nc file
            and data is stored as 3D arrays
        '''
        
        Pa.barrier()
        if Pa.rank == 0:
        
            nx, ny, nz = self.gridsize

            fields_dir = self.fields_dir
            out_dir = self.out_dir

            directories = os.listdir(fields_dir)
            print('\nBeginning combination of ranks in time step directories', directories)

            for d in directories:
                d_path = os.path.join(fields_dir, d)
                ranks = os.listdir(d_path)

                print(f'\t Combining ranks {ranks} of time step (dir) {d}')

                file_path = os.path.join(fields_dir, d, ranks[0])
                save_path = os.path.join(out_dir,'fields/', str(d) + '.nc')
                with xr.open_dataset(file_path, group='fields') as ds:
                    field_keys = ds.variables


                variables_to_save = {}
                for f in field_keys:
                    f_data_3d = np.empty((nx, ny, nz), dtype=np.double, order='c')


                    for r in ranks:
                        if r[-3:] == '.nc':
                            file_path = os.path.join(fields_dir, d, r)

                            with xr.open_dataset(file_path, group='fields') as ds:
                                f_data = ds[f].values # to_numpy()
                            with xr.open_dataset(file_path, group='dims') as ds:
                                dims = ds.variables

                            nl_0 = dims['nl_0'][0] # grid size per processor
                            nl_1 = dims['nl_1'][0]
                            nl_2 = dims['nl_2'][0]

                            indx_lo_0 = dims['indx_lo_0'][0]
                            indx_lo_1 = dims['indx_lo_1'][0]
                            indx_lo_2 = dims['indx_lo_2'][:]


                            self.to_3d(
                                f_data, nl_0, nl_1, nl_2, indx_lo_0, indx_lo_1, indx_lo_2, f_data_3d
                            )

                            variables_to_save[f] = (('x','y','z','t'), np.expand_dims(f_data_3d, axis=3))

                # save the new file instead of the old directory
                self.save_timestep(save_path, variables_to_save, d, Ref)
                shutil.rmtree(d_path)

            print('Finished combining ranks per time step.\n')
        return

    cpdef to_3d(self, double[:] f_data, int nl_0, int nl_1, int nl_2, int indx_lo_0,
                int indx_lo_1, int indx_lo_2, double[:, :, :] f_data_3d):

        cdef:
            int istride = nl_2 * nl_1
            int jstride = nl_2
            int ishift
            int jshift

            int i, j, k

        with nogil:
            for i in range(nl_0):
                ishift = i * istride
                for j in range(nl_1):
                    jshift = j * jstride
                    for k in range(nl_2):
                        ijk = ishift + jshift + k
                        f_data_3d[
                            indx_lo_0 + i, indx_lo_1 + j, indx_lo_2 + k] = f_data[ijk]
                            

    cpdef save_timestep(self, fname, variables, time, ReferenceState.ReferenceState Ref):

        nx, ny, nz = self.gridsize
        dx, dy, dz = self.gridspacing
        
        domain_size = [dx*nx, dy*ny, dz*nz]
        gridpoints = [np.linspace(0,l,n) for (n,l) in zip(self.gridsize,domain_size)]
        coords = {'x':gridpoints[0],
                'y':gridpoints[1],
                'z':gridpoints[2],
                't':[float(time)]}

        ds = xr.Dataset(variables, coords=coords)
        ds = ds.sortby('t')

        if self.collapse_y:
            ds = ds.isel(y=ny//2)
            ds = ds.drop_vars(["y","v"])

        if self.only_T_anomaly:
            T0 = Ref.temperature0_unghosted
            T0 = np.expand_dims(np.ones((nx,nz))*T0,axis=-1)
            ds['temperature_anomaly'] = ds['temperature'] - T0
            ds = ds.drop_vars(["temperature","s"])        
        
        if self.half_x:
            half_x = nx//2 +1*(nx%2==1)
            ds = ds.isel(x=slice(0,half_x))

        ds.to_netcdf(fname)
