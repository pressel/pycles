#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

## XXX imports

import cython
import netCDF4 as nc
import os
cimport numpy as np
import numpy as np
include "parameters.pxi"

cdef class PostProcessing:

    def __init__(self,namelist) :
        self.out_path = ''
        self.fields_path = ''
        self.gridsize = [0,0,0]
        return

    cpdef initialize(self, namelist):
        uuid = str(namelist['meta']['uuid'])
        out_path = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + uuid[-5:])) 
        self.out_path = out_path
        self.fields_path = str(os.path.join(out_path, namelist['fields_io']['fields_dir'])) # see NetCDFIO.pyx
        self.gridsize = [namelist["grid"]["nx"], namelist["grid"]["ny"], namelist["grid"]["nz"]]
        return

    
    cpdef combine3d(self):

        fields_path = self.fields_path
        out_path = self.out_path

        directories = os.listdir(fields_path)
        print('\nBeginning combination of ranks in time step directories', directories)

        for d in directories: # time steps
            d_path = os.path.join(fields_path, d)
            ranks = os.listdir(d_path) # different processes (cpus)

            print(f'\t Combining ranks {ranks} of time step (dir) {d}')

            file_path = os.path.join(fields_path, d, ranks[0])
            rootgrp = nc.Dataset(file_path, 'r')
            field_keys = rootgrp.groups['fields'].variables.keys()
            dims = rootgrp.groups['dims'].variables

            n_0 = dims['n_0'][0]
            n_1 = dims['n_1'][0]
            n_2 = dims['n_2'][0]

            rootgrp.close()

            out_path_full = os.path.join(out_path, str(d) + '.nc')
            if not os.path.exists(out_path_full):
                self.create_file(out_path_full)

            for f in field_keys:
                f_data_3d = np.empty((n_0, n_1, n_2), dtype=np.double, order='c')
                for r in ranks:
                    if r[-3:] == '.nc':
                        file_path = os.path.join(fields_path, d, r)
                        rootgrp = nc.Dataset(file_path, 'r')
                        fields = rootgrp.groups['fields'].variables
                        dims = rootgrp.groups['dims'].variables
                        ng = dims['ng'][0]
                        nl_0 = dims['nl_0'][0]
                        nl_1 = dims['nl_1'][0]
                        nl_2 = dims['nl_2'][0]

                        n_0 = dims['n_0'][0]
                        n_1 = dims['n_1'][0]
                        n_2 = dims['n_2'][0]

                        indx_lo_0 = dims['indx_lo_0'][0]
                        indx_lo_1 = dims['indx_lo_1'][0]
                        indx_lo_2 = dims['indx_lo_2'][:]

                        f_data = fields[f][:]

                        self.to_3d(
                            f_data, nl_0, nl_1, nl_2, indx_lo_0, indx_lo_1, indx_lo_2, f_data_3d
                        )

                        rootgrp.close()
                self.write_field(out_path_full, f, f_data_3d)
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
                            
    cpdef create_file(self, fname):
        nx, ny, nz = self.gridsize
        
        rootgrp = nc.Dataset(fname, 'w', format='NETCDF4')
        fieldgrp = rootgrp.createGroup('fields')
        fieldgrp.createDimension('nx', nx)
        fieldgrp.createDimension('ny', ny)
        fieldgrp.createDimension('nz', nz)

        rootgrp.close()
        return


    cpdef write_field(self, fname, f, data):
        rootgrp = nc.Dataset(fname, 'r+')
        fields = rootgrp.groups['fields']
        var = fields.createVariable(f, 'f8', ('nx', 'ny', 'nz'))
        var[:, :, :] = data

        rootgrp.close()
        return 
