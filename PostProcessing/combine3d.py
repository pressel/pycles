import netCDF4 as nc
import argparse
import os
import sys
import numpy as np
import combine3d


def main():

    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("fields_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    directories = os.listdir(args.fields_dir)
    print('Found the following directories', directories)
    print('Beginning combination of files')

    for d in directories:
        print('\t Combining ' + d)
        d_path = os.path.join(args.fields_dir, d)
        ranks = os.listdir(d_path)
        print('\t\t Combining files')

        print(ranks)
        file_path = os.path.join(args.fields_dir, d, ranks[0])
        rootgrp = nc.Dataset(file_path, 'r')
        field_keys = rootgrp.groups['fields'].variables.keys()
        dims = rootgrp.groups['dims'].variables

        n_0 = dims['n_0'][0]
        n_1 = dims['n_1'][0]
        n_2 = dims['n_2'][0]
        x = dims['x'][:]
        y = dims['y'][:]
        z = dims['z'][:]

        rootgrp.close()

        out_path = os.path.join(args.out_dir, str(1000000 + int(d)) + '.nc')
        if not os.path.exists(out_path):
            create_file(out_path, n_0, n_1, n_2, x, y, z)
        for f in field_keys:
            f_data_3d = np.empty((n_0, n_1, n_2), dtype=np.double, order='c')
            for r in ranks:
                if r[-3:] == '.nc':
                    file_path = os.path.join(args.fields_dir, d, r)
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

                    combine3d.to_3d(
                        f_data, nl_0, nl_1, nl_2, indx_lo_0, indx_lo_1, indx_lo_2, f_data_3d)

                    rootgrp.close()
            write_field(out_path, f, f_data_3d)

    return


def create_file(fname, nx, ny, nz, x, y, z):
    rootgrp = nc.Dataset(fname, 'w', format='NETCDF4')
    fieldgrp = rootgrp.createGroup('fields')
    fieldgrp.createDimension('nx', nx)
    fieldgrp.createDimension('ny', ny)
    fieldgrp.createDimension('nz', nz)

    xh = fieldgrp.createVariable('x', 'f8', ('nx',))
    yh = fieldgrp.createVariable('y', 'f8', ('ny',))
    zh = fieldgrp.createVariable('z', 'f8', ('nz',))

    xh[:] = x
    yh[:] = y
    zh[:] = z


    rootgrp.close()
    return


def write_field(fname, f, data):
    rootgrp = nc.Dataset(fname, 'r+')
    fields = rootgrp.groups['fields']
    var = fields.createVariable(f, 'f8', ('nx', 'ny', 'nz'))
    var[:, :, :] = data

    rootgrp.close()

if __name__ == "__main__":
    main()
