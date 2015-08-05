import netCDF4
import os
cimport ParallelMPI

cdef class NetCDFIO:
    def __init__(self):
        return

    cpdef initialize(self,dict namelist, ParallelMPI.ParallelMPI Pa):

        #Store the path to the stats file if it doesn't exist then make the directory
        try:
            self.stats_path = namelist['io']['stats_path']
        except:
            self.stats_path = './output/'

        if self.stats_path[-1] != '/':
            self.stats_path += '/'

        #create stats path if it doesn't exit
        if not os.path.exists(self.stats_path):
            Pa.root_print(self.stats_path +' does not exit so creating it.')
            os.mkdir(self.stats_path)
        else:
            Pa.root_print(self.stats_path +' exits.')

        #Generate the file name
        try:
            self.stats_file_name = str(namelist['meta']['simname']) + '.stats.nc'
        except:
            Pa.root_print('Simname not given in namelist')
            Pa.root_print('Killing simualtion now!')
            Pa.kill()


        #Read in the unique identifier
        try:
            self.uuid = str(namelist['meta']['uuid'])
            print('Simulation uuid: ' + self.uuid)
        except:
            print('No uuid given in namelist!')
            print('Killing simulation now!')
            Pa.kill()

        #self.setup_stats_file(Pa)

        #import sys; sys.exit()

        return

    cpdef setup_stats_file(self, ParallelMPI.ParallelMPI Pa):

        if Pa.rank == 0:
            nc_file = netCDF4.Dataset(self.stats_path + self.stats_file_name,
                                      'w',format='NETCDF4')

            #Create netcdf groups in the statistics output file
            meta_group = nc_file.createGroup('Meta')
            nc_file.createGroup('ReferenceState')
            nc_file.createGroup('ProfileStats')
            nc_file.createGroup('TimeSeries')

            print len(self.uuid)
            uuid = meta_group.createVariable('UUID','S1',(len(self.uuid),))
            #uuid[:] = self.uuid
            nc_file.close()

        return


