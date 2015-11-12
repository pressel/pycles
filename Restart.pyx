import numpy as np
import os
try:
    import cPickle as pickle
except:
    import pickle as pickle # for Python 3 users

cimport ParallelMPI

cdef class Restart:
    def __init__(self, dict namelist, ParallelMPI.ParallelMPI Pa):
        '''
        Init method for Restart class. Take the namelist dictionary as an argument and determines the output path
        for the restart files. If one cannot be constructed from the namelist information the restart files are placed
        into the directory containing main.py. The uuid is also stored to make sure the restart files remain unique.

        :param namelist:
        :return:
        '''

        self.uuid = str(namelist['meta']['uuid'])

        try:
            outpath = str(os.path.join(str(namelist['output']['output_root'])
                                   + 'Output.' + str(namelist['meta']['simname']) + '.' + self.uuid[-5:]))
            self.restart_path = os.path.join(outpath, 'Restart')
        except:
            self.restart_path = './restart.' + self.uuid[-5:]

        if Pa.rank == 0:
            try:
                os.mkdir(outpath)
            except:
                pass
            try:
                os.mkdir(self.restart_path)
            except:
                pass


        try:
            self.frequency = namelist['restart']['frequency']
        except:
            self.frequency = 30.0

        try:
            if namelist['restart']['init_from']:
                self.input_path = str(namelist['restart']['input_path'])
                self.is_restart_run = True
            Pa.root_print('This is a restart run!')
        except:
                Pa.root_print('Not a restart run!')


        return


    cpdef initialize(self):
        self.restart_data = {}
        self.last_restart_time = 0.0


        return


    cpdef write(self, ParallelMPI.ParallelMPI Pa):

        self.restart_data['last_restart_time'] = self.last_restart_time
        #Set up path for writing restar files
        path = self.restart_path + '/' + str(np.int(self.last_restart_time + self.frequency))

        if Pa.rank == 0:
            if os.path.exists(path):
                Pa.root_print("Restart path exits for safety not overwriting.")
                self.free_memory()
                return
            else:
                Pa.root_print("Creating directory: " +  path + " for restart files.")
                os.mkdir(path)
        Pa.barrier()

        with open(path+ '/' + str(Pa.rank) + '.pkl', 'wb') as f:
            pickle.dump(self.restart_data, f,protocol=2)

        # No point keeping data in dictionary so empty it now
        self.free_memory()

        return

    cpdef read(self, ParallelMPI.ParallelMPI Pa):

        with open(self.input_path + '/' + str(Pa.rank) + '.pkl', 'rb') as f:
            self.restart_data = pickle.load(f)

        return


    cpdef free_memory(self):
        '''
        Free memoery associated with restart_data dictionary.
        :return:
        '''

        self.restart_data = {}

        return
