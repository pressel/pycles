import numpy as np
import os
import shutil
import glob
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
            self.delete_old = namelist['restart']['delete_old']
        except:
            self.delete_old = False
        try:
            times_retained_raw = namelist['restart']['times_retained']
            if not type(times_retained_raw) == list:
                times_retained_raw = [times_retained_raw]

            # Make sure the times are strings, ie '3600' not 3600
            self.times_retained = []
            for time in times_retained_raw:
                self.times_retained.append(str(time))
        except:
            self.times_retained = []

        try:
            if namelist['restart']['init_from']:
                self.input_path = str(namelist['restart']['input_path'])
                self.is_restart_run = True
                Pa.root_print('This run is restarting from data :' + self.input_path )
            else:
                Pa.root_print('Not a restarted simulation.')
        except:
                Pa.root_print('Not a restarted simulation.')




        return


    cpdef initialize(self):
        self.restart_data = {}
        self.last_restart_time = 0.0


        return


    cpdef write(self, ParallelMPI.ParallelMPI Pa):

        self.restart_data['last_restart_time'] = self.last_restart_time
        #Set up path for writing restart files
        path = self.restart_path + '/' + str(np.int(self.last_restart_time))


        # Some preliminary renaming of directories if we are using the 'delete_old' option
        if self.delete_old and Pa.rank == 0:
            recent_dirs = glob.glob(self.restart_path +'/*_recent')
            for recent_dir in recent_dirs:
                prefix = recent_dir[:-7]
                os.rename(recent_dir, prefix+'_old')
            new_dirs = glob.glob(self.restart_path +'/*_new')
            for new_dir in new_dirs:
                prefix = new_dir[:-4]
                os.rename(new_dir, prefix+'_recent')
        if self.delete_old:
            path = path +  '_new'

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

        if self.delete_old and Pa.rank == 0:
            old_dirs = glob.glob(self.restart_path +'/*_old')
            for old_dir in old_dirs:
                trim_prefix = old_dir[len(self.restart_path)+1:-4]
                if trim_prefix in self.times_retained:
                    os.rename(old_dir, self.restart_path+'/'+trim_prefix)
                else:
                    shutil.rmtree(old_dir)


        return

    cpdef read(self, ParallelMPI.ParallelMPI Pa):

        with open(self.input_path + '/' + str(Pa.rank) + '.pkl', 'rb') as f:
            self.restart_data = pickle.load(f)
        Pa.barrier()

        # We rename the input directory in case it ends in one of the suffixes
        # that is used to find files for deletion
        if Pa.rank == 0:
            if self.delete_old:
                os.rename(self.input_path, self.input_path +'_original')


        return


    cpdef free_memory(self):
        '''
        Free memory associated with restart_data dictionary.
        :return:
        '''

        self.restart_data = {}

        return

    cpdef cleanup(self):

        path = self.restart_path
        originals = glob.glob(path+'/*_original')

        for original in originals:
            prefix = original[:-9]
            os.rename(original, prefix)
        recents = glob.glob(path +'/*_recent')

        for recent in recents:
            prefix = recent[:-7]
            os.rename(recent, prefix)
        new_dirs = glob.glob(path +'/*_new')

        for new_dir in new_dirs:
            prefix = new_dir[:-4]
            os.rename(new_dir, prefix)
        return

